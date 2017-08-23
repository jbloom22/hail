package is.hail.methods

import is.hail.HailContext
import is.hail.distributedmatrix.{BlockMatrixIsDistributedMatrix, DistributedMatrix}
import is.hail.distributedmatrix.DistributedMatrix.implicits._
import is.hail.stats.RegressionUtils
import is.hail.utils._
import is.hail.stats._
import is.hail.variant.{Variant, VariantDataset}
import breeze.linalg.*
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import org.apache.spark.mllib.linalg.{DenseMatrix => SparkDenseMatrix, DenseVector => SparkDenseVector, Matrix => SparkMatrix, Vectors}
import org.apache.hadoop.io._
import org.json4s._

object LDMatrix {
  /**
    * Computes the LD matrix for the given VDS.
    * @param vds VDS on which to compute Pearson correlation between pairs of variants.
    * @return An LDMatrix.
    */
  def apply(vds : VariantDataset, optComputeLocally: Option[Boolean]): LDMatrix = {
    val nSamples = vds.nSamples
    val nVariants = vds.countVariants()

    val filteredNormalizedHardCalls = vds.rdd.flatMap { 
      case (v, (va, gs)) => RegressionUtils.normalizedHardCalls(gs, nSamples).map(x => (v, x))
    }
    
    val variantsKept = filteredNormalizedHardCalls.map(_._1).collect()
    assert(variantsKept.isSorted, "ld_matrix: Array of variants is not sorted. This is a bug.")

    val normalizedIndexedRows = filteredNormalizedHardCalls.map(_._2).zipWithIndex()
      .map{ case (values, idx) => IndexedRow(idx, Vectors.dense(values))}
    val normalizedBlockMatrix = new IndexedRowMatrix(normalizedIndexedRows).toBlockMatrixDense()

    val nVariantsKept = variantsKept.length
    val nVariantsDropped = nVariants - nVariantsKept

    info(s"Computing LD matrix with ${variantsKept.length} variants using $nSamples samples. $nVariantsDropped variants were dropped.")

    val localBound = 5000 * 5000
    val nEntries: Long = nVariantsKept * nSamples
    val nSamplesInverse = 1.0 / nSamples

    val computeLocally = optComputeLocally.getOrElse(nEntries <= localBound)

    var indexedRowMatrix: IndexedRowMatrix = null

    if (computeLocally) {
      val localMat: SparkDenseMatrix = normalizedBlockMatrix.toLocalMatrix().asInstanceOf[SparkDenseMatrix]
      val product = localMat multiply localMat.transpose
      indexedRowMatrix =
        BlockMatrixIsDistributedMatrix.from(vds.sparkContext, product, normalizedBlockMatrix.rowsPerBlock,
          normalizedBlockMatrix.colsPerBlock).toIndexedRowMatrix()
    } else {
      import is.hail.distributedmatrix.DistributedMatrix.implicits._
      val dm = DistributedMatrix[BlockMatrix]
      import dm.ops._
      indexedRowMatrix = (normalizedBlockMatrix * normalizedBlockMatrix.t)
        .toIndexedRowMatrix()
    }

    val scaledIndexedRowMatrix = new IndexedRowMatrix(indexedRowMatrix.rows
      .map{case IndexedRow(idx, vals) => IndexedRow(idx, vals.map(d => d * nSamplesInverse))})
    
    LDMatrix(scaledIndexedRowMatrix, variantsKept, nSamples)
  }

  private val metadataRelativePath = "/metadata.json"
  private val matrixRelativePath = "/matrix"
  def read(hc: HailContext, uri: String): LDMatrix = {
    val hadoop = hc.hadoopConf
    hadoop.mkDir(uri)

    val rdd = hc.sc.sequenceFile[LongWritable, ArrayPrimitiveWritable](uri+matrixRelativePath).map { case (lw, apw) =>
      IndexedRow(lw.get(), new SparkDenseVector(apw.get().asInstanceOf[Array[Double]]))
    }

    val LDMatrixMetadata(variants, nSamples) =
      hadoop.readTextFile(uri+metadataRelativePath) { isr =>
        jackson.Serialization.read[LDMatrixMetadata](isr)
      }

    new LDMatrix(new IndexedRowMatrix(rdd), variants, nSamples)
  }
}

/**
  *
  * @param matrix Spark IndexedRowMatrix. Entry (i, j) encodes the r value between variants i and j.
  * @param variants Array of variants indexing the rows and columns of the matrix.
  * @param nSamplesUsed Number of samples used to compute this matrix.
  */
case class LDMatrix(matrix: IndexedRowMatrix, variants: Array[Variant], nSamplesUsed: Int) {
  import LDMatrix._

  def toLocalMatrix: SparkMatrix = {
    matrix.toBlockMatrixDense().toLocalMatrix()
  }

  def write(uri: String) {
    val hadoop = matrix.rows.sparkContext.hadoopConfiguration
    hadoop.mkDir(uri)

    matrix.rows.map { case IndexedRow(i, v) => (new LongWritable(i), new ArrayPrimitiveWritable(v.toArray)) }
      .saveAsSequenceFile(uri+matrixRelativePath)

    hadoop.writeTextFile(uri+metadataRelativePath) { os =>
      jackson.Serialization.write(
        LDMatrixMetadata(variants, nSamplesUsed),
        os)
    }
  }
  
    def eigenRRM(vds: VariantDataset, optNEigs: Option[Int]): Eigen = {
    val variantSet = variants.toSet

    val maxRank = variants.length min nSamplesUsed
    val nEigs = optNEigs.getOrElse(maxRank)
    optNEigs.foreach( k => if (k > nEigs) info(s"Requested $k evects but maximum rank is $maxRank.") )

    if (nEigs.toLong * vds.nSamples > Integer.MAX_VALUE)
      fatal(s"$nEigs eigenvectors times ${vds.nSamples} samples exceeds 2^31 - 1, the maximum size of a local matrix.")
    
    val L = matrix.toLocalMatrix().asBreeze().toDenseMatrix

    info(s"Computing eigenvectors of LD matrix...")
    val eigL = printTime(eigSymD(L))
    
    info(s"Transforming $nEigs variant eigenvectors to sample eigenvectors...")

    // G = normalized genotype matrix (n samples by m variants)
    //   = U * sqrt(S) * V.t
    // U = G * V * inv(sqrt(S))
    // L = 1 / n * G.t * G = V * S_L * V.t
    // K = 1 / m * G * G.t = U * S_K * U.t
    // S_K = S_L * n / m
    // S = S_K * m

    val n = nSamplesUsed.toDouble
    val m = variants.length
    assert(m == eigL.eigenvectors.cols)
    val V = eigL.eigenvectors(::, (m - nEigs) until m)
    val S_K =
      if (nEigs == m)
        eigL.eigenvalues :* (n / m)
      else
        (eigL.eigenvalues((m - nEigs) until m) :* (n / m)).copy
      
    val c2 = 1.0 / math.sqrt(m)
    val sqrtSInv = S_K.map(e => c2 / math.sqrt(e))

    var filteredVDS = vds.filterVariants((v, _, _) => variantSet(v))
    filteredVDS = filteredVDS.persist()
    require(filteredVDS.variants.count() == variantSet.size, "Some variants in LD matrix are missing from VDS")

    // FIXME Clean up this ugliness. Unnecessary back and forth from Breeze to Spark. (Might just need to allow multiplying block matrix by local Breeze matrix.
    val VS = V(* , ::) :* sqrtSInv
    val VSSpark = new SparkDenseMatrix(VS.rows, VS.cols, VS.data, VS.isTranspose)

    import is.hail.distributedmatrix.DistributedMatrix.implicits._
    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._

    val sparkG = ToNormalizedIndexedRowMatrix(filteredVDS).toBlockMatrixDense().t
    val sparkU = (sparkG * VSSpark).toLocalMatrix()
    val U = sparkU.asBreeze().toDenseMatrix
    
    filteredVDS.unpersist()
    
    Eigen(vds.sSignature, vds.sampleIds.toArray, U, S_K)
  }
  
  def eigenDistributedRRM(vds: VariantDataset, optNEigs: Option[Int]): EigenDistributed = {
    val variantSet = variants.toSet

    val maxRank = variants.length min nSamplesUsed
    val nEigs = optNEigs.getOrElse(maxRank)
    optNEigs.foreach( k => if (k > nEigs) info(s"Requested $k evects but maximum rank is $maxRank.") )

    if (nEigs.toLong * vds.nSamples > Integer.MAX_VALUE)
      fatal(s"$nEigs eigenvectors times ${vds.nSamples} samples exceeds 2^31 - 1, the maximum size of a local matrix.")
    
    val L = matrix.toLocalMatrix().asBreeze().toDenseMatrix

    info(s"Computing eigenvectors of LD matrix...")
    val eigL = printTime(eigSymD(L))
    
    info(s"Transforming $nEigs variant eigenvectors to sample eigenvectors...")

    // G = normalized genotype matrix (n samples by m variants)
    //   = U * sqrt(S) * V.t
    // U = G * V * inv(sqrt(S))
    // L = 1 / n * G.t * G = V * S_L * V.t
    // K = 1 / m * G * G.t = U * S_K * U.t
    // S_K = S_L * n / m
    // S = S_K * m

    val n = nSamplesUsed.toDouble
    val m = variants.length
    assert(m == eigL.eigenvectors.cols)
    val V = eigL.eigenvectors(::, (m - nEigs) until m)
    val S_K =
      if (nEigs == m)
        eigL.eigenvalues :* (n / m)
      else
        (eigL.eigenvalues((m - nEigs) until m) :* (n / m)).copy
      
    val c2 = 1.0 / math.sqrt(m)
    val sqrtSInv = S_K.map(e => c2 / math.sqrt(e))

    var filteredVDS = vds.filterVariants((v, _, _) => variantSet(v))
    filteredVDS = filteredVDS.persist()
    require(filteredVDS.variants.count() == variantSet.size, "Some variants in LD matrix are missing from VDS")

    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._

    val VS = V(* , ::) :* sqrtSInv
    
    val G = ToNormalizedIndexedRowMatrix(filteredVDS).toBlockMatrixDense().t 
    val U = G * VS.asSpark()
   
    filteredVDS.unpersist()

    EigenDistributed(vds.sSignature, vds.sampleIds.toArray, U, S_K)
  }
}

case class LDMatrixMetadata(variants: Array[Variant], nSamples: Int)
