package is.hail.stats

import breeze.linalg.{*, DenseMatrix, DenseVector}
import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.distributedmatrix.{BlockMatrixIsDistributedMatrix, DistributedMatrix}
import is.hail.distributedmatrix.DistributedMatrix.implicits._
import is.hail.expr.{TString, TVariant, Type}
import is.hail.utils._
import is.hail.variant.{Variant, VariantDataset}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.json4s.jackson

case class Eigen(rowSignature: Type, rowIds: Array[Annotation], evects: DenseMatrix[Double], evals: DenseVector[Double]) {
  require(evects.rows == rowIds.length)
  require(evects.cols == evals.length)
    
  def nEvects: Int = evals.length
  
  def filterRows(signature: Type, pred: (Annotation => Boolean)): Eigen = {
    require(signature == rowSignature)
        
    val (newRowIds, newRows) = rowIds.zipWithIndex.filter{ case (id, row) => pred(id) }.unzip
    val newEvects = evects.filterRows(newRows.toSet).getOrElse(fatal("Filtering would remove all rows from eigenvectors"))
    
    Eigen(rowSignature, newRowIds, newEvects, evals)
  }
  
  def takeTop(k: Int): Eigen = {
    if (k < 1)
      fatal(s"k must be a positive integer, got $k")
    else if (k >= nEvects)
      this
    else
      Eigen(rowSignature, rowIds,
        evects(::, (nEvects - k) until nEvects).copy, evals((nEvects - k) until nEvects).copy)
  }
  
  def evectsSpark(): linalg.DenseMatrix = new linalg.DenseMatrix(evects.rows, evects.cols, evects.data, evects.isTranspose)
  
  def evalsArray(): Array[Double] = evals.toArray
  
  def distribute(sc: SparkContext): EigenDistributed = {
    val U = BlockMatrixIsDistributedMatrix.from(sc, evects.asSpark(), 1024, 1024)
    EigenDistributed(rowSignature, rowIds, U, evals)
  }
  
  def toEigenDistributedRRM(vds: VariantDataset, nSamplesInLDMatrix: Int): EigenDistributed = {
    if (rowSignature != TVariant)
      fatal(s"Rows must have type TVariant, got $rowSignature")

    val variants = rowIds.map(_.asInstanceOf[Variant])    
    val variantSet = variants.toSet
    val nEigs = evals.length
    
    info(s"Transforming $nEigs variant eigenvectors to sample eigenvectors...")

    // G = normalized genotype matrix (n samples by m variants)
    //   = U * sqrt(S) * V.t
    // U = G * V * inv(sqrt(S))
    // L = 1 / n * G.t * G = V * S_L * V.t
    // K = 1 / m * G * G.t = U * S_K * U.t
    // S_K = S_L * n / m
    // S = S_K * m
    
    val n = nSamplesInLDMatrix.toDouble
    val m = variants.length
    val V = evects
    val S_K = evals :* n / m
    val c2 = 1.0 / math.sqrt(m)
    val sqrtSInv = S_K.map(e => c2 / math.sqrt(e))

    var filteredVDS = vds.filterVariants((v, _, _) => variantSet(v))
    filteredVDS = filteredVDS.persist()
    require(filteredVDS.variants.count() == variantSet.size, "Some variants in LD matrix eigendecomposition are missing from VDS")

    val dm = DistributedMatrix[BlockMatrix]
    import dm.ops._

    val G = ToNormalizedIndexedRowMatrix(filteredVDS).toBlockMatrixDense().t 
    val U = G * (V(* , ::) :* sqrtSInv).asSpark()
   
    filteredVDS.unpersist()

    EigenDistributed(vds.sSignature, vds.sampleIds.toArray, U, S_K)
  }
  
  import Eigen._
  
  def write(hc: HailContext, uri: String) {
    if (rowSignature != TString)
      fatal(s"In order to write, rows must have schema TString, got $rowSignature") // FIXME
    
    val hadoop = hc.sc.hadoopConfiguration
    hadoop.mkDir(uri)

    hadoop.writeDataFile(uri + evectsRelativePath) { os =>
      val evectsData = evects.toArrayShallow
      var i = 0
      while (i < evectsData.length) {
        os.writeDouble(evectsData(i))
        i += 1
      }
    }

    hadoop.writeDataFile(uri + evalsRelativePath) { os =>
      val evalsData = evals.toArray
      var i = 0
      while (i < evalsData.length) {
        os.writeDouble(evalsData(i))
        i += 1
      }
    }
    
    hadoop.writeDataFile(uri + metadataRelativePath) { os =>
      jackson.Serialization.write(
        EigenMetadata(rowIds.length, evals.length, rowIds.map(_.toString())),
          os)
    }
  }
}

object Eigen {
  def apply(rowSignature: Type, rowIds: Array[Annotation], X: DenseMatrix[Double], optRankBound: Option[Int]): Eigen = {
    val n = rowIds.length
    require(n == X.rows && n == X.cols)
    
    info(s"Computing eigenvectors...")
    val eig = printTime(eigSymD(X))
    
    val nEigs = optRankBound.map(_ min n).getOrElse(n)
    
    info(s"Eigendecomposition complete, returning $nEigs eigenvectors.")
        
    val (evects, evals) =
      if (nEigs == n)
        (eig.eigenvectors, eig.eigenvalues)
      else
        (eig.eigenvectors(::, (n - nEigs) until n).copy, eig.eigenvalues((n - nEigs) until n).copy)
    
    Eigen(rowSignature, rowIds, evects, evals)
  }
  
  
  private val metadataRelativePath = "/metadata.json"
  private val evectsRelativePath = "/evects"
  private val evalsRelativePath = "/evals"
  
  def read(hc: HailContext, uri: String): Eigen = {
    val hadoop = hc.hadoopConf

    val EigenMetadata(nSamples, nEigs, sampleIds) =
      hadoop.readTextFile(uri + metadataRelativePath) { isr =>
        jackson.Serialization.read[EigenMetadata](isr)
      }

    assert(nSamples == sampleIds.length)
    
    val nEntries = nSamples * nEigs
    val evectsData = Array.ofDim[Double](nEntries)
    val evalsData = Array.ofDim[Double](nEigs)
    
    hadoop.readDataFile(uri + evectsRelativePath) { is =>
      var i = 0
      while (i < nEntries) {
        evectsData(i) = is.readDouble()
        i += 1
      }
    }

    hadoop.readDataFile(uri + evalsRelativePath) { is =>
      var i = 0
      while (i < nEigs) {
        evalsData(i) = is.readDouble()
        i += 1
      }
    }
    
    new Eigen(TString, sampleIds.asInstanceOf[Array[Annotation]], new DenseMatrix[Double](nSamples, nEigs, evectsData), DenseVector(evalsData))
  }
}

case class EigenMetadata(nSamples: Int, nEigs: Int, sampleIds: Array[String])

case class EigenDistributed(rowSignature: Type, rowIds: Array[Annotation], evects: BlockMatrix, evals: DenseVector[Double]) {
  require(evects.numRows() == rowIds.length)
  require(evects.numCols() == evals.length)
  
  def nEvects: Int = evals.length
  
  def evalsArray(): Array[Double] = evals.toArray
    
  def localize(): Eigen = {
    val U = evects.toLocalMatrix().asBreeze().asInstanceOf[DenseMatrix[Double]]
    Eigen(rowSignature, rowIds, U, evals)
  }
}