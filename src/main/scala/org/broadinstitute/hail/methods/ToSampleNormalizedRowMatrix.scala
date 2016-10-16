package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.spark.mllib.linalg.{Vectors, SparseVector => SSparseVector, DenseMatrix => SDenseMatrix}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{Genotype, GenotypeStream, Variant, VariantDataset}
import com.github.fommil.netlib.{BLAS => NetlibBLAS, F2jBLAS}
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}

import scala.collection.mutable

object ToSampleNormalizedRowMatrix {
  def apply(vds: VariantDataset): RowMatrix = {
    val n = vds.nSamples
    val rows = vds.rdd.flatMap { case (v, (va, gs)) => ToNormalizedGtArray(gs, n) }.map(Vectors.dense)
    val m = rows.count()
    new RowMatrix(rows, m, n)
  }
}

object ToSampleNormalizedIndexedRowMatrix {
  def apply(vds: VariantDataset): IndexedRowMatrix = {
    val n = vds.nSamples
    val variants = vds.variants.collect()
    val variantIdxBc = vds.sparkContext.broadcast(variants.index)
    val indexedRows = vds.rdd.flatMap{ case (v, (va, gs)) => ToNormalizedGtArray(gs, n).map( a => IndexedRow(variantIdxBc.value(v), Vectors.dense(a))) }
    val m = indexedRows.count()
    new IndexedRowMatrix(indexedRows, m, n)
  }
}

object ComputeGrammian {
  def withGrammian(vds: VariantDataset): DenseMatrix[Double] = {
    val n = vds.nSamples
    val G = ToSampleNormalizedRowMatrix(vds).computeGramianMatrix()
    new DenseMatrix[Double](n, n, G.asInstanceOf[SDenseMatrix].values)
  }

  def withBlock(vds: VariantDataset): DenseMatrix[Double] = {
    val n = vds.nSamples
    val B = ToSampleNormalizedIndexedRowMatrix(vds).toBlockMatrix().cache()
    val G = B.transpose.multiply(B).toLocalMatrix()
    new DenseMatrix[Double](n, n, G.asInstanceOf[SDenseMatrix].values)
  }
}

class SparseGtBuilder extends Serializable {
  private val missingRowIndices = new mutable.ArrayBuilder.ofInt()
  private val rowsX = new mutable.ArrayBuilder.ofInt()
  private val valsX = new mutable.ArrayBuilder.ofDouble()
  private var sparseLength = 0 // length of rowsX and valsX (ArrayBuilder has no length), used to track missingRowIndices
  private var sumX = 0
  private var nHet = 0

  def merge(row: Int, g: Genotype): SparseGtBuilder = {
    g.gt match {
      case Some(0) =>
      case Some(1) =>
        rowsX += row
        valsX += 1d
        sparseLength += 1
        sumX += 1
        nHet += 1
      case Some(2) =>
        rowsX += row
        valsX += 2d
        sparseLength += 1
        sumX += 2
      case None =>
        missingRowIndices += sparseLength
        rowsX += row
        valsX += 0d // placeholder for meanX
        sparseLength += 1
      case _ => throw new IllegalArgumentException("Genotype value " + g.gt.get + " must be 0, 1, or 2.")
    }

    this
  }

  // variant is atomic => combOp merge not called
  def merge(that: SparseGtBuilder): SparseGtBuilder = {
    missingRowIndices ++= that.missingRowIndices.result().map(_ + sparseLength)
    rowsX ++= that.rowsX.result()
    valsX ++= that.valsX.result()
    sparseLength += that.sparseLength
    sumX += that.sumX
    nHet += that.nHet

    this
  }

  def toGtSSparseVector(nSamples: Int): Option[SparseVector[Double]] = {
    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = nSamples - nMissing

    // all HomRef | all Het | all HomVar || all missing
    if (sumX == 0 || nHet == nPresent || sumX == 2 * nPresent || nPresent == 0)
      None
    else {
      val rowsXArray = rowsX.result()
      val valsXArray = valsX.result()
      val meanX = sumX.toDouble / nPresent

      missingRowIndicesArray.foreach(valsXArray(_) = meanX)

      // Variant is atomic => combOp merge not called => rowsXArray is sorted (as expected by SparseVector constructor)
      assert(rowsXArray.isIncreasing)

      Some(new SSparseVector(nSamples, rowsXArray, valsXArray))
    }
  }

  def toGtSparseVector(nSamples: Int): Option[SparseVector[Double]] = {
    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = nSamples - nMissing

    // all HomRef | all Het | all HomVar || all missing
    if (sumX == 0 || nHet == nPresent || sumX == 2 * nPresent || nPresent == 0)
      None
    else {
      val rowsXArray = rowsX.result()
      val valsXArray = valsX.result()
      val meanX = sumX.toDouble / nPresent

      missingRowIndicesArray.foreach(valsXArray(_) = meanX)

      // Variant is atomic => combOp merge not called => rowsXArray is sorted (as expected by SparseVector constructor)
      assert(rowsXArray.isIncreasing)

      Some(new SparseVector(rowsXArray, valsXArray, nSamples))
    }
  }
}

object ToSparseIndexedRowMatrix {
  def apply(vds: VariantDataset): (Array[Variant], IndexedRowMatrix) = {
    val variants = vds.variants.collect()
    val nVariants = variants.length
    val nSamples = vds.nSamples
    val variantIdxBroadcast = vds.sparkContext.broadcast(variants.index)
    val sampleIndexBc = vds.sparkContext.broadcast(vds.sampleIds.zipWithIndex.toMap)

    val mat = vds.aggregateByVariantWithKeys[SparseGtBuilder](new SparseGtBuilder())(
      (sb, v, s, g) => sb.merge(sampleIndexBc.value(s), g),
      (sb1, sb2) => sb1.merge(sb2))
      .flatMap{ case (v, sb) => sb.toGtSSparseVector(nVariants).map(IndexedRow(variantIdxBroadcast.value(v), _)) }

    (variants, new IndexedRowMatrix(mat.cache(), nVariants, nSamples))
  }
}

// These methods filter out constant vectors, but their names don't indicate it
object ToSparseRDD {
  def apply(vds: VariantDataset): RDD[(Variant, Vector[Double])] = {
    val nSamples = vds.nSamples
    val sampleIndexBc = vds.sparkContext.broadcast(vds.sampleIds.zipWithIndex.toMap)

    vds.aggregateByVariantWithKeys[SparseGtBuilder](new SparseGtBuilder())(
      (sb, v, s, g) => sb.merge(sampleIndexBc.value(s), g),
      (sb1, sb2) => sb1.merge(sb2))
      .flatMap { case (v, sb) => sb.toGtSSparseVector(nSamples).map((v, _)) }
  }
}


object ToNormalizedGtArray {
  def apply(gs: Iterable[Genotype], nSamples: Int): Option[Array[Double]] = {
    val (nPresent, gtSum, gtSumSq) = gs.flatMap(_.gt).foldLeft((0, 0, 0))((acc, gt) => (acc._1 + 1, acc._2 + gt, acc._3 + gt * gt))
    val nMissing = nSamples - nPresent
    val allHomRef = gtSum == 0
    val allHet = gtSum == nPresent && gtSumSq == nPresent
    val allHomVar = gtSum == 2 * nPresent

    if (allHomRef || allHomVar || allHet)
      None
    else {
      val gtMean = gtSum.toDouble / nPresent
      val gtSumSqAll = gtSumSq + nMissing * gtMean * gtMean
      val gtSumAll = gtSum + nMissing * gtMean
      val gtNormSqRec = 1d / math.sqrt(gtSumSqAll - gtSumAll * gtSumAll / nSamples)

      Some(gs.map(_.gt.map(g => (g - gtMean) * gtNormSqRec).getOrElse(0d)).toArray)
    }
  }
}

// FIXME: reconcile with logreg copy
object toGtDenseMatrix {
  def apply(gs: Iterable[Genotype]): Option[DenseMatrix[Double]] = {
    val (nPresent, gtSum, allHet) = gs.flatMap(_.nNonRefAlleles).foldLeft((0, 0, true))((acc, gt) => (acc._1 + 1, acc._2 + gt, acc._3 && (gt == 1)))

    if (gtSum == 0 || gtSum == 2 * nPresent || allHet)
      None
    else {
      val gtMean = gtSum.toDouble / nPresent
      val gtArray = gs.map(_.gt.map(_.toDouble).getOrElse(gtMean)).toArray
      Some(new DenseMatrix(gtArray.length, 1, gtArray))
    }
  }
}

// FIXME: not working
object ToRRM {
  def apply(vds: VariantDataset): DenseMatrix[Double] = {
    require(vds.nSamples > 0)

    val N = vds.nSamples
    val triN: Int = if (N % 2 == 0) (N / 2) * (N + 1) else N * ((N + 1) / 2)

    //println(vds.rdd.count())


    val C = vds.rdd
      .mapPartitions { it =>
        val A = new mutable.ArrayBuilder.ofDouble()
        it.foreach { case (v, (va, gs)) =>
          ToNormalizedGtArray(gs, N).foreach(A ++= _)
        }
        Iterator(A.result())
      }
      .treeAggregate(DenseVector[Double](new Array[Double](N * N)))(
        seqOp = (C, A) => {
          if (!A.isEmpty) {
            val UPLO = "U"
            val TRANS = "N"
            val K = A.length / N
            val ALPHA = 1d
            val LDA = N
            val BETA = 0d
            val LDC = N

            println(N, K, A.length)

            // http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_gae0ba56279ae3fa27c75fefbc4cc73ddf.html
            NativeBLAS.dsyrk(UPLO, TRANS, N, K, ALPHA, A, LDA, BETA, C.data, LDC)
          }
          C
        },
        combOp = (C1, C2) => C1 += C2
      )

    copyTriuToTril(N, new DenseMatrix[Double](N, N, C.data))
  }

  def copyTriuToTril(n: Int, C: DenseMatrix[Double]): DenseMatrix[Double] = {
    var i = 0
    var j = 0
    while (i < n) {
      while (j < i) {
        C(i, j) = C(j, i)
        j += 1
      }
      i += 1
    }
    C
  }
}