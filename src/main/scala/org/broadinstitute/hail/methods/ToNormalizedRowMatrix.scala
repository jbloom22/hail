package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.spark.mllib.linalg.{Vectors, SparseVector => SSparseVector, DenseMatrix => SDenseMatrix}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{Genotype, Variant, VariantDataset}
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}

import scala.collection.mutable

object ToNormalizedRowMatrix {
  def apply(vds: VariantDataset): RowMatrix = {
    val n = vds.nSamples
    val rows = vds.rdd.flatMap { case (v, (va, gs)) => ToNormalizedGtArray(gs, n) }.map(Vectors.dense)
    val m = rows.count()
    new RowMatrix(rows, m, n)
  }
}

object ToNormalizedIndexedRowMatrix {
  def apply(vds: VariantDataset): IndexedRowMatrix = {
    val n = vds.nSamples
    val variants = vds.variants.collect()
    val variantIdxBc = vds.sparkContext.broadcast(variants.index)
    val indexedRows = vds.rdd.flatMap { case (v, (va, gs)) => ToNormalizedGtArray(gs, n).map(a => IndexedRow(variantIdxBc.value(v), Vectors.dense(a))) }
    val m = indexedRows.count()
    new IndexedRowMatrix(indexedRows, m, n)
  }
}

object ComputeGrammian {
  def apply(A: RowMatrix): DenseMatrix[Double] = {
    val n = A.numCols().toInt
    val G = A.computeGramianMatrix().asInstanceOf[SDenseMatrix].values
    new DenseMatrix[Double](n, n, G)
  }

  def apply(A: IndexedRowMatrix): DenseMatrix[Double] = {
    val n = A.numCols().toInt
    val B = A.toBlockMatrix().cache()
    val G = B.transpose.multiply(B).toLocalMatrix().asInstanceOf[SDenseMatrix].values
    B.blocks.unpersist()
    new DenseMatrix[Double](n, n, G)
  }
}

object ComputeRRM {
  def withoutBlocks(vds: VariantDataset): (DenseMatrix[Double], Int) = {
    val A = ToNormalizedRowMatrix(vds)
    val mRec = 1d / A.numRows()
    (ComputeGrammian(A) :* mRec, A.numRows().toInt)
  }

  def withBlocks(vds: VariantDataset): (DenseMatrix[Double], Int) = {
    val A = ToNormalizedIndexedRowMatrix(vds)
    val mRec = 1d / A.numRows()
    (ComputeGrammian(A) :* mRec, A.numRows().toInt)
  }
}

// FIXME: remove row input, integrate sample mask
class SparseGtBuilder extends Serializable {
  private val missingRowIndices = new mutable.ArrayBuilder.ofInt()
  private val rowsX = new mutable.ArrayBuilder.ofInt()
  private val valsX = new mutable.ArrayBuilder.ofDouble()
  private var sparseLength = 0
  // length of rowsX and valsX (ArrayBuilder has no length), used to track missingRowIndices
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

  def toGtSparseVector(nSamples: Int): Option[SparseVector[Double]] = {
    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = nSamples - nMissing

    // all HomRef | all Het | all HomVar || all missing
    if (sumX == 0 || nHet == nPresent || sumX == 2 * nPresent)
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

// mean centered and variance normalized
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
      val gtMeanAll = (gtSum + nMissing * gtMean) / nSamples
      val gtMeanSqAll = (gtSumSq + nMissing * gtMean * gtMean) / nSamples
      val gtStdDevRec = 1d / math.sqrt(gtMeanSqAll - gtMeanAll * gtMeanAll)

      if (gtStdDevRec.isNaN())
        println(gtSum, gtSumSq, gtMeanSqAll, gtMeanAll * gtMeanAll, gs.map(_.gt.get))

      Some(gs.map(_.gt.map(g => (g - gtMean) * gtStdDevRec).getOrElse(0d)).toArray)
    }
  }
}

object ToGtDenseVector {
  def apply(gs: Iterable[Genotype], sampleMask: Array[Boolean], nSamples: Int): Option[DenseVector[Double]] = {
    val gtArray = new Array[Double](nSamples)
    val missingRowIndices = new mutable.ArrayBuilder.ofInt()
    var row = 0
    var gtSum = 0

    gs.iterator.zipWithIndex.foreach { case (g, i) =>
      if (sampleMask(i)) {
        val gt = g.unboxedGT
        if (gt != -1) {
          gtArray(row)
          gtSum += gt
        } else
          missingRowIndices += row
        row += 1
      }
    }

    val missingRowIndicesArray = missingRowIndices.result()
    val nPresent = nSamples - missingRowIndicesArray.length
    val allHet =
      if (gtSum != nPresent)
        false
      else
        gtArray.forall(_ == 1d)

    if (gtSum == 0 || gtSum == 2 * nPresent || allHet)
      None
    else {
      val gtMean = gtSum.toDouble / nPresent
      missingRowIndicesArray.foreach(gtArray(_) = gtMean)

      Some(new DenseVector(gtArray))
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