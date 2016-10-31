package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.spark.mllib.linalg.{Vectors, DenseMatrix => SDenseMatrix}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix, RowMatrix}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{Genotype, VariantDataset}
import com.github.fommil.netlib.BLAS.{getInstance => NativeBLAS}

import scala.collection.mutable

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

class SparseGtBuilder extends Serializable {
  private val missingRowIndices = new mutable.ArrayBuilder.ofInt()
  private val rowsX = new mutable.ArrayBuilder.ofInt()
  private val valsX = new mutable.ArrayBuilder.ofDouble()
  private var row = 0
  private var sparseLength = 0 // current length of rowsX and valsX, used to track missingRowIndices
  private var sumX = 0

  def merge(g: Genotype): SparseGtBuilder = {
    (g.unboxedGT: @unchecked) match {
      case 0 =>
      case 1 =>
        rowsX += row
        valsX += 1d
        sparseLength += 1
        sumX += 1
      case 2 =>
        rowsX += row
        valsX += 2d
        sparseLength += 1
        sumX += 2
      case -1 =>
        missingRowIndices += sparseLength
        rowsX += row
        valsX += 0d // placeholder for meanX
        sparseLength += 1
    }
    row += 1

    this
  }

  def toSparseGtVector(nSamples: Int): (SparseVector[Double], Boolean, Double) = {
    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = nSamples - nMissing
    val rowsXArray = rowsX.result()
    val valsXArray = valsX.result()

    val isConstant = sumX == 0 || sumX == 2 * nPresent || (sumX == nPresent && valsXArray.forall(_ == 1d))

    val meanX = if (nPresent > 0) sumX.toDouble / nPresent else Double.NaN

    missingRowIndicesArray.foreach(valsXArray(_) = meanX)

    // assert(rowsXArray.isIncreasing)

    (new SparseVector(rowsXArray, valsXArray, nSamples), isConstant, meanX)
  }
}


















class SparseIndexGtBuilder extends Serializable {
  private val hetIndices = new mutable.ArrayBuilder.ofInt()
  private val homVarIndices = new mutable.ArrayBuilder.ofInt()
  private val missingIndices = new mutable.ArrayBuilder.ofInt()
  private var row = 0

  def merge(g: Genotype): SparseIndexGtBuilder = {
    (g.unboxedGT: @unchecked) match {
      case 0 =>
      case 1 =>
        hetIndices += row
      case 2 =>
        homVarIndices += row
      case -1 =>
        missingIndices += row
    }
    row += 1

    this
  }

  def toGtIndexArrays(n: Int): SparseIndexGtArrays =
    SparseIndexGtArrays(n, hetIndices.result(), homVarIndices.result(), missingIndices.result())
}

case class SparseIndexGtArrays(n: Int, hetIndices: Array[Int], homVarIndices: Array[Int], missingIndices: Array[Int]) {
  // FIXME: enforce disjointness?

  val isNonConstant: Boolean = {
    val nHomRef = n - missingIndices.size - hetIndices.size - homVarIndices.size

    (nHomRef > 0 && hetIndices.size > 0) ||
      (nHomRef > 0 && homVarIndices.size > 0) ||
      (hetIndices.size > 0 && homVarIndices.size > 0)
  }

  // FIXME:
  val mean = {
    // require(isNonConstant)
    val nPresent = n - missingIndices.size
    if (nPresent == 0) Double.NaN else (hetIndices.size + 2 * homVarIndices.size) / nPresent.toDouble
  }

  def toGtDenseVector: DenseVector[Double] = {
    val data = Array.ofDim[Double](n)

    hetIndices.foreach(data(_) = 1)
    homVarIndices.foreach(data(_) = 2)
    missingIndices.foreach(data(_) = mean)

    DenseVector(data)
  }
}

// assumes column major
object DM_SIA_Eq_DV {
  def apply(dm: DenseMatrix[Double], sia: SparseIndexGtArrays): DenseVector[Double] = {
    // FIXME: relax these requirements
    require(dm.cols == sia.n)
    require(dm.majorStride == dm.rows)
    require(dm.offset == 0)
    require(!dm.isTranspose)

    val cols = dm.cols
    val rows = dm.rows
    val data = dm.data

    val res = Array.ofDim[Double](rows)

    var i = 0
    var j = 0
    var offset = 0

    while (j < sia.missingIndices.size) {
      offset = rows * sia.missingIndices(j)
      i = 0
      while (i < rows) {
        res(i) += data(offset + i)
        i += 1
      }
      j += 1
    }

    i = 0
    while (i < rows) {
      res(i) *= sia.mean
      i += 1
    }

    j = 0
    while (j < sia.hetIndices.size) {
      offset = rows * sia.hetIndices(j)
      i = 0
      while (i < rows) {
        res(i) += data(offset + i)
        i += 1
      }
      j += 1
    }

    j = 0
    while (j < sia.homVarIndices.size) {
      offset = rows * sia.homVarIndices(j)
      i = 0
      while (i < rows) {
        res(i) += 2 * data(offset + i)
        i += 1
      }
      j += 1
    }

    DenseVector(res)
  }
}


// currently not used
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





// FIXME: not working, trying to work in blocks
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