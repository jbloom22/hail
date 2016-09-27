package org.broadinstitute.hail.stats

import breeze.linalg._
import org.apache.spark.mllib.linalg.{Matrices, DenseMatrix => SDenseMatrix}
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.methods.{ToIndexedRowMatrix, ToStandardizedIndexedRowMatrix}
import org.broadinstitute.hail.variant.{Variant, VariantDataset}
import org.broadinstitute.hail.utils._

object LMM {
  def applyVds(vds: VariantDataset,
    filtTest: Variant => Boolean,
    filtGRM: Variant => Boolean,
    C: DenseMatrix[Double],
    y: DenseVector[Double],
    k: Int,
    optDelta: Option[Double] = None,
    useREML: Boolean = true): LMMResult = {

    val Wt = ToStandardizedIndexedRowMatrix(vds.filterVariants((v, va, gs) => filtGRM(v)))._2 // W is samples by variants, Wt is variants by samples
    val (variants, genotypes) = ToIndexedRowMatrix(vds.filterVariants((v, va, gs) => filtTest(v)))

    LMM(Wt, variants, genotypes, C, y, k, optDelta)
  }

  def apply(Wt: IndexedRowMatrix,
    variants: Array[Variant],
    genotypes: IndexedRowMatrix,
    C: DenseMatrix[Double],
    y: DenseVector[Double],
    k: Int,
    optDelta: Option[Double] = None,
    useREML: Boolean = true): LMMResult = {

    require(k <= Wt.numRows())
    require(k <= Wt.numCols())

    val svd: SingularValueDecomposition[IndexedRowMatrix, org.apache.spark.mllib.linalg.Matrix] = Wt.computeSVD(k)

    val U = svd.V // W = svd.V * svd.s * svd.U.t
    val UB = new DenseMatrix[Double](svd.V.numRows, svd.V.numCols, svd.V.toArray)

    // println(UB.rows, UB.cols)

    val s = toBDenseVector(svd.s.toDense)
    val SB =  s :* s // K = U * (svd.s * svd.s) * V.t

    // SB.foreach(println)

    val diagLMM = DiagLMM(UB.t * C, UB.t * y, SB, optDelta, useREML)

    val diagLMMBc = genotypes.rows.sparkContext.broadcast(diagLMM)

    // FIXME: variant names are already local, we shouldn't be gathering them all, right?!

    val lmmResult = genotypes.multiply(U).rows.map(r =>
      (variants(r.index.toInt), diagLMMBc.value.likelihoodRatioTest(toBDenseVector(r.vector.toDense))))

    LMMResult(diagLMM, lmmResult)
  }
}

object DiagLMM {
  def apply(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], optDelta: Option[Double] = None, useREML: Boolean = true): DiagLMM = {
    require(C.rows == y.length)

    val delta = optDelta.getOrElse(deltaGridVals(C, y, S, useREML).minBy(_._2)._1)

    val n = y.length
    val D = S + delta
    val dy = y :/ D
    val dC = C(::, *) :/ D
    val ydy = y dot dy
    val Cdy = C.t * dy
    val CdC = C.t * dC
    val b = CdC \ Cdy
    val s2 = (ydy - (Cdy dot b)) / (if (useREML) n - C.cols else n)

    DiagLMM(C, y, dy, ydy, b, s2, math.log(s2), delta, D.map(1 / _), useREML)
  }

  def deltaGridVals(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], useREML: Boolean): IndexedSeq[(Double, Double)] = {

    val logmin = -10.0
    val logmax = 10.0
    val logres = 0.1

    val grid = (logmin to logmax by logres).map(math.exp)

    val n = y.length
    val c = C.cols

    // up to constant shift and scale by 2
    def negLogLkhd(delta: Double, useREML: Boolean): Double = {
      val D = S + delta
      val dy = y :/ D
      val ydy = y dot dy
      val Cdy = C.t * dy
      val CdC = C.t * (C(::, *) :/ D)
      val b = CdC \ Cdy
      val r = ydy - (Cdy dot b)

      if (useREML)
        sum(breeze.numerics.log(D)) + (n - c) * math.log(r) + logdet(CdC)._2
      else
        sum(breeze.numerics.log(D)) + n * math.log(r)
    }

    grid.map(delta => (delta, negLogLkhd(delta, useREML)))
  }
}

case class DiagLMM(
  C: DenseMatrix[Double],
  y: DenseVector[Double],
  dy: DenseVector[Double],
  ydy: Double,
  nullB: DenseVector[Double],
  nullS2: Double,
  logNullS2: Double,
  delta: Double,
  invD: DenseVector[Double],
  useREML: Boolean) {

  def likelihoodRatioTest(x: DenseVector[Double]): LMMStat = {
    require(x.length == y.length)

    val n = x.length
    val X = DenseMatrix.horzcat(x.asDenseMatrix.t, C)

    val xdx = X.t * (X(::, *) :* invD)
    val xdy = X.t * dy

    val b = xdx \ xdy
    val s2 = (ydy - (xdy dot b)) / (if (useREML) n - C.cols else n)
    val chi2 = n * (logNullS2 - math.log(s2))
    val p = chiSquaredTail(1, chi2)

    LMMStat(b, s2, chi2, p)
  }
}

case class LMMStat(b: DenseVector[Double], s2: Double, chi2: Double, p: Double)

case class LMMResult(diagLMM: DiagLMM, stats: RDD[(Variant, LMMStat)])


object LMMLowRank {
  def applyVds(vds: VariantDataset,
    filtTest: Variant => Boolean,
    filtGRM: Variant => Boolean,
    C: DenseMatrix[Double],
    y: DenseVector[Double],
    k: Int,
    optDelta: Option[Double] = None,
    useREML: Boolean = true): LMMResultLowRank = {

    val Wt = ToStandardizedIndexedRowMatrix(vds.filterVariants((v, va, gs) => filtGRM(v)))._2 // W is samples by variants, Wt is variants by samples
    val (variants, genotypes) = ToIndexedRowMatrix(vds.filterVariants((v, va, gs) => filtTest(v)))

    LMMLowRank(Wt, variants, genotypes, C, y, k, optDelta, useREML)
  }

  def apply(Wt: IndexedRowMatrix,
    variants: Array[Variant],
    genotypes: IndexedRowMatrix,
    C: DenseMatrix[Double],
    y: DenseVector[Double],
    k: Int,
    optDelta: Option[Double] = None,
    useREML: Boolean = true): LMMResultLowRank = {

    require(k <= Wt.numRows())
    require(k <= Wt.numCols())

    val n = y.length

    val svd: SingularValueDecomposition[IndexedRowMatrix, org.apache.spark.mllib.linalg.Matrix] = Wt.computeSVD(k)

    val U = svd.V // W = svd.V * svd.s * svd.U.t
    val UB = new DenseMatrix[Double](svd.V.numRows, svd.V.numCols, svd.V.toArray)

    // println(UB.rows, UB.cols)

    val s = toBDenseVector(svd.s.toDense)
    val SB = s :* s // K = U * (svd.s * svd.s) * V.t, s has length k

    // SB.foreach(println)

    val Cp = UB.t * C
    val yp = UB.t * y
    val CpC = (C.t * C) - (Cp.t * Cp)
    val Cpy = (C.t * y) - (Cp.t * yp)
    val ypy = (y dot y) - (yp dot yp)

    val diagLMMLowRank = DiagLMMLowRank(n, Cp, yp, SB, ypy, Cpy, CpC, optDelta, useREML)

    val diagLMMLowRankBc = genotypes.rows.sparkContext.broadcast(diagLMMLowRank)

    // FIXME: variant names are already local, we shouldn't be gathering them all, right?!

    // adding in y
    val U_y = Matrices.horzcat(Array(U, new SDenseMatrix(y.length, 1, y.toArray)))

    val lmmResultLowRank = genotypes.multiply(U_y).rows.map { r =>
      val result = toBDenseVector(r.vector.toDense)
      (variants(r.index.toInt), diagLMMLowRankBc.value.likelihoodRatioTest(result(0 until k), result(k)))
    }

    LMMResultLowRank(diagLMMLowRank, lmmResultLowRank)
  }
}

object DiagLMMLowRank {
  def apply(n: Int, C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], ypy: Double, Cpy: DenseVector[Double], CpC: DenseMatrix[Double], optDelta: Option[Double] = None, useREML: Boolean = true): DiagLMMLowRank = {
    require(C.rows == y.length)

    val delta = optDelta.getOrElse(deltaGridVals(n, C, y, S, ypy, Cpy, CpC, useREML).minBy(_._2)._1)

    val D = S + delta
    val dy = y :/ D
    val dC = C(::, *) :/ D
    val ydy = y dot dy
    val Cdy = C.t * dy
    val CdC = C.t * dC
    val b = (CdC + CpC / delta) \ (Cdy + Cpy / delta)
    val r1 = ydy - (Cdy dot b)
    val r2 = (ypy - (Cpy dot b)) / delta
    val s2 = (r1 + r2) / (if (useREML) n - C.cols else n)

    DiagLMMLowRank(n, C, y, dy, ydy, ypy, Cpy, CpC, b, s2, math.log(s2), delta, D.map(1 / _), useREML)
  }

  def deltaGridVals(n: Int, C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], ypy: Double, Cpy: DenseVector[Double], CpC: DenseMatrix[Double], useREML: Boolean): IndexedSeq[(Double, Double)] = {

    val logmin = -10.0
    val logmax = 10.0
    val logres = 0.1

    val grid = (logmin to logmax by logres).map(math.exp)

    // up to constant shift and scale by 2

    def negLogLkhd(delta: Double, useREML: Boolean): Double = {
      val k = y.length
      val c = C.cols
      val D = S + delta
      val dy = y :/ D
      val dC = C(::, *) :/ D
      val ydy = y dot dy
      val Cdy = C.t * dy
      val CdC = C.t * dC
      val b = (CdC + CpC / delta) \ (Cdy + Cpy / delta)
      val r1 = ydy - (Cdy dot b)
      val r2 = (ypy - (Cpy dot b)) / delta
      val r = r1 + r2

      if (useREML)
        sum(breeze.numerics.log(D)) + (n - k) * math.log(delta) + (n - c) * math.log(r) + logdet(CdC + CpC / delta)._2
      else
        sum(breeze.numerics.log(D)) + (n - k) * math.log(delta) + n * math.log(r)
    }

    grid.map(delta => (delta, negLogLkhd(delta, useREML)))
  }

}

case class DiagLMMLowRank(
  n: Int,
  C: DenseMatrix[Double],
  y: DenseVector[Double],
  dy: DenseVector[Double],
  ydy: Double,
  ypy: Double,
  Cpy: DenseVector[Double],
  CpC: DenseMatrix[Double],
  nullB: DenseVector[Double],
  nullS2: Double,
  logNullS2: Double,
  delta: Double,
  invD: DenseVector[Double],
  useREML: Boolean) {

  def likelihoodRatioTest(x: DenseVector[Double], xpy: Double): LMMStat = {
    require(x.length == y.length)

    val c = C.cols

    val X = DenseMatrix.horzcat(x.asDenseMatrix.t, C)

    val XdX = X.t * (X(::, *) :* invD)
    val Xdy = X.t * dy

    val xpx = x dot x
    val Cpx = C.t * x
    val Xpy = DenseVector.vertcat(DenseVector(xpy), Cpy)

    val XpX = DenseMatrix.zeros[Double](c + 1, c + 1)  // ugly
    XpX(0, 0) = xpx
    for (i <- 1 to c) {
      XpX(0, i) = Cpx(i - 1)
      XpX(i, 0) = Cpx(i - 1)
      for (j <- 0 to c) {
        XpX(i, j) = CpC(i - 1, j - 1)
      }
    }

    val b = (XdX + XpX / delta) \ (Xdy + Xpy / delta)

    val r1 = ydy - (Xdy dot b)
    val r2 = (ypy - (Xpy dot b)) / delta
    val s2 = (r1 + r2) / (if (useREML) n - c else n)

    val chi2 = n * (logNullS2 - math.log(s2))
    val p = chiSquaredTail(1, chi2)

    LMMStat(b, s2, chi2, p)
  }
}

case class LMMResultLowRank(diagLMMLowRank: DiagLMMLowRank, stats: RDD[(Variant, LMMStat)])