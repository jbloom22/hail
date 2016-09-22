package org.broadinstitute.hail.stats

import breeze.linalg._
import org.apache.commons.math3.special.Gamma
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
    optDelta: Option[Double] = None): LMMResult = {

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
    optDelta: Option[Double] = None): LMMResult = {

    require(k <= Wt.numRows())
    require(k <= Wt.numCols())

    val svd: SingularValueDecomposition[IndexedRowMatrix, org.apache.spark.mllib.linalg.Matrix] = Wt.computeSVD(k)

    val U = svd.V // W = svd.V * svd.s * svd.U.t
    val s = toBDenseVector(svd.s.toDense)
    val SB =  s :* s // K = U * (svd.s * svd.s) * V.t
    val UB = new DenseMatrix[Double](svd.V.numRows, svd.V.numCols, svd.V.toArray)

    val diagLMM = DiagLMM(UB.t * C, UB.t * y, SB, optDelta)

    val diagLMMBc = genotypes.rows.sparkContext.broadcast(diagLMM)

    // FIXME: variant names are already local, we shouldn't be gathering them all, right?!

    val lmmResult = genotypes.multiply(U).rows.map(r =>
      (variants(r.index.toInt), diagLMMBc.value.likelihoodRatioTest(toBDenseVector(r.vector.toDense))))

    LMMResult(diagLMM, lmmResult)
  }
}

object DiagLMM {
  def apply(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], optDelta: Option[Double] = None): DiagLMM = {
    require(C.rows == y.length)

    val delta = optDelta.getOrElse(deltaGridVals(C, y, S).minBy(_._2)._1)

    val n = y.length
    val D = S + delta
    val dy = y :/ D
    val ydy = y dot dy
    val xdy = C.t * dy
    val xdx = C.t * (C(::, *) :/ D)
    val b = xdx \ xdy
    val s2 = (ydy - (xdy dot b)) / n

    DiagLMM(C, y, dy, ydy, b, s2, math.log(s2), delta, D.map(1 / _))
  }

  def deltaGridVals(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double]): IndexedSeq[(Double, Double)] = {

    val logmin = -10.0
    val logmax = 10.0
    val logres = 0.1

    val grid = (logmin to logmax by logres).map(math.exp)

    val n = y.length

    // up to constant shift and scale
    def negLogLkhd(delta: Double): Double = {
      val D = S + delta
      val dy = y :/ D
      val ydy = y dot dy
      val xdy = C.t * dy
      val xdx = C.t * (C(::, *) :/ D)
      val b = xdx \ xdy
      val s2 = (ydy - (xdy dot b)) / n

      sum(breeze.numerics.log(D)) + n * math.log(s2)
    }

    grid.map(delta => (delta, negLogLkhd(delta)))
  }

  // remove once logreg is in
  def chiSquaredTail(df: Double, x: Double) = Gamma.regularizedGammaQ(df / 2, x / 2)
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
  invD: DenseVector[Double]) {

  def likelihoodRatioTest(x: DenseVector[Double]): LMMStat = {
    require(x.length == y.length)

    val n = x.length
    val X = DenseMatrix.horzcat(x.asDenseMatrix.t, C)

    val xdx = X.t * (X(::, *) :* invD)
    val xdy = X.t * dy

    val b = xdx \ xdy
    val s2 = (ydy - (xdy dot b)) / n
    val chi2 = n * (logNullS2 - math.log(s2))
    val p = DiagLMM.chiSquaredTail(1, chi2)

    LMMStat(b, s2, chi2, p)
  }
}

case class LMMStat(b: DenseVector[Double], s2: Double, chi2: Double, p: Double)

case class LMMResult(diagLMM: DiagLMM, stats: RDD[(Variant, LMMStat)])