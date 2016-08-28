package org.broadinstitute.hail.stats

import breeze.linalg._
import org.apache.commons.math3.special.Gamma
import breeze.numerics._

// for FaST-LMM, dInv = vector with ith element 1 / (S_ii + delta)
class DiagLMM(X: DenseMatrix[Double], y: DenseVector[Double], invD: DenseVector[Double]) {
  val n = X.rows
  require(n == y.length)

  def likelihoodRatioTest(dy: DenseVector[Double], ydy: Double, nullStat: Double): LMMStats = {
    // require x to vary

    val xdx = X.t * (X(::,*) :* invD)
    val xdy = X.t * dy
    val b =  xdx \ xdy
    val nSigmaGSq = ydy - (xdy dot b)
    val chi2 = n * (nullStat - log(nSigmaGSq))
    // val chi2 = xdy dot b // BoltLMM
    val p = chiSquaredTail(1, chi2)
    LMMStats(b, chi2, p)
  }

  def getLogNSigmaGSq(dy: DenseVector[Double], ydy: Double): Double = {
    val xdx = X.t * (X(::, *) :* invD)
    val xdy = X.t * dy
    val b = xdx \ xdy

    log(ydy - (xdy dot b))
  }

  def getChi2(dy: DenseVector[Double], ydy: Double): Double = {
    val xdx = X.t * (X(::, *) :* invD)
    val xdy = X.t * dy
    val b = xdx \ xdy

    xdy dot b // BoltLMM
  }

  def chiSquaredTail(df: Double, x: Double) = Gamma.regularizedGammaQ(df / 2, x / 2)
}

case class LMMStats(b: DenseVector[Double], chi2: Double, p: Double)