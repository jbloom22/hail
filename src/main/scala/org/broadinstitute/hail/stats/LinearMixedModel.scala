package org.broadinstitute.hail.stats

import breeze.linalg._
import org.apache.commons.math3.special.Gamma
import breeze.numerics._

// for FaST-LMM, dInv = vector with ith element 1 / (S_ii + delta)
class DiagLMM(X: DenseMatrix[Double], y: DenseVector[Double], invD: DenseVector[Double]) {
  val n = X.rows
  require(n == y.length)

  def likelihoodRatioTest(dy: DenseVector[Double], ydy: Double, logNullS2: Double): LMMStats = {
    val (b, s2) = fit(dy, ydy)
    val chi2 = n * (logNullS2 - log(s2))
    val p = chiSquaredTail(1, chi2)

    LMMStats(b, s2, chi2, p)
  }

  def fit(dy: DenseVector[Double], ydy: Double): (DenseVector[Double], Double) = {
    val xdx = X.t * (X(::, *) :* invD)
    val xdy = X.t * dy
    val b = xdx \ xdy
    val s2 = (ydy - (xdy dot b)) / n

    (b, s2)
  }

  def chiSquaredTail(df: Double, x: Double) = Gamma.regularizedGammaQ(df / 2, x / 2)

  // val c = X.cols
  // val chi2 = (n - c) * (nullRss / rss - 1) // R
  // val chi2 = xdy dot b // BoltLMM
}

//class LMM(X: DenseMatrix[Double], y: DenseVector[Double], Ut: DenseMatrix[Double], S: DenseVector[Double], delta: Double) {
//  val invD = (S :+ delta).map(1 / _)
//
//  DiagLMM(Ut * X, Ut * y, invD)
//}


case class LMMStats(b: DenseVector[Double], s2: Double, chi2: Double, p: Double)