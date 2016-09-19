package org.broadinstitute.hail.stats

import breeze.linalg._
import org.apache.commons.math3.special.Gamma
import breeze.numerics._

class DiagLMMDelta(X: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double]) {

  def logLkhdPlusConstant(delta: Double) = {
    val n = y.length
    val D = S + delta
    val dy = y :/ D

    val xdx = X.t * (X(::, *) :/ D)
    val xdy = X.t * dy
    val ydy = y dot dy
    val b = xdx \ xdy
    val s2 = (ydy - (xdy dot b)) / n

    sum(log(D)) + n * math.log(s2)
  }
}


// DiagLMM is model of this form with D known and diagonal: y ~ N(X * b, sigmaGSq * D)
// invD is the diagonal of the inverse of D.
// DiagLMM arises by diagonalizing variance of LMM: y0 ~ N(X0 * b, sigmaGSq * (K + delta * Id))
// SVD of kernel K = U * S * U.t
// Set y = U.t * y0, X = U.t * X0, D = S + delta * Id
// In this case, invD has ith element 1 / (S_ii + delta)

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

case class LMMStats(b: DenseVector[Double], s2: Double, chi2: Double, p: Double)