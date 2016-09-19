package org.broadinstitute.hail.stats

import breeze.linalg._
import org.apache.commons.math3.special.Gamma
import breeze.numerics._



object DiagLMM {
  def apply(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], optDelta: Option[Double] = None): DiagLMM = {
    require(C.rows == y.length)

    val delta = optDelta.getOrElse(findDelta(C, y, S))

    val n = y.length
    val D = S + delta
    val dy = y :/ D
    val ydy = y dot dy
    val xdy = C.t * dy
    val xdx = C.t * (C(::, *) :/ D)
    val b = xdx \ xdy
    val s2 = (ydy - (xdy dot b)) / n

    DiagLMM(C, y, dy, ydy, b, s2, log(s2), delta, D.map(1 / _))
  }

  def findDelta(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double]): Double = {
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

      sum(log(D)) + n * math.log(s2)
    }

    val minIndex = grid.map(negLogLkhd).zipWithIndex.minBy(_._1)._2

    grid(minIndex)
  }

  // can removed once logreg is in
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

  def likelihoodRatioTest(x: DenseVector[Double]): LMMStats = {
    require(x.length == y.length)

    val n = x.length
    val X = DenseMatrix.horzcat(x.asDenseMatrix.t, C)

    val xdx = X.t * (X(::, *) :* invD)
    val xdy = X.t * dy

    val b = xdx \ xdy
    val s2 = (ydy - (xdy dot b)) / n
    val chi2 = n * (logNullS2 - log(s2))
    val p = DiagLMM.chiSquaredTail(1, chi2)

    LMMStats(b, s2, chi2, p)
  }
}

case class LMMStats(b: DenseVector[Double], s2: Double, chi2: Double, p: Double)