package org.broadinstitute.hail.stats

import breeze.linalg._
import breeze.stats.mean
import breeze.numerics.log
import breeze.stats.distributions.{MultivariateGaussian, Rand}
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test

class LinearMixedModelSuite extends SparkSuite {

  @Test def diagLMMTest() {

    val C = DenseMatrix(
      (1.0, 0.0, -1.0),
      (1.0, 2.0, 3.0),
      (1.0, 1.0, 5.0),
      (1.0, -2.0, 0.0),
      (1.0, -2.0, -4.0),
      (1.0, 4.0, 3.0))

    val gts = DenseVector(0d, 1d, 2d, 0d, 0d, 1d)
    val X = DenseMatrix.horzcat(gts.asDenseMatrix.t, C)
    val y = DenseVector(0d, 0d, 1d, 1d, 1d, 1d)

    val invD = DenseVector(1d, 1d, 1d, 1d, 1d, 1d)

    val dy = invD :* y
    val ydy = y dot dy

    val nullModel = new DiagLMM(C, y, invD)
    val (nullB, nullS2) = nullModel.fit(dy, ydy)

    val model = new DiagLMM(X, y, invD)
    val stats = model.likelihoodRatioTest(dy, ydy, log(nullS2))

    println(stats)
  }

  @Test def lmmTest() {

    val y0 = DenseVector(0d, 0d, 1d, 1d, 1d, 1d)

    val C0 = DenseMatrix(
      (1.0, 0.0, -1.0),
      (1.0, 2.0, 3.0),
      (1.0, 1.0, 5.0),
      (1.0, -2.0, 0.0),
      (1.0, -2.0, -4.0),
      (1.0, 4.0, 3.0))

    val G = DenseMatrix(
      (0d, 1d, 1d, 2d),
      (1d, 0d, 2d, 2d),
      (2d, 0d, 1d, 2d),
      (0d, 0d, 0d, 2d),
      (0d, 1d, 0d, 0d),
      (1d, 1d, 0d, 0d))

    val n = G.rows
    val m = G.cols

    val W = G(::, 0 to 1)

    for (i <- 0 until W.cols) {
      W(::,i) -= mean(W(::,i))
      W(::,i) /= norm(W(::,i))
    }

    val svdW = svd(W)
    val Ut = svdW.U.t
    val S = svdW.S.padTo(n, 0).toDenseVector

    val y = Ut * y0
    val C = Ut * C0

//  val delta = findDelta(y, C)
    val delta = 1d

    val invD = S.map(s => 1 / (s * s + delta))
    val dy = invD :* y
    val ydy = y dot dy

    val nullModel = new DiagLMM(C, y, invD)
    val (nullB, nullS2) = nullModel.fit(dy, ydy)
    val logNullS2 = log(nullS2)

    val results: Map[Int, LMMStats] = (0 until m).map { v =>
      val gts = G(::, v to v)
      val X = DenseMatrix.horzcat(Ut * gts, C)
      val model = new DiagLMM(X, y, invD)
      val stats = model.likelihoodRatioTest(dy, ydy, logNullS2)
      (v, stats)
    }.toMap

    println()
    results.foreach(println)
  }

  @Test def genAndFitLMMTest() {
    val n = 1000 // even
    val m = 100
    val c = 5
    def b = DenseVector(2.0, 1.0, 0.0, -1.0, -2.0)  // length is c

    val C0 = DenseMatrix.horzcat(DenseMatrix.ones[Double](n, 1), DenseMatrix.fill[Double](n, c - 1)(Rand.gaussian.draw()))

    val W = DenseMatrix.vertcat(DenseMatrix.fill[Double](n / 2, m)(Rand.gaussian.draw()), DenseMatrix.fill[Double](n / 2, m)(Rand.gaussian.draw()) + .5)

    for (i <- 0 until W.cols) {
      W(::,i) -= mean(W(::,i))
      W(::,i) /= norm(W(::,i))
    }

    val K = W * W.t

    val svdW = svd(W)

    def sigmaGSq = 1d
    def delta = 1d
    def V = sigmaGSq * (K + delta * DenseMatrix.eye[Double](n))

    def distY0 = MultivariateGaussian(C0 * b, V)

    def y0 = distY0.sample()

    val Ut = svdW.U.t
    val S = svdW.S.padTo(n, 0).toDenseVector

    val y = Ut * y0
    val C = Ut * C0

    val deltaFit = delta
    // val deltaFit = findDelta(y, C)

    val invD = S.map(s => 1 / (s * s + deltaFit))
    val dy = invD :* y
    val ydy = y dot dy
    val nullModel = new DiagLMM(C, y, invD)
    val (nullB, nullS2) = nullModel.fit(dy, ydy)

    println("delta:")
    println(delta)
    println(deltaFit)
    println()
    println("sigmaG2:")
    println(sigmaGSq)
    println(nullS2)

    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${b(i)}, ${nullB(i)}"))

  }
}
