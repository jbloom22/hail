package org.broadinstitute.hail.stats

import breeze.linalg._
import breeze.stats.{mean, stddev}
import breeze.numerics.log
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
}
