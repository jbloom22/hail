package org.broadinstitute.hail.stats

import breeze.linalg._
import breeze.stats.mean
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.Variant

class LinearMixedModelSuite extends SparkSuite {

  @Test def diagLMMTest() {

    val C = DenseMatrix(
      (1.0, 0.0, -1.0),
      (1.0, 2.0, 3.0),
      (1.0, 1.0, 5.0),
      (1.0, -2.0, 0.0),
      (1.0, -2.0, -4.0),
      (1.0, 4.0, 3.0))

    val y = DenseVector(0d, 0d, 1d, 1d, 1d, 1d)
    val S = DenseVector(1d, 1d, 1d, 1d, 1d, 1d)
    val delta = 1

    val model = DiagLMM(C, y, S, Some(delta))

    val gts = DenseVector(0d, 1d, 2d, 0d, 0d, 1d)

    val stats = model.likelihoodRatioTest(gts)

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
    val S = (svdW.S :* svdW.S).padTo(n, 0).toDenseVector

    val y = Ut * y0
    val C = Ut * C0

//  val delta = findDelta(y, C)
    val delta = 1d

    val model = DiagLMM(C, y, S, Some(delta))

    val results: Map[Int, LMMStat] = (0 until m).map { v =>
      val gts = G(::, v to v).toDenseVector
      val stats = model.likelihoodRatioTest(gts)
      (v, stats)
    }.toMap

    results.foreach(println)
  }

  @Test def genAndFitLMMTest() {
    val seed = 1
    implicit val rand: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    val n = 100 // even
    val m = 200
    val c = 5
    def b = DenseVector(2.0, 1.0, 0.0, -1.0, -2.0)  // length is c

    val C0 = DenseMatrix.horzcat(DenseMatrix.ones[Double](n, 1), DenseMatrix.fill[Double](n, c - 1)(rand.gaussian.draw()))

    val W = DenseMatrix.vertcat(DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()), DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()) + .5)

    for (i <- 0 until W.cols) {
      W(::,i) -= mean(W(::,i))
      W(::,i) /= norm(W(::,i))
    }

    val K = W * W.t

    val svdW = svd(W)

    def sigmaGSq = 1d
    def delta = 1d
    def V = sigmaGSq * (K + delta * DenseMatrix.eye[Double](n))

    def distY0 = MultivariateGaussian(C0 * b, V)(rand)

    def y0 = distY0.sample()

    val Ut = svdW.U.t
    val S = (svdW.S :* svdW.S).padTo(n, 0).toDenseVector // square singular values of W to get eigenvalues of K

    val y = Ut * y0
    val C = Ut * C0

    // val model = DiagLMM(C, y, S, Some(delta))
    val model = DiagLMM(C, y, S)

    println("delta:")
    println(delta)
    println(model.delta)
    println()
    println("s2:")
    println(sigmaGSq)
    println(model.nullS2)

    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${b(i)}, ${model.nullB(i)}"))
  }

  @Test def genAndFitLMMTestIndexedRowMatrix() {
    val seed = 1
    implicit val rand: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    val n = 100 // even
    val m = 200
    val variantIdx = 0 until m
    val variants = (0 until m).map(i => Variant("1", i, "A", "C")).toArray
    val k = 100
    val c = 5

    def b = DenseVector(2.0, 1.0, 0.0, -1.0, -2.0)  // length is c

    val C0 = DenseMatrix.horzcat(DenseMatrix.ones[Double](n, 1), DenseMatrix.fill[Double](n, c - 1)(rand.gaussian.draw()))

    val W = DenseMatrix.vertcat(DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()), DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()) + .5)

    for (i <- 0 until W.cols) {
      W(::,i) -= mean(W(::,i))
      W(::,i) /= norm(W(::,i))
    }

    val K = W * W.t

    val svdW = svd(W)

    def sigmaGSq = 1d
    def delta = 1d
    def V = sigmaGSq * (K + delta * DenseMatrix.eye[Double](n))

    def distY0 = MultivariateGaussian(C0 * b, V)(rand)

    def y0 = distY0.sample()

    val Ut = svdW.U.t
    val S = (svdW.S :* svdW.S).padTo(n, 0).toDenseVector // square singular values of W to get eigenvalues of K

    val Wcols = (0 until W.cols).map(j => IndexedRow(j, toSDenseVector(W(::, j))))

    val Wt = new IndexedRowMatrix(sc.makeRDD(Wcols), m, n)

    val genotypes = new IndexedRowMatrix(sc.makeRDD(IndexedSeq[IndexedRow]()), 0, n)

    val lmmResult = LMM(Wt, variants, genotypes, C0, y0, k, None)

    val model = lmmResult.diagLMM

    println("delta:")
    println(delta)
    println(model.delta)
    println()
    println("s2:")
    println(sigmaGSq)
    println(model.nullS2)

    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${b(i)}, ${model.nullB(i)}"))
  }
}
