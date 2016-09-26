package org.broadinstitute.hail.stats

import breeze.linalg._
import breeze.numerics.sqrt
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

    println(stats.b(0))
    println(stats.b(1))
    println(stats.b(2))
    println(stats.s2)
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
    val m = 10
    val c = 5
    val b = DenseVector(2.0, 1.0, 0.0, -1.0, -2.0)  // length is c

    val C0 = DenseMatrix.horzcat(DenseMatrix.ones[Double](n, 1), DenseMatrix.fill[Double](n, c - 1)(rand.gaussian.draw()))

    val W = DenseMatrix.vertcat(DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()), DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()) + .5)

    for (i <- 0 until W.cols) {
      W(::,i) -= mean(W(::,i))
      W(::,i) /= norm(W(::,i))
    }

    val K = W * W.t

    val svdW = svd(W)

    val sigmaGSq = 1d
    val delta = 1d
    val V = sigmaGSq * (K + delta * DenseMatrix.eye[Double](n))

    val distY0 = MultivariateGaussian(C0 * b, V)(rand)

    val y0 = distY0.sample()

    val Ut = svdW.U.t

    // println(Ut.cols, Ut.rows)

    val S = (svdW.S :* svdW.S).padTo(n, 0).toDenseVector // square singular values of W to get eigenvalues of K

    // S.foreach(println)

    val y = Ut * y0
    val C = Ut * C0

    val model = DiagLMM(C, y, S, optDelta = None, useREML = false)
    //val model = DiagLMM(C, y, S)

    println()
    println("ML delta:")
    println(delta)
    println(model.delta)
    println()
    println("s2:")
    println(sigmaGSq)
    println(model.nullS2)
    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${b(i)}, ${model.nullB(i)}"))

    val modelR = DiagLMM(C, y, S, optDelta = None, useREML = true)
    //val model = DiagLMM(C, y, S)

    println()
    println("REML delta:")
    println(delta)
    println(modelR.delta)
    println()
    println("s2:")
    println(sigmaGSq)
    println(modelR.nullS2)
    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${b(i)}, ${modelR.nullB(i)}"))
  }

  @Test def genAndFitLMMTestDist() {
    val seed = 1
    implicit val rand: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    val n = 100 // even
    val m = 10
    val variantIdx = 0 until m
    val variants = (0 until m).map(i => Variant("1", i, "A", "C")).toArray
    val k = 10
    val c = 5

    val b = DenseVector(2.0, 1.0, 0.0, -1.0, -2.0)  // length is c

    val C0 = DenseMatrix.horzcat(DenseMatrix.ones[Double](n, 1), DenseMatrix.fill[Double](n, c - 1)(rand.gaussian.draw()))

    val W = DenseMatrix.vertcat(DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()), DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()) + .5)

    for (i <- 0 until W.cols) {
      W(::,i) -= mean(W(::,i))
      W(::,i) /= norm(W(::,i))
    }

    val K = W * W.t

    val sigmaGSq = 1d
    val delta = 1d
    val V = sigmaGSq * (K + delta * DenseMatrix.eye[Double](n))

    val distY0 = MultivariateGaussian(C0 * b, V)(rand)

    val y0 = distY0.sample()

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

  @Test def genAndFitLMMTestDistLowRank() {
    val seed = 1
    implicit val rand: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    val n = 100 // even
    val m = 10
//    val variantIdx = 0 until m
//    val variants = (0 until m).map(i => Variant("1", i, "A", "C")).toArray
    val k = 10
    val c = 5

    val b = DenseVector(2.0, 1.0, 0.0, -1.0, -2.0)  // length is c

    val C0 = DenseMatrix.horzcat(DenseMatrix.ones[Double](n, 1), DenseMatrix.fill[Double](n, c - 1)(rand.gaussian.draw()))

    val W = DenseMatrix.vertcat(DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()), DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()) + .5)

    for (i <- 0 until W.cols) {
      W(::,i) -= mean(W(::,i))
      W(::,i) /= norm(W(::,i))
    }

    val K = W * W.t

    val sigmaGSq = 1d
    val delta = 1d
    val V = sigmaGSq * (K + delta * DenseMatrix.eye[Double](n))

    val distY0 = MultivariateGaussian(C0 * b, V)(rand)

    val y0 = distY0.sample()

    val Wcols = (0 until W.cols).map(j => IndexedRow(j, toSDenseVector(W(::, j))))

    val Wt = new IndexedRowMatrix(sc.makeRDD(Wcols), m, n)

    val variants = Array[Variant]()
    val genotypes = new IndexedRowMatrix(sc.makeRDD(IndexedSeq[IndexedRow]()), 0, n)

    val lmmResultLowRank = LMMLowRank(Wt, variants, genotypes, C0, y0, k, None, useREML = false)
    val model = lmmResultLowRank.diagLMMLowRank

    println("ML delta:")
    println(delta)
    println(model.delta)
    println()
    println("s2:")
    println(sigmaGSq)
    println(model.nullS2)
    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${b(i)}, ${model.nullB(i)}"))

    val lmmResultLowRankR = LMMLowRank(Wt, variants, genotypes, C0, y0, k, None, useREML = true)
    val modelR = lmmResultLowRankR.diagLMMLowRank

    println()
    println("REML delta:")
    println(delta)
    println(modelR.delta)
    println()
    println("s2:")
    println(sigmaGSq)
    println(modelR.nullS2)

    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${b(i)}, ${modelR.nullB(i)}"))

  }

  @Test def testLowRank() {
    val seed = 1
    implicit val rand: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    // generate data

    val n = 8 // even
    val m = 4
    val c = 2
    val b = DenseVector(2.0, -1.0) // length is c
    val C = DenseMatrix.fill[Double](n, c)(rand.gaussian.draw())
    def delta = 2d
    def sigmaGSq = 1d

    val W = DenseMatrix.vertcat(DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()), DenseMatrix.fill[Double](n / 2, m)(rand.gaussian.draw()) + 1d)
    for (i <- 0 until W.cols) {
      W(::, i) -= mean(W(::, i))
      W(::, i) /= norm(W(::, i))
    }
    val K = W * W.t

    println(W)

    val V = sigmaGSq * (K + delta * DenseMatrix.eye[Double](n))
    val yDist = MultivariateGaussian(C * b, V)(rand)
    val y = yDist.sample()

    println()
    println(s"y = $y")

    // solve using full rank

    val svdW = svd(W)
    val S = (svdW.S :* svdW.S).padTo(n, 0).toDenseVector // square singular values of W to get eigenvalues of K
    val U = svdW.U

    println()
    println("S")
    (0 until n).foreach(i => println(s"$i: ${S(i)}"))
    println()
    println(s"U is (${U.rows}, ${U.cols})")
    println(U)

    // 0 means rotated by U.t
    val y0 = U.t * y
    val C0 = U.t * C

    val D = S + delta
    val dy0 = y0 :/ D
    val dC0 = C0(::, *) :/ D
    val y0dy0 = y0 dot dy0
    val C0dy0 = C0.t * dy0
    val C0dC0 = C0.t * dC0
    val beta0 = C0dC0 \ C0dy0
    val s20 = (y0dy0 - (C0dy0 dot beta0)) / n

    println()
    println("rotate")
    println("s2:")
    println(sigmaGSq)
    println(s20)

    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${ b(i) }, ${ beta0(i) }"))

    // solve by rotate and scale
    // 1 means rotated and scaled

    val sqrtD = sqrt(D)
    val y1 = y0 :/ sqrtD
    val C1 = C0(::, *) :/ sqrtD

    val y1y1 = y1 dot y1
    val C1y1 = C1.t * y1
    val C1C1 = C1.t * C1
    val beta1 = C1C1 \ C1y1
    val s21 = (y1y1 - (C1y1 dot beta1)) / n

    println()
    println("rotate and scale")
    println("s2:")
    println(sigmaGSq)
    println(s21)

    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${ b(i) }, ${ beta1(i) }"))

    println(s"y = $y")

    // solve for b0 by rotate, scale, and project out cov
    // 2 means rotated and scaled and projected out cov

    val x1 = C1(::, 0)
    val cov = C1(::, 1 to 1)
    val Qt = qr.reduced.justQ(cov).t

    val x2 = Qt * x1
    val y2 = Qt * y1

    val y2y2 = (y1 dot y1) - (y2 dot y2)
    val x2y2 = (x1 dot y1) - (x2 dot y2)
    val x2x2 = (x1 dot x1) - (x2 dot x2)
    val beta2 = x2y2 / x2x2
    val s22 = (y2y2 - (x2y2 * beta2)) / n

    println()
    println("rotate and scale and project out cov")
    println("s2:")
    println(sigmaGSq)
    println(s22)

    println()
    println("b0")
    println(s"0: ${b(0)}, $beta2")



    // 3 means project low rank
    val k = 4 // k >= m gives same answer

    val S3 = S(0 until k).copy
    val U3 = U(::, 0 until k).copy

    println()
    println(s"S low rank = $S3")
    println(s"U low rank =\n$U3")

    val y3 = U3.t * y
    val C3 = U3.t * C

    val D3 = S3 + delta
    val dy3 = y3 :/ D3
    val dC3 = C3(::, *) :/ D3
    val y3dy3 = y3 dot dy3
    val C3dy3 = C3.t * dy3
    val C3dC3 = C3.t * dC3

    val C3pC3 = (C.t * C) - (C3.t * C3)
    val C3py3 = (C.t * y) - (C3.t * y3)
    val y3py3 = (y dot y) - (y3 dot y3)


    // r1 and r2 here are a different decomposition than in the paper
    val beta3 = (C3dC3 + (C3pC3 / delta)) \ (C3dy3 + (C3py3 / delta))
    val r1 = y3dy3 - (C3dy3 dot beta3)
    val r2 = (y3py3 - (C3py3 dot beta3)) / delta
    val s23 = (r1 + r2) / n

    println()
    println("project low rank")
    println("s2:")
    println(sigmaGSq)
    println(s23)

    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${ b(i) }, ${ beta3(i) }"))

    // 4 means projected to complement of image of U3 in Rn
    val C4 = C - U3 * C3
    val y4 = y - U3 * y3

    val C4pC4 = DenseMatrix.zeros[Double](c, c)
    for (i <- 0 until n) {
      C4pC4 += C4(i,::).t * C4(i,::)
    }

    println()
    println(C4pC4) // all equal
    println(C4.t * C4)
    println(C3pC3)

    val C4py4 = DenseVector.zeros[Double](c)
    for (i <- 0 until n) {
      C4py4 += C4(i,::).t * y4(i)
    }

    println() // all equal
    println(C4py4)
    println(C4.t * y4)
    println(C3py3)

    val y4py4 = y4.t * y4

    println() // all equal
    println(y4py4)
    println(y3py3)

    val beta4 = (C3dC3 + (C4pC4 / delta)) \ (C3dy3 + (C4py4 / delta))

    val normres13 = sqrt(y3dy3 - (C3dy3 dot beta4))
    val normres23 = sqrt(y3py3 - (C3py3 dot beta4))

    val res1 = y3 - C3 * beta4
    val normres24 = norm(y4 - C4 * beta4)

    // r1 and r2 here are as in the paper
    val r14 = res1 dot (res1 :/ D3)
    val r24 = normres24 * normres24 / delta
    val s24 = (r14 + r24) / n

    println()
    println(normres13)
    println(normres23)
    println()
    println(sqrt(r14))
    println(normres24)

    println()
    println("project low rank literal")
    println("s2:")
    println(sigmaGSq)
    println(s24)

    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${ b(i) }, ${ beta4(i) }"))

    // using LMMLowRank

    val Wcols = (0 until W.cols).map(j => IndexedRow(j, toSDenseVector(W(::, j))))
    val Wt = new IndexedRowMatrix(sc.makeRDD(Wcols), m, n)
    val variants = Array[Variant]()
    val genotypes = new IndexedRowMatrix(sc.makeRDD(IndexedSeq[IndexedRow]()), 0, n)
    val lmmResultLowRank = LMMLowRank(Wt, variants, genotypes, C, y, k, Some(delta))
    val model = lmmResultLowRank.diagLMMLowRank

    println()
    println("project low rank class")
    println("s2:")
    println(sigmaGSq)
    println(model.nullS2)

    println()
    println("b")
    (0 until c).foreach(i => println(s"$i: ${b(i)}, ${model.nullB(i)}"))
  }
}
