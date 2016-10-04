package org.broadinstitute.hail.stats

import breeze.linalg.{DenseMatrix, eigSym, svd}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister
import org.broadinstitute.hail.SparkSuite
import org.testng.annotations.Test
import org.broadinstitute.hail.stats._
import org.broadinstitute.hail.utils._

class symEigDSuite extends SparkSuite {
  @Test def eigSymTest() = {
    val seed = 0
    implicit val rand: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    val n = 5
    val m = 10
    val W = DenseMatrix.fill[Double](n, m)(rand.gaussian.draw())
    val K = W * W.t

    val svdW = svd(W)
    val svdK = svd(K)
    val eigSymK = eigSym(K)
    val eigSymDK = eigSymD(K)
    val eigSymRK = eigSymR(K)

    // eigSymD = svdW
    for (j <- 0 until n) {
      assert(D_==(svdW.S(j) * svdW.S(j), eigSymDK.eigenvalues(n - j - 1)))
      for (i <- 0 until n) {
        assert(D_==(math.abs(svdW.U(i, j)), math.abs(eigSymDK.eigenvectors(i, n - j - 1))))
      }
    }

    // eigSymR = svdK
    for (j <- 0 until n) {
      assert(D_==(svdK.S(j), eigSymDK.eigenvalues(n - j - 1)))
      for (i <- 0 until n) {
        assert(D_==(math.abs(svdK.U(i, j)), math.abs(eigSymDK.eigenvectors(i, n - j - 1))))
      }
    }

    // eigSymD = eigSym
    for (j <- 0 until n) {
      assert(D_==(eigSymK.eigenvalues(j), eigSymDK.eigenvalues(j)))
      for (i <- 0 until n) {
        assert(D_==(math.abs(eigSymK.eigenvectors(i, j)), math.abs(eigSymDK.eigenvectors(i, j))))
      }
    }
  }

  @Test def symEigSpeedTest() {
    val seed = 0
    implicit val rand: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))

    def timeSymEig() = {
      for (n <- 500 to 5500 by 500) {
        val W = DenseMatrix.fill[Double](n, n)(rand.gaussian.draw())
        val K = W * W.t

        def computeSVD() { svd(W) }
        def computeSVDK() { svd(K) }
        def computeEigSymD() { eigSymD(K) }
        def computeEigSymR() { eigSymR(K) }
        def computeEigSym() { eigSym(K) }

        println(s"$n dim")
        print("svd:     ")
        printTime(computeSVD())
        print("svdK:    ")
        printTime(computeSVDK())
        print("eigSym:  ")
        printTime(computeEigSym())
        print("eigSymR: ")
        printTime(computeEigSymR())
        print("eigSymD: ")
        printTime(computeEigSymD())
        println()
      }
    }

    // Do not run with standard tests
    // timeSymEig()
  }
}