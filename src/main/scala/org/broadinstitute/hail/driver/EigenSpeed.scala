package org.broadinstitute.hail.driver

import breeze.linalg.{DenseMatrix, eigSym, svd}
import org.apache.commons.math3.random.JDKRandomGenerator
import org.broadinstitute.hail.stats._
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object EigenSpeed extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-min", aliases = Array("--mindim"), usage = "min dim")
    var mindim: Int = 1000

    @Args4jOption(required = false, name = "-max", aliases = Array("--maxdim"), usage = "max dim")
    var maxdim: Int = 1000

    @Args4jOption(required = false, name = "-s", aliases = Array("--step"), usage = "step size")
    var step: Int = 500
  }

  def newOptions = new Options

  def name = "eigenspeed"

  def description = "Run speed test of several symmetric eigensolvers"

  def supportsMultiallelic = true

  def requiresVDS = false

  def run(state: State, options: Options): State = {

    def timeString[A](f: => A): String = {
      val t = time(f)._2
      val ft = formatTime(t)
      s"$ft \t $t"
    }

    val seed = 0
    val rand = new JDKRandomGenerator()

    rand.setSeed(seed)

    for {n <- options.mindim to options.maxdim by options.step} {
      val W = DenseMatrix.fill[Double](n, n)(rand.nextGaussian())
      val K = W * W.t

      def computeSVD() { svd(W) }
      def computeSVDK() { svd(K) }
      def computeEigSymD() { eigSymD(K) }
      def computeEigSymR() { eigSymR(K) }
      def computeEigSym() { eigSym(K) }

      info(s"$n, eigSymD: ${ timeString(computeEigSymD()) }")
      info(s"$n, eigSymD: ${ timeString(computeEigSymD()) }")
      info(s"$n, eigSymR: ${ timeString(computeEigSymR()) }")
      info(s"$n, svdK:    ${ timeString(computeSVDK()) }")
      info(s"$n, svd:     ${ timeString(computeSVD()) }")
      info(s"$n, eigSym:  ${ timeString(computeEigSym()) }")
      println()
    }

    state
  }
}
