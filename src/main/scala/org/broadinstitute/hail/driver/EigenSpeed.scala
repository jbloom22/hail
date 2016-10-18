package org.broadinstitute.hail.driver

import breeze.linalg.{DenseMatrix, eigSym, svd}
import org.broadinstitute.hail.stats._
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object EigenSpeed extends Command {

  class Options extends BaseOptions {
    @Args4jOption(required = false, name = "-min", aliases = Array("--mindim"), usage = "min dim")
    var mindim: Int = 1000

    @Args4jOption(required = false, name = "-max", aliases = Array("--maxdim"), usage = "max dim")
    var maxdim: Int = 1000

    @Args4jOption(required = false, name = "-step", aliases = Array("--step"), usage = "step size")
    var step: Int = 500

    @Args4jOption(required = false, name = "-seed", aliases = Array("--seed"), usage = "random seed")
    var seed: Int = 0

    @Args4jOption(required = false, name = "-all", aliases = Array("--all"), usage = "flag to use all algorithms (eigSymD)")
    var all: Boolean = false
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

    scala.util.Random.setSeed(options.seed)

    for {n <- options.mindim to options.maxdim by options.step} {
      val W = DenseMatrix.fill[Double](n, n)(scala.util.Random.nextGaussian())
      val K = W * W.t

      info(s"$n, eigSymD: ${ timeString({ eigSymD(K) }) }")
      info(s"$n, eigSymD: ${ timeString({ eigSymD(K) }) }")
      if (options.all) {
        info(s"$n, eigSymR: ${ timeString({ eigSymR(K) }) }")
        info(s"$n, svdK:    ${ timeString({ svd(K) }) }")
        info(s"$n, svd:     ${ timeString({ svd(W) }) }")
        info(s"$n, eigSym:  ${ timeString({ eigSym(K) }) }")
      }
      println()
    }

    state
  }
}
