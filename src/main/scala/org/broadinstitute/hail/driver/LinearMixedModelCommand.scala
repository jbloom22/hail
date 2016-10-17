package org.broadinstitute.hail.driver

import breeze.linalg._
import org.broadinstitute.hail.annotations.Annotation
import org.broadinstitute.hail.expr._
import org.broadinstitute.hail.stats.{LMM, LMMStat}
import org.broadinstitute.hail.utils._
import org.kohsuke.args4j.{Option => Args4jOption}

object LinearMixedModelCommand extends Command {

  def name = "lmmreg"

  def description = "Test each variant for association using a linear mixed model"

  class Options extends BaseOptions {
    @Args4jOption(required = true, name = "-y", aliases = Array("--response"), usage = "Response sample annotation")
    var ySA: String = _

    @Args4jOption(required = false, name = "-c", aliases = Array("--covariates"), usage = "Covariate sample annotations, comma-separated")
    var covSA: String = ""

    @Args4jOption(required = true, name = "-kfe", aliases = Array("--kernelfilterexpr"), usage = "Variant filter expression for kernel")
    var kernelFiltExprSA: String = _

    @Args4jOption(required = true, name = "-afe", aliases = Array("--assocfilterexpr"), usage = "Variant filter expression for association")
    var assocFiltExprSA: String = _

    @Args4jOption(required = false, name = "-ml", aliases = Array("--useml"), usage = "Use ml instead of reml to fit delta")
    var useML: Boolean = false

    @Args4jOption(required = false, name = "-delta", aliases = Array("--delta"), usage = "Fixed delta value (overrides fitting delta)")
    var delta: Double = Double.NaN // is this right?

    @Args4jOption(required = false, name = "-r", aliases = Array("--root"), usage = "Variant annotation root, a period-delimited path starting with `va'")
    var root: String = "va.lmmreg"
  }

  def newOptions = new Options

  def supportsMultiallelic = false

  def requiresVDS = true

  def run(state: State, options: Options): State = {
    val vds = state.vds

    val pathVA = Parser.parseAnnotationRoot(options.root, Annotation.VARIANT_HEAD)

    val symTab = Map(
      "s" -> (0, TSample),
      "sa" -> (1, vds.saSignature))

    val ec = EvalContext(symTab)

    def toDouble(t: BaseType, code: String): Any => Double = t match {
      case TInt => _.asInstanceOf[Int].toDouble
      case TLong => _.asInstanceOf[Long].toDouble
      case TFloat => _.asInstanceOf[Float].toDouble
      case TDouble => _.asInstanceOf[Double]
      case TBoolean => _.asInstanceOf[Boolean].toDouble
      case _ => fatal(s"Sample annotation `$code' must be numeric or Boolean, got $t")
    }

    val (yT, yQ) = Parser.parse(options.ySA, ec)
    val yToDouble = toDouble(yT, options.ySA)
    val ySA = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      yQ().map(yToDouble)
    }

    val (covT, covQ) = Parser.parseExprs(options.covSA, ec).unzip
    val covToDouble = (covT, options.covSA.split(",").map(_.trim)).zipped.map(toDouble)
    val covSA = vds.sampleIdsAndAnnotations.map { case (s, sa) =>
      ec.setAll(s, sa)
      (covQ.map(_ ()), covToDouble).zipped.map(_.map(_))
    }

    val (yForCompleteSamples, covForCompleteSamples, completeSamples) =
      (ySA, covSA, vds.sampleIds)
        .zipped
        .filter((y, c, s) => y.isDefined && c.forall(_.isDefined))

    val yArray = yForCompleteSamples.map(_.get).toArray
    val y = DenseVector(yArray)

    val covArray = covForCompleteSamples.flatMap(_.map(_.get)).toArray
    val k = covT.size
    val cov =
      if (k == 0)
        None
      else
        Some(new DenseMatrix(
          rows = completeSamples.size,
          cols = k,
          data = covArray,
          offset = 0,
          majorStride = k,
          isTranspose = true))

    val completeSampleSet = completeSamples.toSet

    val n = y.size
    val d = n - k - 2

    if (d < 1)
      fatal(s"$n samples and $k ${plural(k, "covariate")} with intercept implies $d degrees of freedom.")

    info(s"Running lmmreg on $n samples with $k sample ${plural(k, "covariate")}...")

    val C: DenseMatrix[Double] = cov match {
      case Some(dm) => DenseMatrix.horzcat(dm, DenseMatrix.ones[Double](n, 1))
      case None => DenseMatrix.ones[Double](n, 1)
    }

    val vdsForCompleteSamples = vds.filterSamples((s, sa) => completeSampleSet(s))
    val vdsAssoc = vdsForCompleteSamples.filterVariantsExpr(options.assocFiltExprSA, keep = true)
    val vdsKernel = vdsForCompleteSamples.filterVariantsExpr(options.kernelFiltExprSA, keep = true)

    val optDelta =
      if (options.delta.isNaN)
        None
      else {
        if (options.delta <= 0d)
          fatal(s"delta must be positive, got ${ options.delta }")
        else
          Some(options.delta)
      }

    val lmmreg = LMM(vdsKernel, vdsAssoc, C, y, optDelta, options.useML)

    val (newVAS, inserter) = vdsForCompleteSamples.insertVA(LMMStat.`type`, pathVA)

    state.copy(
      vds = vds.copy(
        rdd = vds.rdd.orderedLeftJoinDistinct(lmmreg.rdd.toOrderedRDD)
          .mapValues{ case ((va, gs), optlmmstat) => (inserter(va, optlmmstat.map(_.toAnnotation)), gs) }
          .asOrderedRDD,
        vaSignature = newVAS
      )
    )
  }
}
