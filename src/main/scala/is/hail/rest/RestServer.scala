package is.hail.rest

import breeze.linalg.{DenseMatrix, DenseVector}
import is.hail.stats.RegressionUtils
import is.hail.variant._
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.spark.rdd.RDD
import org.http4s.server.blaze.BlazeBuilder

object RestServerLinreg {
  def apply(vds: VariantDataset, covariates: Array[String], useDosages: Boolean,
    port: Int, maxWidth: Int, hardLimit: Int) {

    val restService = new RestServiceLinreg(vds, covariates, useDosages, maxWidth, hardLimit)
    
    val task = BlazeBuilder.bindHttp(port, "0.0.0.0")
      .mountService(restService.service, "/")
      .run
    task.awaitShutdown()
  }
}

object RestServerScoreCovariance {
  def apply(vds: VariantDataset, covariates: Array[String], useDosages: Boolean,
    port: Int, maxWidth: Int, hardLimit: Int) {

    val restService = new RestServiceScoreCovariance(vds, covariates, useDosages, maxWidth, hardLimit)
    
    val task = BlazeBuilder.bindHttp(port, "0.0.0.0")
      .mountService(restService.service, "/")
      .run
    task.awaitShutdown()
  }
}

object RestServer {
  def windowToString(window: Interval[Locus]): String =
    s"${ window.start.contig }:${ window.start.position }-${ window.end.position - 1 }"

  def getYCovAndSetMask(sampleMask: Array[Boolean],
    vds: VariantDataset,
    window: Interval[Locus],
    yNameOpt: Option[String],
    covNames: Array[String],
    covVariants: Array[Variant] = Array.empty[Variant],
    useDosages: Boolean,
    availableCovariates: Set[String],
    availableCovariateToIndex: Map[String, Int],
    sampleIndexToPresentness: Array[Array[Boolean]], 
    covariateIndexToValues: Array[Array[Double]]): (Option[DenseVector[Double]], DenseMatrix[Double]) = {

    val nSamples = vds.nSamples
    
    // sample mask
    val yCovIndices = (yNameOpt.toArray ++ covNames).map(availableCovariateToIndex).sorted

    var nMaskedSamples = 0
    var sampleIndex = 0
    while (sampleIndex < nSamples) {
      val include = yCovIndices.forall(sampleIndexToPresentness(sampleIndex))
      sampleMask(sampleIndex) = include
      if (include) nMaskedSamples += 1
      sampleIndex += 1
    }

    var arrayIndex = 0
    
    // y
    val yOpt = yNameOpt.map { yName =>
      val yArray = Array.ofDim[Double](nMaskedSamples)
      val yData = covariateIndexToValues(availableCovariateToIndex(yName))
      sampleIndex = 0
      while (sampleIndex < nSamples) {
        if (sampleMask(sampleIndex)) {
          yArray(arrayIndex) = yData(sampleIndex)
          arrayIndex += 1
        }
        sampleIndex += 1
      }
      DenseVector(yArray)
    }

    // cov: set intercept, phenotype covariate, and variant covariate values
    val nCovs = 1 + covNames.size + covVariants.size
    val covArray = Array.ofDim[Double](nMaskedSamples * nCovs)

    // intercept
    arrayIndex = 0
    while (arrayIndex < nMaskedSamples) {
      covArray(arrayIndex) = 1
      arrayIndex += 1
    }

    // phenotype covariates
    covNames.foreach { covName =>
      val thisCovData = covariateIndexToValues(availableCovariateToIndex(covName))
      sampleIndex = 0
      while (sampleIndex < nSamples) {
        if (sampleMask(sampleIndex)) {
          covArray(arrayIndex) = thisCovData(sampleIndex)
          arrayIndex += 1
        }
        sampleIndex += 1
      }
    }

    // variant covariates
    val sampleMaskBc = vds.sparkContext.broadcast(sampleMask)

    val covVariantWithGenotypes = vds
      .filterVariantsList(covVariants.toSet, keep = true)
      .rdd
      .map { case (v, (va, gs)) => (v,
        if (!useDosages)
          RegressionUtils.hardCalls(gs, nMaskedSamples, sampleMaskBc.value).toArray
        else
          RegressionUtils.dosages(gs, nMaskedSamples, sampleMaskBc.value).toArray)
      }
      .collect()

    if (covVariantWithGenotypes.size < covVariants.size) {
      val missingVariants = covVariants.toSet.diff(covVariantWithGenotypes.map(_._1).toSet)
      throw new RestFailure(s"VDS does not contain variant ${ plural(missingVariants.size, "covariate") } ${ missingVariants.mkString(", ") }")
    }

    if (!covVariants.map(_.locus).forall(window.contains)) {
      val outlierVariants = covVariants.filter(v => !window.contains(v.locus))
      warn(s"Window ${ windowToString(window) } does not contain variant ${ plural(outlierVariants.size, "covariate") } ${ outlierVariants.mkString(", ") }. This may increase latency.")
    }

    var variantIndex = 0
    while (variantIndex < covVariants.size) {
      val thisCovGenotypes = covVariantWithGenotypes(variantIndex)._2
      var maskedSampleIndex = 0
      while (maskedSampleIndex < nMaskedSamples) {
        covArray(arrayIndex) = thisCovGenotypes(maskedSampleIndex)
        arrayIndex += 1
        maskedSampleIndex += 1
      }
      variantIndex += 1
    }
    val cov = new DenseMatrix[Double](nMaskedSamples, nCovs, covArray)

    (yOpt, cov)
  }
}

class RestFailure(message: String) extends Exception(message) {
  info(s"RestFailure: $message")
}

case class VariantFilter(operand: String,
  operator: String,
  value: String,
  operand_type: String)

case class Covariate(`type`: String,
  name: Option[String],
  chrom: Option[String],
  pos: Option[Int],
  ref: Option[String],
  alt: Option[String])

case class SingleVariant(chrom: Option[String],
  pos: Option[Int],
  ref: Option[String],
  alt: Option[String])