package is.hail.rest

import breeze.linalg.{DenseMatrix, DenseVector}
import is.hail.methods.LinearRegression
import is.hail.stats.RegressionUtils
import is.hail.variant._
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.http4s.headers.`Content-Type`
import org.http4s._
import org.http4s.MediaType._
import org.http4s.dsl._
import org.http4s.server._

import scala.concurrent.ExecutionContext
import org.json4s._
import org.json4s.jackson.Serialization
import org.json4s.jackson.Serialization.{read, write}

import scala.collection.mutable

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

case class GetStatsRequest(passback: Option[String],
  md_version: Option[String],
  api_version: Int,
  phenotype: Option[String],
  covariates: Option[Array[Covariate]],
  variant_filters: Option[Array[VariantFilter]],
  limit: Option[Int],
  count: Option[Boolean],
  sort_by: Option[Array[String]])

case class RestStat(chrom: String,
  pos: Int,
  ref: String,
  alt: String,
  `p-value`: Option[Double])

case class GetStatsResult(is_error: Boolean,
  error_message: Option[String],
  passback: Option[String],
  stats: Option[Array[RestStat]],
  nsamples: Option[Int],
  count: Option[Int])

class RestFailure(message: String) extends Exception(message) {
  info(s"RestFailure: $message")
}

class RestService(vds: VariantDataset, covariates: Array[String], maxWidth: Int, hardLimit: Int) { 
  private val nSamples: Int = vds.nSamples
  private val sampleMask: Array[Boolean] = Array.ofDim[Boolean](nSamples)
  private val availableCovariates: Set[String] = covariates.toSet
  private val availableCovariateToIndex: Map[String, Int] = covariates.zipWithIndex.toMap
  private val (sampleIndexToPresentness: Array[Array[Boolean]], 
                 covariateIndexToValues: Array[Array[Double]]) = RegressionUtils.getSampleAndCovMaps(vds, covariates)
 
  def windowToString(window: Interval[Locus]): String =
    s"${window.start.contig}:${window.start.position}-${window.end.position - 1}"
  
  def getYCovSampleMask(window: Interval[Locus],yName: String, covNames: Array[String],
    covVariants: Array[Variant] = Array.empty[Variant]): (DenseVector[Double], DenseMatrix[Double], Array[Boolean]) = {
    
    // set sample mask
    val yCovIndices = (yName +: covNames).map(availableCovariateToIndex).sorted
    
    var nMaskedSamples = 0
    var sampleIndex = 0
    while (sampleIndex < nSamples) {
      val include = yCovIndices.forall(sampleIndexToPresentness(sampleIndex))
      sampleMask(sampleIndex) = include
      if (include) nMaskedSamples += 1
      sampleIndex += 1
    }
    
    // set y
    val yArray = Array.ofDim[Double](nMaskedSamples)
    val yData = covariateIndexToValues(availableCovariateToIndex(yName))
    
    var arrayIndex = 0
    sampleIndex = 0
    while (sampleIndex < nSamples) {
      if (sampleMask(sampleIndex)) {
        yArray(arrayIndex) = yData(sampleIndex)
        arrayIndex += 1
      }
      sampleIndex += 1
    }
    
    val y = DenseVector(yArray)

    // set cov
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
      .map { case (v, (va, gs)) => (v, RegressionUtils.hardCalls(gs, nMaskedSamples, sampleMaskBc.value).toArray) }
      .collect()
    
    if (covVariantWithGenotypes.size < covVariants.size) {
      val missingVariants = covVariants.toSet.diff(covVariantWithGenotypes.map(_._1).toSet)
      throw new RestFailure(s"VDS does not contain variant ${plural(missingVariants.size, "covariate")} ${missingVariants.mkString(", ")}")
    }
    
    if (!covVariants.map(_.locus).forall(window.contains)) {
      val outlierVariants = covVariants.filter(v => !window.contains(v.locus))
      warn(s"Window ${windowToString(window)} does not contain variant ${plural(outlierVariants.size, "covariate")} ${outlierVariants.mkString(", ")}. This may increase latency.")
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
    
    (y, cov, sampleMask)
  }
  
  def getStats(req: GetStatsRequest): GetStatsResult = {
    req.md_version.foreach { md_version =>
      if (md_version != "mdv1")
        throw new RestFailure(s"Unknown md_version `$md_version'. Available md_versions: mdv1")
    }

    if (req.api_version != 1)
      throw new RestFailure(s"Unsupported API version `${req.api_version}'. Supported API versions: 1")

    // construct yName, covNames, and covVariants
    val covNamesSet = mutable.Set[String]()
    val covVariantsSet = mutable.Set[Variant]()
    
    req.covariates.foreach { covariates =>
      for (c <- covariates)
        c.`type` match {
          case "phenotype" =>
            c.name match {
              case Some(name) =>
                if (availableCovariates(name))
                  if (covNamesSet(name))
                    throw new RestFailure(s"Covariate $name is included as a covariate more than once")
                  else
                    covNamesSet += name
                else
                  throw new RestFailure(s"$name is not a valid covariate name")
              case None =>
                throw new RestFailure("Covariate of type 'phenotype' must include 'name' field in request")
            }
          case "variant" =>
            (c.chrom, c.pos, c.ref, c.alt) match {
              case (Some(chr), Some(pos), Some(ref), Some(alt)) =>
                val v = Variant(chr, pos, ref, alt)
                if (covVariantsSet(v))
                  throw new RestFailure(s"$v is included as a covariate more than once")
                else  
                  covVariantsSet += v
              case missingFields =>
                throw new RestFailure("Covariate of type 'variant' must include 'chrom', 'pos', 'ref', and 'alt' fields in request")
            }
          case other =>
            throw new RestFailure(s"Covariate type must be 'phenotype' or 'variant': got $other")
        }
    }
    
    val yName = req.phenotype.getOrElse {
      throw new RestFailure("Missing required field: phenotype")
    }
    if (!availableCovariates(yName))
      throw new RestFailure(s"$yName is not a valid phenotype name")
    
    val covNames = covNamesSet.toArray
    val covVariants = covVariantsSet.toArray
    
    if (covNamesSet(yName))
      throw new RestFailure(s"$yName appears as both response phenotype and covariate")

    // construct variant filters
    var chrom = ""
    
    var minPos = 1
    var maxPos = Int.MaxValue // 2,147,483,647 is greater than length of longest chromosome

    var minMAC = 0
    var maxMAC = Int.MaxValue

    val nonNegIntRegex = """\d+""".r
    
    req.variant_filters.foreach(_.foreach { f =>
      f.operand match {
        case "chrom" =>
          if (!(f.operator == "eq" && f.operand_type == "string"))
            throw new RestFailure(s"chrom filter operator must be 'eq' and operand_type must be 'string': got '${f.operator}' and '${f.operand_type}'")
          else if (f.value.isEmpty)
            throw new RestFailure("chrom filter value cannot be the empty string")         
          else if (chrom.isEmpty)
            chrom = f.value
          else if (chrom != f.value)
            throw new RestFailure(s"Got incompatible chrom filters: '$chrom' and '${f.value}'")
        case "pos" =>
          if (f.operand_type != "integer")
            throw new RestFailure(s"pos filter operand_type must be 'integer': got '${f.operand_type}'")
          else if (!nonNegIntRegex.matches(f.value))
            throw new RestFailure(s"Value of position in variant_filter must be a valid non-negative integer: got '${f.value}'")
          else {
            val pos = f.value.toInt
            f.operator match {
              case "gte" => minPos = minPos max pos
              case "gt" => minPos = minPos max (pos + 1)
              case "lte" => maxPos = maxPos min pos
              case "lt" => maxPos = maxPos min (pos - 1)
              case "eq" => minPos = minPos max pos; maxPos = maxPos min pos
              case other =>
                throw new RestFailure(s"pos filter operator must be 'gte', 'gt', 'lte', 'lt', or 'eq': got '$other'")
            }
          }
        case "mac" =>
          if (f.operand_type != "integer")
            throw new RestFailure(s"mac filter operand_type must be 'integer': got '${f.operand_type}'")
          else if (!nonNegIntRegex.matches(f.value))
            throw new RestFailure(s"mac filter value must be a valid non-negative integer: got '${f.value}'")
          val mac = f.value.toInt
          f.operator match {
            case "gte" => minMAC = minMAC max mac
            case "gt" => minMAC = minMAC max (mac + 1)
            case "lte" => maxMAC = maxMAC min mac
            case "lt" => maxMAC = maxMAC min (mac - 1)
            case other =>
              throw new RestFailure(s"mac filter operator must be 'gte', 'gt', 'lte', 'lt': got '$other'")
          }
        case other => throw new RestFailure(s"Filter operand must be 'chrom' or 'pos': got '$other'")
      }
    })

    // construct window
    if (chrom.isEmpty)
      throw new RestFailure("No chromosome specified in variant_filter")
    
    val width = maxPos - minPos
    if (width > maxWidth)
      throw new RestFailure(s"Interval length cannot exceed $maxWidth: got $width")
        
    if (width < 0)
      throw new RestFailure(s"Window is empty: got start $minPos and end $maxPos")
    
    val window = Interval(Locus(chrom, minPos), Locus(chrom, maxPos + 1))
    
    info(s"Using window ${windowToString(window)} of size ${width + 1}")

    // filter and computer
    val windowedVds = vds.filterIntervals(IntervalTree(Array(window)), keep = true)

    if (req.count.getOrElse(false)) {
      val count = windowedVds.countVariants().toInt
      GetStatsResult(is_error = false, None, req.passback, None, None, Some(count))
    }
    else {
      val (y, cov, sampleMask) = getYCovSampleMask(window, yName, covNames, covVariants)    
      val restStatsRDD = LinearRegression.applyRest(windowedVds, y, cov, sampleMask, minMAC, maxMAC)

      var restStats =
        if (req.limit.isEmpty)
          restStatsRDD.collect() // avoids first pass of take, modify if stats exceed memory
        else {
          val limit = req.limit.get
          if (limit < 0)
            throw new RestFailure(s"limit must be non-negative: got $limit")
          restStatsRDD.take(limit)
        }

      if (restStats.size > hardLimit)
        restStats = restStats.take(hardLimit)

      req.sort_by.foreach { sortFields => 
        if (!sortFields.areDistinct())
          throw new RestFailure("sort_by arguments must be distinct")
        
        val nRedundant =
          if (sortFields.endsWith(Array("pos", "ref", "alt")))
            3
          else if (sortFields.endsWith(Array("pos", "ref")))
            2
          else if (sortFields.endsWith(Array("pos")))
            1
          else
            0
        
        sortFields.dropRight(nRedundant).reverse.foreach { f =>
          restStats = f match {
            case "pos" => restStats.sortBy(_.pos)
            case "ref" => restStats.sortBy(_.ref)
            case "alt" => restStats.sortBy(_.alt)
            case "p-value" => restStats.sortBy(_.`p-value`.getOrElse(2d)) // missing values at end
            case _ => throw new RestFailure(s"Valid sort_by arguments are `pos', `ref', `alt', and `p-value': got $f")
          }
        }
      }
      GetStatsResult(is_error = false, None, req.passback, Some(restStats), Some(y.size), Some(restStats.size))
    }
  }

  def service(implicit executionContext: ExecutionContext = ExecutionContext.global): HttpService = Router(
    "" -> rootService)

  def rootService(implicit executionContext: ExecutionContext) = HttpService {
    case _ -> Root =>
      // The default route result is NotFound. Sometimes MethodNotAllowed is more appropriate.
      MethodNotAllowed()

    case req@POST -> Root / "getStats" =>
      println("in getStats")

      req.decode[String] { text =>
        info("request: " + text)

        implicit val formats = Serialization.formats(NoTypeHints)

        var passback: Option[String] = None
        try {
          val getStatsReq = read[GetStatsRequest](text)
          passback = getStatsReq.passback
          val result = getStats(getStatsReq)
          Ok(write(result))
            .putHeaders(`Content-Type`(`application/json`))
        } catch {
          case e: Exception =>
            val result = GetStatsResult(is_error = true, Some(e.getMessage), passback, None, None, None)
            BadRequest(write(result))
              .putHeaders(`Content-Type`(`application/json`))
        }
      }
  }
}