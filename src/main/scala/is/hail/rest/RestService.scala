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
  `p-value`: Option[Double]) // FIXME: rename pval

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
  val (sampleMap: Array[Array[Boolean]], allCovariateData: Map[String, Array[Double]]) = RegressionUtils.getSampleAndCovMaps(vds, covariates)
  val availableCovariates: Set[String] = allCovariateData.keySet
  val covariateIndices: Map[String, Int] = covariates.zipWithIndex.toMap
  val nSamples: Int = vds.nSamples
  
  def getStats(req: GetStatsRequest): GetStatsResult = {
    req.md_version.foreach { md_version =>
      if (md_version != "mdv1")
        throw new RestFailure(s"Unknown md_version `$md_version'. Available md_versions: mdv1")
    }

    if (req.api_version != 1)
      throw new RestFailure(s"Unsupported API version `${req.api_version}'. Supported API versions: 1")

    // check and construct pheno and covNames, and covVariants
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
    
    assert(covNamesSet.size == covNames.size)
    if (covNamesSet(yName))
      throw new RestFailure(s"$yName appears as both response phenotype and covariate")

    // check and construct variant filters
    var chrom = ""
    
    var minPos = 0
    var maxPos = Int.MaxValue // 2,147,483,647 is greater than length of longest chromosome

    var minMAC = 0
    var maxMAC = Int.MaxValue

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
          f.operator match {
            case "gte" => minPos = minPos max f.value.toInt
            case "gt" => minPos = minPos max (f.value.toInt + 1)
            case "lte" => maxPos = maxPos min f.value.toInt
            case "lt" => maxPos = maxPos min (f.value.toInt - 1)
            case "eq" => minPos = minPos max f.value.toInt; maxPos = maxPos min f.value.toInt
            case other =>
              throw new RestFailure(s"pos filter operator must be 'gte', 'gt', 'lte', 'lt', or 'eq': got '$other'")
          }
        case "mac" =>
          if (f.operand_type != "integer")
            throw new RestFailure(s"mac filter operand_type must be 'integer': got '${f.operand_type}'")
          f.operator match {
            case "gte" => minMAC = minMAC max f.value.toInt
            case "gt" => minMAC = minMAC max (f.value.toInt + 1)
            case "lte" => maxMAC = maxMAC min f.value.toInt
            case "lt" => maxMAC = maxMAC min (f.value.toInt - 1)
            case other =>
              throw new RestFailure(s"mac filter operator must be 'gte', 'gt', 'lte', 'lt': got '$other'")
          }
        case other => throw new RestFailure(s"Filter operand must be 'chrom' or 'pos': got '$other'")
      }
    })

    if (chrom.isEmpty)
      throw new RestFailure("No chromosome specified in variant_filter")
    
    val width = maxPos - minPos
    if (width > maxWidth)
      throw new RestFailure(s"Interval length cannot exceed $maxWidth: got $width")
        
    if (width < 0)
      throw new RestFailure(s"Window is empty: got start $minPos and end $maxPos")
    
    info(s"Window is $chrom:$minPos-$maxPos. Width is $width.")
    
    val window = Interval(Locus(chrom, minPos), Locus(chrom, maxPos + 1))

    val windowedVds = vds.filterIntervals(IntervalTree(Array(window)), keep = true)   
    
    def getPhenoCovMask(yName: String, covNames: Array[String], covVariants: Array[Variant] = Array.empty[Variant]): (DenseVector[Double], DenseMatrix[Double], Array[Boolean]) = {
      // determine sample mask
      val yCovIndices = (yName +: covNames).map(covariateIndices).sorted
      
      val sampleMask = Array.ofDim[Boolean](nSamples)
      var nMaskedSamples = 0
     
      var i = 0
      while (i < nSamples) {
        val include = yCovIndices.forall(sampleMap(i)) // FIXME: this doesn't short circuit...
        if (include) {
          sampleMask(i) = true
          nMaskedSamples += 1
        }  
        i += 1
      }
      
      // phenotype y
      val yArray = Array.ofDim[Double](nMaskedSamples)
      val yAllSamples = allCovariateData(yName)
      var j = 0
      var k = 0
      while (j < nSamples) {
        if (sampleMask(j)) {
          yArray(k) = yAllSamples(j)
          k += 1
        }
        j += 1
      }

      // covariates
      val nCovs = 1 + covNames.size + covVariants.size
      val covArray = Array.ofDim[Double](nMaskedSamples * nCovs)
      
      // intercept
      i = 0
      while (i < nMaskedSamples) {
        covArray(i) = 1
        i += 1
      }

      // phenotypic covariates
      k = nMaskedSamples
      covNames.foreach { covName =>
        val covAllSamples = allCovariateData(covName)
        j = 0        
        while (j < nSamples) {
          if (sampleMask(j)) {
            covArray(k) = covAllSamples(j)
            k += 1
          }
          j += 1
        }
      }
      
      val covVariantsImmutSet = covVariantsSet.toSet
      
      // variant covariates FIXME: add variant covariates
      val covVariantGenotypes = windowedVds
        .filterVariantsList(covVariantsImmutSet, keep = true)
        .rdd
        .map { case (v, (va, gs)) => (v, RegressionUtils.hardCalls(gs, nMaskedSamples, sampleMask).toArray) } // FIXME can speed up
        .collect()
      
      if (covVariantGenotypes.size < covVariants.size) {
        val missingVariants = covVariantsImmutSet.diff(covVariantGenotypes.map(_._1).toSet)
        throw new RestFailure(s"$missingVariants are missing from window or vds")
      }  
      
      i = 0
      while (i < covVariants.size) { 
        j = 0
        while (j < nMaskedSamples) {
          covArray(k) = covVariantGenotypes(i)._2(j)
          j += 1
          k += 1
        }
        i += 1
      }
      
      assert(k == covArray.size)
      
      val y = DenseVector(yArray)
      val cov = new DenseMatrix[Double](nMaskedSamples, nCovs, covArray)
      (y, cov, sampleMask)
    }
    
    if (req.count.getOrElse(false)) {
      val count = windowedVds.countVariants().toInt
      GetStatsResult(is_error = false, None, req.passback, None, None, Some(count)) // FIXME: make sure consistent with spec
    }
    else {
      val (y, cov, sampleMask) = getPhenoCovMask(yName, covNames, covVariants)    
      
      val restStatsRDD = LinearRegression.applyRest(windowedVds, y, cov, sampleMask, minMAC, maxMAC)

      var restStats =
        if (req.limit.isEmpty)
          restStatsRDD.collect() // avoids first pass of take, modify if stats grows beyond memory capacity
        else {
          val limit = req.limit.get
          if (limit < 0)
            throw new RestFailure(s"limit must be non-negative: got $limit")
          restStatsRDD.take(limit)
        }

      if (restStats.size > hardLimit)
        restStats = restStats.take(hardLimit)

      if (req.sort_by.isEmpty)
        restStats = restStats.sortBy(s => (s.pos, s.ref, s.alt)) // FIXME: may be able to remove now
      else {
        val sortFields = req.sort_by.get
        if (!sortFields.areDistinct())
          throw new RestFailure("sort_by arguments must be distinct")

        //      var fields = a.toList
        //
        //      // Default sort order is [pos, ref, alt] and sortBy is stable
        //      if (fields.nonEmpty && fields.head == "pos") {
        //        fields = fields.tail
        //        if (fields.nonEmpty && fields.head == "ref") {
        //          fields = fields.tail
        //          if (fields.nonEmpty && fields.head == "alt")
        //            fields = fields.tail
        //        }
        //      }

        sortFields.reverse.foreach { f =>
          restStats = f match {
            case "pos" => restStats.sortBy(_.pos)
            case "ref" => restStats.sortBy(_.ref)
            case "alt" => restStats.sortBy(_.alt)
            case "p-value" => restStats.sortBy(_.`p-value`.getOrElse(2d))
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