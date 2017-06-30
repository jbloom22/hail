package is.hail.rest

import is.hail.methods.LinearRegression
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

class RestFailure(message: String) extends Exception(message)

class RestService(vds: VariantDataset, phenoTable: PhenotypeTable, maxWidth: Int, hardLimit: Int) {
  val availablePhenotypes: Set[String] = phenoTable.phenotypes.toSet  

  def getStats(req: GetStatsRequest): GetStatsResult = {
    req.md_version.foreach { md_version =>
      if (md_version != "mdv1")
        throw new RestFailure(s"Unknown md_version `$md_version'. Available md_versions: mdv1")
    }

    if (req.api_version != 1)
      throw new RestFailure(s"Unsupported API version `${req.api_version}'. Supported API versions: 1")

    // check pheno and covs
    val yName = req.phenotype.getOrElse("T2D") // FIXME: remove default, change spec
    val covNamesSet = mutable.Set[String]()
    val covVariantsSet = mutable.Set[Variant]()
    
    req.covariates.foreach { covariates =>
      for (c <- covariates)
        c.`type` match {
          case "phenotype" =>
            c.name match {
              case Some(name) =>
                if (availablePhenotypes(name))
                  covNamesSet += name
                else
                  throw new RestFailure(s"$name is not a valid phenotype")
              case None =>
                throw new RestFailure("Covariate of type 'phenotype' must include 'name' field in request")
            }
          case "variant" =>
            (c.chrom, c.pos, c.ref, c.alt) match {
              case (Some(chr), Some(pos), Some(ref), Some(alt)) =>
                covVariantsSet += Variant(chr, pos, ref, alt)
              case missingFields =>
                throw new RestFailure("Covariate of type 'variant' must include 'chrom', 'pos', 'ref', and 'alt' fields in request")
            }
          case other =>
            throw new RestFailure(s"Covariate type must be 'phenotype' or 'variant': got $other")
        }
    }
    
    if (covNamesSet(yName))
      throw new RestFailure(s"$yName appears as both response phenotype and a covariate phenotype")

    val covNames = covNamesSet.toArray
    val covVariants = covVariantsSet.toArray

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
            case "eq" => minPos = f.value.toInt; maxPos = f.value.toInt
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
    
    val window = Interval(Locus(chrom, minPos), Locus(chrom, maxPos))
    
    println(s"Using window: $window")
    
    val filteredVds = vds.filterIntervals(IntervalTree(Array(window)), keep = true)       
    val onlyCount = req.count.getOrElse(false)
    
    if (onlyCount) {
      val count = filteredVds.countVariants().toInt
      GetStatsResult(is_error = false, None, req.passback, None, None, Some(count)) // FIXME: make sure consistent with spec
    }
    else {
      val (restStatsRDD, nSamplesKept) = LinearRegression.restApply(filteredVds, phenoTable, yName, covNames, minMAC, maxMAC)

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

      // FIXME: is standard sorting now preserved with OrderedRDD?

      if (req.sort_by.isEmpty)
        restStats = restStats.sortBy(s => (s.pos, s.ref, s.alt))
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
      GetStatsResult(is_error = false, None, req.passback, Some(restStats), Some(nSamplesKept), Some(restStats.size))
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