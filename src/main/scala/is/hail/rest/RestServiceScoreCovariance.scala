package is.hail.rest

import breeze.linalg.DenseMatrix
import is.hail.stats.{RegressionUtils, ToNormalizedIndexedRowMatrix}
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

case class GetScoreCovarianceRequest(passback: Option[String],
  md_version: Option[String],
  api_version: Int,
  phenotype: Option[String],
  covariates: Option[Array[Covariate]],
  variant_filters: Option[Array[VariantFilter]],
  variant_list: Option[Array[SingleVariant]],
  compute_cov: Option[Boolean],
  limit: Option[Int],
  count: Option[Boolean])
  
case class GetScoreCovarianceResult(is_error: Boolean,
  error_message: Option[String],
  passback: Option[String],
  active_variants: Option[Array[SingleVariant]],
  scores: Option[Array[Double]],
  covariance: Option[Array[Double]],
  sigma_sq: Option[Double],
  nsamples: Option[Int],
  count: Option[Int])

class RestServiceScoreCovariance(vds: VariantDataset, covariates: Array[String], useDosages: Boolean, maxWidth: Int, hardLimit: Int) { 
  private val nSamples: Int = vds.nSamples
  private val sampleMask: Array[Boolean] = Array.ofDim[Boolean](nSamples)
  private val availableCovariates: Set[String] = covariates.toSet
  private val availableCovariateToIndex: Map[String, Int] = covariates.zipWithIndex.toMap
  private val (sampleIndexToPresentness: Array[Array[Boolean]], 
                 covariateIndexToValues: Array[Array[Double]]) = RegressionUtils.getSampleAndCovMaps(vds, covariates)
  
  def getStats(req: GetScoreCovarianceRequest): GetScoreCovarianceResult = {
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
    
    val covNames = covNamesSet.toArray
    val covVariants = covVariantsSet.toArray
    
    val yNameOpt = req.phenotype
      
    yNameOpt.foreach { yName =>
      if (!availableCovariates(yName))
        throw new RestFailure(s"$yName is not a valid phenotype name")
      if (covNamesSet(yName))
        throw new RestFailure(s"$yName appears as both response phenotype and covariate")
    }

    // construct variant filters
    var chrom = ""
    
    var minPos = 1
    var maxPos = Int.MaxValue // 2,147,483,647 is greater than length of longest chromosome

    var minMAC = 1
    var maxMAC = Int.MaxValue

    val nonNegIntRegEx = """\d+""".r
    
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
          else if (!nonNegIntRegEx.matches(f.value))
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
          else if (!nonNegIntRegEx.matches(f.value))
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
    
    info(s"Using window ${RestServer.windowToString(window)} of size ${width + 1}")

    val variantsSet = mutable.Set[Variant]()
    
    req.variant_list.foreach { _.foreach { sv =>
      val v =
        if (sv.chrom.isDefined && sv.pos.isDefined && sv.ref.isDefined && sv.alt.isDefined) {
          val v = Variant(sv.chrom.get, sv.pos.get, sv.ref.get, sv.alt.get)
          if (v.contig != chrom || v.start < minPos || v.start > maxPos)
            throw new RestFailure(s"Variant ${v.toString} from 'variant_list' is not in the window ${RestServer.windowToString(window)}")
          variantsSet += v
        }
        else
          throw new RestFailure(s"All variants in 'variant_list' must include 'chrom', 'pos', 'ref', and 'alt' fields: got ${(sv.chrom.getOrElse("NA"), sv.pos.getOrElse("NA"), sv.ref.getOrElse("NA"),sv.alt.getOrElse("NA"))}")
        }
      }
    
    // filter and compute
    val filteredVds =
      if (variantsSet.isEmpty)
        vds.filterIntervals(IntervalTree(Array(window)), keep = true)
      else
        vds.filterVariantsList(variantsSet.toSet, keep = true)

    if (req.count.getOrElse(false)) {
      val count = filteredVds.countVariants().toInt
      GetScoreCovarianceResult(is_error = false, None, req.passback, None, None, None, None, None, Some(count))
    }
    else {
      val (yOpt, cov) = RestServer.getYCovAndSetMask(sampleMask, vds, window, yNameOpt, covNames, covVariants,
        useDosages, availableCovariates, availableCovariateToIndex, sampleIndexToPresentness, covariateIndexToValues)    
      
      val activeVariants = filteredVds.variants.collect().map( v =>
        SingleVariant(Some(v.contig), Some(v.start), Some(v.ref), Some(v.alt)))
      
//      val covariance = filteredVds.ldMatrix().matrix.toBlockMatrix().toLocalMatrix().toArray // FIXME: replace with upper triangular

      val X = ToNormalizedIndexedRowMatrix(filteredVds)
      
      val Xb = X.toBlockMatrix()
      
      val covariance = Xb.multiply(Xb.transpose).toLocalMatrix().toArray
      
      import org.apache.spark.mllib.linalg.{DenseMatrix => SparkDenseMatrix}
      
      val scoresOpt: Option[Array[Double]] = yOpt.map { y =>
        val yMat = new SparkDenseMatrix(1, y.length, y.toArray, true)
        X.multiply(yMat).toBlockMatrix().toLocalMatrix().toArray
      }
      
      val sigmaSqOpt: Option[Double] = yOpt.map(stuff => 0d) // FIXME
      
//      val sc = filteredVds.sparkContext
//      val yOptBc = sc.broadcast(yOpt)
//      val covBc = sc.broadcast(cov)
//      
//      val restStatsRDD = filteredVds.map { case (v, (va, gs) => 
//        val yOpt = yOptBc.value
//        val cov = covBc.value
//        
//      }
//      
//      var restStats =
//        if (req.limit.isEmpty)
//          restStatsRDD.collect() // avoids first pass of take, modify if stats exceed memory
//        else {
//          val limit = req.limit.get
//          if (limit < 0)
//            throw new RestFailure(s"limit must be non-negative: got $limit")
//          restStatsRDD.take(limit)
//        }
//
//      if (restStats.size > hardLimit)
//        restStats = restStats.take(hardLimit)
//      }
    
      GetScoreCovarianceResult(is_error = false, None, req.passback,
        Some(activeVariants),
        scoresOpt,
        Some(covariance),
        sigmaSqOpt,
        Some(cov.rows),
        Some(activeVariants.length))
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
          val getStatsReq = read[GetScoreCovarianceRequest](text)
          passback = getStatsReq.passback
          val result = getStats(getStatsReq)
          Ok(write(result))
            .putHeaders(`Content-Type`(`application/json`))
        } catch {
          case e: Exception =>
            val result = GetScoreCovarianceResult(is_error = true, Some(e.getMessage), passback, None, None, None, None, None, None)
            BadRequest(write(result))
              .putHeaders(`Content-Type`(`application/json`))
        }
      }
  }
}