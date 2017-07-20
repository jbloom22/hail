package is.hail.rest

import breeze.linalg.{norm, qr, sum}
import breeze.stats.variance
import is.hail.stats.RegressionUtils
import is.hail.variant._
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.json4s.jackson.Serialization.read

import scala.collection.mutable

case class GetRequestScoreCovariance(passback: Option[String],
  md_version: Option[String],
  api_version: Int,
  phenotype: Option[String],
  covariates: Option[Array[Covariate]],
  variant_filters: Option[Array[VariantFilter]],
  variant_list: Option[Array[SingleVariant]],
  compute_cov: Option[Boolean],
  limit: Option[Int],
  count: Option[Boolean]) extends GetRequest

case class GetResultScoreCovariance(is_error: Boolean,
  error_message: Option[String],
  passback: Option[String],
  active_variants: Option[Array[SingleVariant]],
  scores: Option[Array[Double]],
  covariance: Option[Array[Double]],
  sigma_sq: Option[Double],
  nsamples: Option[Int],
  count: Option[Int]) extends GetResult

class RestServiceScoreCovariance(vds: VariantDataset, covariates: Array[String], useDosages: Boolean, maxWidth: Int, hardLimit: Int) extends RestService { 
  private val nSamples: Int = vds.nSamples
  private val sampleMask: Array[Boolean] = Array.ofDim[Boolean](nSamples)
  private val availableCovariates: Set[String] = covariates.toSet
  private val availableCovariateToIndex: Map[String, Int] = covariates.zipWithIndex.toMap
  private val (sampleIndexToPresentness: Array[Array[Boolean]], 
                 covariateIndexToValues: Array[Array[Double]]) = RegressionUtils.getSampleAndCovMaps(vds, covariates)
  
  def readText(text: String): GetRequestScoreCovariance = read[GetRequestScoreCovariance](text)
  
  def getError(message: String, passback: Option[String]) = GetResultScoreCovariance(is_error = true, Some(message), passback, None, None, None, None, None, None)
  
  def getStats(req0: GetRequest): GetResultScoreCovariance = {
    val req = req0.asInstanceOf[GetRequestScoreCovariance]
    req.md_version.foreach { md_version =>
      if (md_version != "mdv1")
        throw new RestFailure(s"Unknown md_version `$md_version'. Available md_versions: mdv1")
    }

    if (req.api_version != 1)
      throw new RestFailure(s"Unsupported API version `${ req.api_version }'. Supported API versions: 1")

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
            throw new RestFailure(s"chrom filter operator must be 'eq' and operand_type must be 'string': got '${ f.operator }' and '${ f.operand_type }'")
          else if (f.value.isEmpty)
            throw new RestFailure("chrom filter value cannot be the empty string")
          else if (chrom.isEmpty)
            chrom = f.value
          else if (chrom != f.value)
            throw new RestFailure(s"Got incompatible chrom filters: '$chrom' and '${ f.value }'")
        case "pos" =>
          if (f.operand_type != "integer")
            throw new RestFailure(s"pos filter operand_type must be 'integer': got '${ f.operand_type }'")
          else if (!nonNegIntRegEx.matches(f.value))
            throw new RestFailure(s"Value of position in variant_filter must be a valid non-negative integer: got '${ f.value }'")
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
            throw new RestFailure(s"mac filter operand_type must be 'integer': got '${ f.operand_type }'")
          else if (!nonNegIntRegEx.matches(f.value))
            throw new RestFailure(s"mac filter value must be a valid non-negative integer: got '${ f.value }'")
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

    info(s"Using window ${ RestService.windowToString(window) } of size ${ width + 1 }")

    val variantsSet = mutable.Set[Variant]()

    req.variant_list.foreach {
      _.foreach { sv =>
        val v =
          if (sv.chrom.isDefined && sv.pos.isDefined && sv.ref.isDefined && sv.alt.isDefined) {
            val v = Variant(sv.chrom.get, sv.pos.get, sv.ref.get, sv.alt.get)
            if (v.contig != chrom || v.start < minPos || v.start > maxPos)
              throw new RestFailure(s"Variant ${ v.toString } from 'variant_list' is not in the window ${ RestService.windowToString(window) }")
            variantsSet += v
          }
          else
            throw new RestFailure(s"All variants in 'variant_list' must include 'chrom', 'pos', 'ref', and 'alt' fields: got ${ (sv.chrom.getOrElse("NA"), sv.pos.getOrElse("NA"), sv.ref.getOrElse("NA"), sv.alt.getOrElse("NA")) }")
      }
    }

    // filter and compute
    val filteredVds =
      if (variantsSet.isEmpty)
        vds.filterIntervals(IntervalTree(Array(window)), keep = true)
      else
        vds.filterVariantsList(variantsSet.toSet, keep = true)

    if (req.count.getOrElse(false)) {
      val count = filteredVds.countVariants().toInt // FIXME: this does not account for MAC
      GetResultScoreCovariance(is_error = false, None, req.passback, None, None, None, None, None, Some(count))
    }
    else {
      val (yOpt, cov) = RestService.getYCovAndSetMask(sampleMask, vds, window, yNameOpt, covNames, covVariants,
        useDosages, availableCovariates, availableCovariateToIndex, sampleIndexToPresentness, covariateIndexToValues)
            
      val (x, activeVariants) = ToFilteredCenteredIndexedRowMatrix(filteredVds, cov.rows, sampleMask, minMAC, maxMAC)
      
      val limit = req.limit.getOrElse(hardLimit)
      if (activeVariants.length > limit)
        throw new RestFailure(s"Number of active variants $activeVariants exceeds limit $limit")
      
      val covariance =
        if (req.compute_cov.isEmpty || (req.compute_cov.isDefined && req.compute_cov.get)) {
          val Xb = x.toBlockMatrix()
          Some(Xb.multiply(Xb.transpose).toLocalMatrix().toArray)
        } else
          None

      val scoresOpt = yOpt.map { y =>
        val yMat = new org.apache.spark.mllib.linalg.DenseMatrix(y.length, 1, y.toArray, true)
        x.multiply(yMat).toBlockMatrix().toLocalMatrix().toArray
      }

      val sigmaSqOpt: Option[Double] = yOpt.map { y =>
        val d = cov.rows - cov.cols - 1
        val qrd = qr.reduced(cov)
        val beta = qrd.r \ (qrd.q.t * y)
        val res = y - cov * beta     
        
        (res dot res) / d
      }

      GetResultScoreCovariance(is_error = false, None, req.passback,
        Some(activeVariants),
        scoresOpt,
        covariance,
        sigmaSqOpt,
        Some(cov.rows),
        Some(activeVariants.length))
    }
  }
}

// n is nCompleteSamples
object ToFilteredCenteredIndexedRowMatrix {  
  def apply(vds: VariantDataset, n: Int, mask: Array[Boolean], minMAC: Int, maxMAC: Int): (IndexedRowMatrix, Array[SingleVariant]) = {
    val inRange = RestService.inRangeFunc(n, minMAC, maxMAC)

    require(vds.wasSplit)
    val variants = vds.variants.collect()
    val variantIdxBc = vds.sparkContext.broadcast(variants.index)

    val indexedRows = vds.rdd.flatMap { case (v, (_, gs)) =>  // FIXME: consider behavior when all missing
      val (x, ac) = RegressionUtils.hardCallsWithAC(gs, n, mask)
      val mu = sum(x) / n // FIXME: inefficient
      if (inRange(ac))
        Some(IndexedRow(variantIdxBc.value(v), Vectors.dense((x - mu).toArray)))
      else
        None
    }

    val irm = new IndexedRowMatrix(indexedRows, variants.length, n)

    val activeVariants = irm.rows.map(ir => ir.index.toInt).collect().sorted
      .map { i =>
        val v = variants(i)
        SingleVariant(Some(v.contig), Some(v.start), Some(v.ref), Some(v.alt)) // FIXME: error handling
      }
    
    (irm, activeVariants)
  }
}