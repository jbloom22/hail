package is.hail.methods

import breeze.linalg._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.rest.RestStat
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import net.sourceforge.jdistlib.T
import org.apache.spark.rdd.RDD

object LinearRegression {
  val schema = TStruct(
    ("beta", TDouble),
    ("se", TDouble),
    ("tstat", TDouble),
    ("pval", TDouble))

  def apply(vds: VariantDataset, yExpr: String, covExpr: Array[String], root: String, useDosages: Boolean, minAC: Int, minAF: Double): VariantDataset = {
    require(vds.wasSplit)

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
    val sampleMask = vds.sampleIds.map(completeSamples.toSet).toArray

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (minAC < 1)
      fatal(s"Minumum alternate allele count must be a positive integer, got $minAC")
    if (minAF < 0 || minAF > 1)
      fatal(s"Minumum alternate allele frequency must lie in [0.0, 1.0], got $minAF")
    val combinedMinAC = math.max(minAC, (math.ceil(2 * n * minAF) + 0.5).toInt)

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"Running linear regression on $n samples with $k ${ plural(k, "covariate") } including intercept...")

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y

    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast((y dot y) - (Qty dot Qty))

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val (newVAS, inserter) = vds.insertVA(LinearRegression.schema, pathVA)
    
    vds.mapAnnotations { case (v, va, gs) =>
      val (x: Vector[Double], ac) =
        if (!useDosages) // replace by hardCalls in 0.2, with ac post-imputation
          RegressionUtils.hardCallsWithAC(gs, n, sampleMaskBc.value)
        else {
          val x0 = RegressionUtils.dosages(gs, n, sampleMaskBc.value)
          (x0, sum(x0))
        }

      // constant checking to be removed in 0.2
      val nonConstant = useDosages || !RegressionUtils.constantVector(x)
      
      val linregAnnot =
        if (ac >= combinedMinAC && nonConstant)
          LinearRegressionModel.fit(x, yBc.value, yypBc.value, QtBc.value, QtyBc.value, d)
        else
          null

      val newAnnotation = inserter(va, linregAnnot)
      assert(newVAS.typeCheck(newAnnotation))
      newAnnotation
    }.copy(vaSignature = newVAS)
  }
  
  def applyRest(vds: VariantDataset, y: DenseVector[Double], cov: DenseMatrix[Double],
    sampleMask: Array[Boolean], minMAC: Int, maxMAC: Int): RDD[RestStat] = {
    
    require(vds.wasSplit)
    require(minMAC >= 0 && maxMAC >= minMAC)

    val n = y.size
    val k = cov.cols
    val d = n - k - 1
    
    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")
    
    val lowerMinAC = 1 max minMAC
    val lowerMaxAC = n min maxMAC
    val upperMinAC = 2 * n - lowerMaxAC
    val upperMaxAC = 2 * n - lowerMinAC
    
    def inRange(ac: Int): Boolean = (ac >= lowerMinAC && ac <= lowerMaxAC) || (ac >= upperMinAC && ac <= upperMaxAC)
    
    info(s"Running linear regression on $n samples with $k ${ plural(k, "covariate") } including intercept...")

    val Qt = qr.reduced.justQ(cov).t
    val Qty = Qt * y

    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val yBc = sc.broadcast(y)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast((y dot y) - (Qty dot Qty))

    vds.rdd.map { case (v, (va, gs)) =>
      val (x: SparseVector[Double], ac) = RegressionUtils.hardCallsWithAC(gs, n, sampleMaskBc.value)

      val optPval =
        if (inRange(ac.toInt)) {
          val qtx = QtBc.value * x
          val xxp = (x dot x) - (qtx dot qtx)
          val xyp = (x dot yBc.value) - (qtx dot QtyBc.value)

          val b = xyp / xxp
          val se = math.sqrt((yypBc.value / xxp - b * b) / d)
          val t = b / se
          val p = 2 * T.cumulative(-math.abs(t), d, true, false)

          if (!p.isNaN)
            Some(p)
          else
            None
        }
        else
          None

      RestStat(v.contig, v.start, v.ref, v.alt, optPval)
    }
  }
}