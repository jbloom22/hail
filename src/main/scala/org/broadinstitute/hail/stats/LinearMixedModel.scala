package org.broadinstitute.hail.stats

import breeze.linalg._
import breeze.numerics.sqrt
import org.apache.spark.mllib.linalg.{DenseMatrix => SDenseMatrix}
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.{TDouble, TStruct, Type}
import org.broadinstitute.hail.methods.{ToSparseRDD, ToStandardizedIndexedRowMatrix}
import org.broadinstitute.hail.variant.{Variant, VariantDataset}
import org.broadinstitute.hail.utils._

object LMM {
  def applyVds(vdsAssoc: VariantDataset,
    vdsKernel: VariantDataset,
    C: DenseMatrix[Double],
    y: DenseVector[Double],
    optDelta: Option[Double] = None,
    useREML: Boolean = true): LMMResult = {

    val Wt = ToStandardizedIndexedRowMatrix(vdsKernel)._2 // W is samples by variants, Wt is variants by samples
    val G = ToSparseRDD(vdsAssoc)

    LMM(Wt, G, C, y, optDelta, useREML)
  }

  def progressReport(msg: String) = {
    val prog = s"\nprogresss: $msg, ${formatTime(System.nanoTime())}\n"
//    println(prog)
    log.info(prog)
  }

  def apply(Wt: IndexedRowMatrix,
    G: RDD[(Variant, Vector[Double])],
    C: DenseMatrix[Double],
    y: DenseVector[Double],
    optDelta: Option[Double] = None,
    useREML: Boolean = true): LMMResult = {

    assert(Wt.numCols == y.length)

    val n = y.length
    val m = Wt.numRows()
    progressReport(s"Computing kernel for $n samples using $m variants....")

    val Kspark = Wt.computeGramianMatrix().asInstanceOf[SDenseMatrix]
    // val blockWt = Wt.toBlockMatrix().cache()
    //    val Kspark = blockWt.transpose.multiply(blockWt).toLocalMatrix().asInstanceOf[SDenseMatrix]
    //    assert(Kspark.numCols == n && Kspark.numRows == n)

    val K = new DenseMatrix[Double](n, n, Kspark.values)

    progressReport("Computing SVD... ") // should use better Lapack method

    val eigK = eigSymD(K)
    val Ut = eigK.eigenvectors.t
    val S = eigK.eigenvalues //place S in global annotations?
    assert(S.length == n)

    progressReport("Largest evals: " + (0 until math.min(n, 10)).map(S(_).formatted("%.5f")).mkString(", "))
    progressReport("Smallest evals: " + ((n - 1) to math.max(0, n - 10) by -1).map(S(_).formatted("%.5f")).mkString(", "))

    progressReport(s"Estimating delta using ${if (useREML) "REML" else "ML"}... ")

    val UtC = Ut * C
    val Uty = Ut * y

    val diagLMM = DiagLMM(UtC, Uty, S, optDelta, useREML)

    progressReport(s"delta = ${diagLMM.delta}")

    // temporary
    val header = "rank\teval"
    val evalString = (0 until n).map(i => s"$i\t${S(i)}").mkString("\n")
//    log.info(s"\nEIGENVALUES\n$header\n$evalString\n\n")
//    val bw = new java.io.BufferedWriter(new java.io.FileWriter(new java.io.File("evals.tsv")))
//    bw.write(s"$header\n$evalString\n")
//    bw.close()

//    val diagLMMBc = genotypes.rows.sparkContext.broadcast(diagLMM)
//    val Uspark = new SDenseMatrix(n, n, U.data)
//    val lmmResult = genotypes.multiply(Uspark).rows.map(r =>
//      (variants(r.index.toInt), diagLMMBc.value.likelihoodRatioTest(toBDenseVector(r.vector.toDense))))

    progressReport(s"Computing LMM statistics for each variant...")

    val T = Ut(::,*) :* diagLMM.sqrtInvD
    val Qt = qr.reduced.justQ(diagLMM.TC).t
    val QtTy = Qt * diagLMM.Ty
    val TyQtTy = (diagLMM.Ty dot diagLMM.Ty) - (QtTy dot QtTy)

    val sc = G.sparkContext
    val TBc = sc.broadcast(T)

    val scalerLMMBc = sc.broadcast(ScalerLMM(diagLMM.Ty, diagLMM.TyTy, Qt, QtTy, TyQtTy, diagLMM.logNullS2, useREML))

    val lmmResult = G.mapValues(x => scalerLMMBc.value.likelihoodRatioTest(TBc.value * x))

    println(formatTime(System.nanoTime()))

    LMMResult(diagLMM, lmmResult)
  }
}

object DiagLMM {
  def apply(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], optDelta: Option[Double] = None, useREML: Boolean = true): DiagLMM = {
    require(C.rows == y.length)

    val delta = optDelta.getOrElse(fitDelta(C, y, S, useREML)._1)

    val n = y.length
    val sqrtInvD = sqrt(S + delta).map(1 / _)
    val TC = C(::, *) :* sqrtInvD
    val Ty = y :* sqrtInvD
    val TyTy = Ty dot Ty
    val TCTy = TC.t * Ty
    val TCTC = TC.t * TC
    val b = TCTC \ TCTy
    val s2 = (TyTy - (TCTy dot b)) / (if (useREML) n - C.cols else n)

    DiagLMM(b, s2, math.log(s2), delta, sqrtInvD, TC, Ty, TyTy, useREML)
  }

  def fitDelta(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], useREML: Boolean): (Double, IndexedSeq[(Double, Double)]) = {

    val logmin = -10
    val logmax = 10
    val pointsPerUnit = 100 // number of points per unit of log space

    val grid = (logmin * pointsPerUnit to logmax * pointsPerUnit).map(_.toDouble / pointsPerUnit) // avoids rounding of (logmin to logmax by logres)

    val n = y.length
    val c = C.cols

    // up to constant shift and scale by 2
    def negLogLkhd(delta: Double, useREML: Boolean): Double = {
      val D = S + delta
      val dy = y :/ D
      val ydy = y dot dy
      val Cdy = C.t * dy
      val CdC = C.t * (C(::, *) :/ D)
      val b = CdC \ Cdy
      val r = ydy - (Cdy dot b)

      if (useREML)
        sum(breeze.numerics.log(D)) + (n - c) * math.log(r) + logdet(CdC)._2
      else
        sum(breeze.numerics.log(D)) + n * math.log(r)
    }

    val gridVals = grid.map(logDelta => (logDelta, negLogLkhd(math.exp(logDelta), useREML)))

    // temporarily included to inspect delta optimization
    // perhaps interesting to return "curvature" at maximum
//    val header = "logDelta\tnegLogLkhd"
//    val gridValsString = gridVals.map{ case (d, nll) => s"${d.formatted("%.4f")}\t$nll" }.mkString("\n")
//    val bw = new java.io.BufferedWriter(new java.io.FileWriter(new java.io.File("delta.tsv")))
//    bw.write(s"$header\n$gridValsString\n")
//    bw.close()

//    log.info(s"\nDELTAVALUES\n$header\n$gridValsString\n\n")

    val logDelta = gridVals.minBy(_._2)._1

    if (logDelta == logmin)
      fatal(s"failed to fit delta: maximum likelihood at lower search boundary e^$logmin")
    else if (logDelta == logmax)
      fatal(s"failed to fit delta: maximum likelihood at upper search boundary e^$logmax")

    (math.exp(logDelta), gridVals)
  }
}

case class DiagLMM(
  nullB: DenseVector[Double],
  nullS2: Double,
  logNullS2: Double,
  delta: Double,
  sqrtInvD: DenseVector[Double],
  TC: DenseMatrix[Double],
  Ty: DenseVector[Double],
  TyTy: Double,
  useREML: Boolean)

case class ScalerLMM(
  y: DenseVector[Double],
  yy: Double,
  Qt: DenseMatrix[Double],
  Qty: DenseVector[Double],
  yQty: Double,
  logNullS2: Double,
  useREML: Boolean) {

  def likelihoodRatioTest(x: Vector[Double]): LMMStat = {

    val n = y.length
    val Qtx = Qt * x
    val xQtx: Double = (x dot x) - (Qtx dot Qtx)
    val xQty: Double = (x dot y) - (Qtx dot Qty)

    val b: Double = xQty / xQtx
    val s2 = (yQty - xQty * b) / (if (useREML) n - Qt.rows else n)
    val chi2 = n * (logNullS2 - math.log(s2))
    val p = chiSquaredTail(1, chi2)

    LMMStat(b, s2, chi2, p)
  }
}


object LMMStat {
  def `type`: Type = TStruct(
    ("beta", TDouble),
    ("sigmaG2", TDouble),
    ("chi2", TDouble),
    ("pval", TDouble))
}

case class LMMStat(b: Double, s2: Double, chi2: Double, p: Double) {
  def toAnnotation: Annotation = Annotation(b, s2, chi2, p)
}

case class LMMResult(diagLMM: DiagLMM, rdd: RDD[(Variant, LMMStat)])