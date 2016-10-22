package org.broadinstitute.hail.stats

import breeze.linalg._
import breeze.numerics.sqrt
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.{TDouble, TStruct, Type}
import org.broadinstitute.hail.methods.{ComputeRRM, DM_SIA_Eq_DV, SparseGtBuilder, SparseIndexGtBuilder}
import org.broadinstitute.hail.variant.VariantDataset
import org.broadinstitute.hail.utils._

object LinearMixedModel {
  def schema: Type = TStruct(
    ("beta", TDouble),
    ("sigmaG2", TDouble),
    ("chi2", TDouble),
    ("pval", TDouble))

  def apply(vds: VariantDataset,
    kernelFiltExprVA: String,
    pathVA: List[String],
    completeSamples: IndexedSeq[String],
    C: DenseMatrix[Double],
    y: DenseVector[Double],
    sparsityThreshold: Double,
    optDelta: Option[Double] = None,
    useML: Boolean = false,
    useBlockedMatrix: Boolean = false): VariantDataset = {

    val n = y.length

    val completeSamplesSet = completeSamples.toSet
    val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray

    info(s"lmmreg: Computing kernel for $n samples...")

    // FIXME: use vds.cache() here?

    val vdsKernel = vds.filterVariantsExpr(kernelFiltExprVA, keep = true).filterSamples((s, sa) => completeSamplesSet(s))

    val (kernel, m) =
      if (useBlockedMatrix)
        ComputeRRM.withBlocks(vdsKernel)
      else
        ComputeRRM.withoutBlocks(vdsKernel)

    assert(kernel.rows == n && kernel.cols == n)

    info(s"lmmreg: RRM computed using $m variants. Computing eigenvectors... ") // should use better Lapack method

    val eigK = eigSymD(kernel)
    val Ut = eigK.eigenvectors.t
    val S = eigK.eigenvalues //place S in global annotations?
    assert(S.length == n)

    info("lmmreg: Largest evals: " + ((n - 1) to math.max(0, n - 10) by -1).map(S(_).formatted("%.5f")).mkString(", "))
    info("lmmreg: Smallest evals: " + (0 until math.min(n, 10)).map(S(_).formatted("%.5f")).mkString(", "))
    info(s"lmmreg: Estimating delta using ${ if (useML) "ML" else "REML" }... ")

    val UtC = Ut * C
    val Uty = Ut * y

    val diagLMM = DiagLMM(UtC, Uty, S, optDelta, useML)

    val header = "rank\teval"
    val evalString = (0 until n).map(i => s"$i\t${ S(i) }").mkString("\n")
    log.info(s"\nEIGENVALUES\n$header\n$evalString\n\n")

    info(s"lmmreg: delta = ${ diagLMM.delta }. Computing LMM statistics for each variant...")

    val T = (Ut(::, *) :* diagLMM.sqrtInvD)
    val Qt = qr.reduced.justQ(diagLMM.TC).t
    val QtTy = Qt * diagLMM.Ty
    val TyQtTy = (diagLMM.Ty dot diagLMM.Ty) - (QtTy dot QtTy)

    val sc = vds.sparkContext
    val TBc = sc.broadcast(T)
    val sampleMaskBc = sc.broadcast(sampleMask)
    val scalerLMMBc = sc.broadcast(ScalerLMM(diagLMM.Ty, diagLMM.TyTy, Qt, QtTy, TyQtTy, diagLMM.logNullS2, useML))

    val (newVAS, inserter) = vds.insertVA(LinearMixedModel.schema, pathVA)

    vds.mapAnnotations { case (v, va, gs) =>
      val (xSparse, xMean) = {
        val sb = new SparseGtBuilder()
        gs.iterator.zipWithIndex.foreach { case (g, i) => if (sampleMaskBc.value(i)) sb.merge(g) }
        sb.toSparseGtVector(n)
      }

      // FIXME: handle None better
      val xOpt =
        if (xMean <= sparsityThreshold)
          xSparse
        else
          xSparse.map(_.toDenseVector)

      val lmmregStat = xOpt.map(x => scalerLMMBc.value.likelihoodRatioTest(TBc.value * x))

//      val sb = new SparseIndexGtBuilder()
//      gs.iterator.zipWithIndex.foreach { case (g, i) => if (sampleMaskBc.value(i)) sb.merge(g) }
//      val sia = sb.toGtIndexArrays(n)
//
//      val Tx =
//        if (sia.isNonConstant) {
//          if (sia.mean < sparsityThreshold)
//            Some(DM_SIA_Eq_DV(TBc.value, sia))
//          else
//            Some(TBc.value * sia.toGtDenseVector)
//        } else
//          None
//
//      val lmmregStat = Tx.map(scalerLMMBc.value.likelihoodRatioTest)

      inserter(va, lmmregStat)
    }.copy(vaSignature = newVAS)
  }
}

object DiagLMM {
  def apply(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], optDelta: Option[Double] = None, useML: Boolean = false): DiagLMM = {
    require(C.rows == y.length)

    val delta = optDelta.getOrElse(fitDelta(C, y, S, useML)._1)

    val n = y.length
    val sqrtInvD = sqrt(S + delta).map(1 / _)
    val TC = C(::, *) :* sqrtInvD
    val Ty = y :* sqrtInvD
    val TyTy = Ty dot Ty
    val TCTy = TC.t * Ty
    val TCTC = TC.t * TC
    val b = TCTC \ TCTy
    val s2 = (TyTy - (TCTy dot b)) / (if (useML) n else n - C.cols)

    DiagLMM(b, s2, math.log(s2), delta, sqrtInvD, TC, Ty, TyTy, useML)
  }

  def fitDelta(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], useML: Boolean): (Double, IndexedSeq[(Double, Double)]) = {

    val logmin = -10
    val logmax = 10
    val pointsPerUnit = 100 // number of points per unit of log space

    val grid = (logmin * pointsPerUnit to logmax * pointsPerUnit).map(_.toDouble / pointsPerUnit) // avoids rounding of (logmin to logmax by logres)

    val n = y.length
    val c = C.cols

    // up to constant shift and scale by 2
    def negLogLkhd(delta: Double, useML: Boolean): Double = {
      val D = S + delta
      val dy = y :/ D
      val ydy = y dot dy
      val Cdy = C.t * dy
      val CdC = C.t * (C(::, *) :/ D)
      val b = CdC \ Cdy
      val r = ydy - (Cdy dot b)

      if (useML)
        sum(breeze.numerics.log(D)) + n * math.log(r)
      else
        sum(breeze.numerics.log(D)) + (n - c) * math.log(r) + logdet(CdC)._2
    }

    val gridVals = grid.map(logDelta => (logDelta, negLogLkhd(math.exp(logDelta), useML)))

    val header = "logDelta\tnegLogLkhd"
    val gridValsString = gridVals.map{ case (d, nll) => s"${d.formatted("%.4f")}\t$nll" }.mkString("\n")
    log.info(s"\nDELTAVALUES\n$header\n$gridValsString\n\n")

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
  useML: Boolean)

case class ScalerLMM(
  y: DenseVector[Double],
  yy: Double,
  Qt: DenseMatrix[Double],
  Qty: DenseVector[Double],
  yQty: Double,
  logNullS2: Double,
  useML: Boolean) {

  def likelihoodRatioTest(x: Vector[Double]): Annotation = {

    val n = y.length
    val Qtx = Qt * x
    val xQtx: Double = (x dot x) - (Qtx dot Qtx)
    val xQty: Double = (x dot y) - (Qtx dot Qty)

    val b: Double = xQty / xQtx
    val s2 = (yQty - xQty * b) / (if (useML) n else n - Qt.rows)
    val chi2 = n * (logNullS2 - math.log(s2))
    val p = chiSquaredTail(1, chi2)

    Annotation(b, s2, chi2, p)
  }
}
