package org.broadinstitute.hail.stats

import breeze.linalg._
import org.apache.spark.mllib.linalg.{Matrices, DenseMatrix => SDenseMatrix}
import org.apache.spark.mllib.linalg.SingularValueDecomposition
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.annotations._
import org.broadinstitute.hail.expr.{TDouble, TStruct, Type}
import org.broadinstitute.hail.methods.{ToSparseIndexedRowMatrix, ToSparseRDD, ToStandardizedIndexedRowMatrix}
import org.broadinstitute.hail.variant.{Genotype, Variant, VariantDataset}
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

    LMM(Wt, G, C, y, optDelta)
  }

  def progressReport(msg: String) = {
    val prog = s"\nprogresss: $msg, ${formatTime(System.nanoTime())}\n"
    println(prog)
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

//    val blockWt = Wt.toBlockMatrix().cache()
//    val Kspark = blockWt.transpose.multiply(blockWt).toLocalMatrix().asInstanceOf[SDenseMatrix]
//    assert(Kspark.numCols == n && Kspark.numRows == n)

    val Kspark = Wt.computeGramianMatrix().asInstanceOf[SDenseMatrix]

    val K = new DenseMatrix[Double](n, n, Kspark.values)

    progressReport("Computing SVD... ") // should use better Lapack method

    val svdK = svd(K)
    val Ut = svdK.U.t
    val S = svdK.S //place S in global annotations?
    assert(S.length == n)

    progressReport("Largest evals: " + (0 until math.min(n, 10)).map(S(_).formatted("%.5f")).mkString(", "))
    progressReport("Smallest evals: " + ((n - 1) to math.max(0, n - 10) by -1).map(S(_).formatted("%.5f")).mkString(", "))

    progressReport("Estimating delta... ")

    val diagLMM = DiagLMM(Ut * C, Ut * y, S, optDelta, useREML)

    progressReport(s"delta = ${diagLMM.delta}, h2 = ${1d / diagLMM.delta}.")

    // temporary
    val header = "rank\teval"
    val evalString = (0 until n).map(i => s"$i\t${S(i)}").mkString("\n")
//    val bw = new java.io.BufferedWriter(new java.io.FileWriter(new java.io.File("evals.tsv")))
//    bw.write(s"$header\n$evalString\n")
//    bw.close()
    log.info(s"\nEIGENVALUES\n$header\n$evalString\n\n")

//    val diagLMMBc = genotypes.rows.sparkContext.broadcast(diagLMM)
//
//    val Uspark = new SDenseMatrix(n, n, U.data)
//
//    val lmmResult = genotypes.multiply(Uspark).rows.map(r =>
//      (variants(r.index.toInt), diagLMMBc.value.likelihoodRatioTest(toBDenseVector(r.vector.toDense))))

    progressReport(s"Computing lmm statistics for each variant...")

    val sc = G.sparkContext
    val diagLMMBc = sc.broadcast(diagLMM)
    val UtBc = sc.broadcast(Ut)

    val lmmResult = G.mapValues( x => diagLMMBc.value.likelihoodRatioTest(UtBc.value, x))

    println(formatTime(System.nanoTime()))

    LMMResult(diagLMM, lmmResult)
  }
}

object DiagLMM {
  def apply(C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], optDelta: Option[Double] = None, useREML: Boolean = true): DiagLMM = {
    require(C.rows == y.length)

    val delta = optDelta.getOrElse(fitDelta(C, y, S, useREML)._1)

    // FIXME: add warning or fatal when delta is not in interior

    val n = y.length
    val D = S + delta
    val dy = y :/ D
    val dC = C(::, *) :/ D
    val ydy = y dot dy
    val Cdy = C.t * dy
    val CdC = C.t * dC
    val b = CdC \ Cdy
    val s2 = (ydy - (Cdy dot b)) / (if (useREML) n - C.cols else n)

    DiagLMM(C, y, dy, ydy, b, s2, math.log(s2), delta, D.map(1 / _), useREML)
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
    val header = "logDelta\tnegLogLkhd"
    val gridValsString = gridVals.map{ case (d, nll) => s"${d.formatted("%.2f")}\t$nll" }.mkString("\n")
//    val bw = new java.io.BufferedWriter(new java.io.FileWriter(new java.io.File("delta.tsv")))
//    bw.write(s"$header\n$gridValsString\n")
//    bw.close()

    log.info(s"\nDELTAVALUES\n$header\n$gridValsString\n\n")

    val logDelta = gridVals.minBy(_._2)._1

    if (logDelta == logmin)
      fatal("failed to fit delta: maximum likelihood at lower search boundary 1e-10")
    else if (logDelta == logmax)
      fatal("failed to fit delta: maximum likelihood at upper search boundary 1e10")
    
    (math.exp(logDelta), gridVals)
  }
}

case class DiagLMM(
  C: DenseMatrix[Double],
  y: DenseVector[Double],
  dy: DenseVector[Double],
  ydy: Double,
  nullB: DenseVector[Double],
  nullS2: Double,
  logNullS2: Double,
  delta: Double,
  invD: DenseVector[Double],
  useREML: Boolean) {

  def likelihoodRatioTest(Ut: DenseMatrix[Double], x0: Vector[Double]): LMMStat = {
    require(x0.length == y.length)

    val n = x0.length
    val x = Ut * x0
    val X = DenseMatrix.horzcat(x.asDenseMatrix.t, C)

    val dX = X(::, *) :* invD
    val XdX = X.t * dX // can precompute CdC
    val Xdy = X.t * dy

    val b = XdX \ Xdy
    val s2 = (ydy - (Xdy dot b)) / (if (useREML) n - C.cols else n)
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

case class LMMStat(b: DenseVector[Double], s2: Double, chi2: Double, p: Double) {
  def toAnnotation: Annotation = Annotation(b(0), s2, chi2, p)
}

case class LMMResult(diagLMM: DiagLMM, rdd: RDD[(Variant, LMMStat)])

object LMMLowRank {
  def applyVds(vds: VariantDataset,
    filtAssoc: (Variant, Annotation, Iterable[Genotype]) => Boolean,
    filtGRM: (Variant, Annotation, Iterable[Genotype]) => Boolean,
    C: DenseMatrix[Double],
    y: DenseVector[Double],
    k: Int,
    optDelta: Option[Double] = None,
    useREML: Boolean = true): LMMResultLowRank = {

    val Wt = ToStandardizedIndexedRowMatrix(vds.filterVariants(filtGRM))._2 // W is samples by variants, Wt is variants by samples
    val (variants, genotypes) = ToSparseIndexedRowMatrix(vds.filterVariants(filtAssoc))

    LMMLowRank(Wt, variants, genotypes, C, y, k, optDelta, useREML)
  }

  def apply(Wt: IndexedRowMatrix,
    variants: Array[Variant],
    genotypes: IndexedRowMatrix,
    C: DenseMatrix[Double],
    y: DenseVector[Double],
    k: Int,
    optDelta: Option[Double] = None,
    useREML: Boolean = true): LMMResultLowRank = {

    require(k <= Wt.numRows())
    require(k <= Wt.numCols())

    val n = y.length

    val svd: SingularValueDecomposition[IndexedRowMatrix, org.apache.spark.mllib.linalg.Matrix] = Wt.computeSVD(k)

    val U = svd.V // W = svd.V * svd.s * svd.U.t
    val UB = new DenseMatrix[Double](svd.V.numRows, svd.V.numCols, svd.V.toArray)

    // println(UB.rows, UB.cols)

    val s = toBDenseVector(svd.s.toDense)
    val SB = s :* s // K = U * (svd.s * svd.s) * V.t, s has length k

    // SB.foreach(println)

    val Cp = UB.t * C
    val yp = UB.t * y
    val CpC = (C.t * C) - (Cp.t * Cp)
    val Cpy = (C.t * y) - (Cp.t * yp)
    val ypy = (y dot y) - (yp dot yp)

    val diagLMMLowRank = DiagLMMLowRank(n, Cp, yp, SB, ypy, Cpy, CpC, optDelta, useREML)

    val diagLMMLowRankBc = genotypes.rows.sparkContext.broadcast(diagLMMLowRank)

    // adding in y
    val U_y = Matrices.horzcat(Array(U, new SDenseMatrix(y.length, 1, y.toArray)))

    val lmmResultLowRank = genotypes.multiply(U_y).rows.map { r =>
      val result = toBDenseVector(r.vector.toDense)
      (variants(r.index.toInt), diagLMMLowRankBc.value.likelihoodRatioTest(result(0 until k), result(k)))
    }

    LMMResultLowRank(diagLMMLowRank, lmmResultLowRank)
  }
}

object DiagLMMLowRank {
  def apply(n: Int, C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], ypy: Double, Cpy: DenseVector[Double], CpC: DenseMatrix[Double], optDelta: Option[Double] = None, useREML: Boolean = true): DiagLMMLowRank = {
    require(C.rows == y.length)

    val delta = optDelta.getOrElse(deltaGridVals(n, C, y, S, ypy, Cpy, CpC, useREML).minBy(_._2)._1)

    val D = S + delta
    val dy = y :/ D
    val dC = C(::, *) :/ D
    val ydy = y dot dy
    val Cdy = C.t * dy
    val CdC = C.t * dC
    val b = (CdC + CpC / delta) \ (Cdy + Cpy / delta)
    val r1 = ydy - (Cdy dot b)
    val r2 = (ypy - (Cpy dot b)) / delta
    val s2 = (r1 + r2) / (if (useREML) n - C.cols else n)

    DiagLMMLowRank(n, C, y, dy, ydy, ypy, Cpy, CpC, b, s2, math.log(s2), delta, D.map(1 / _), useREML)
  }

  def deltaGridVals(n: Int, C: DenseMatrix[Double], y: DenseVector[Double], S: DenseVector[Double], ypy: Double, Cpy: DenseVector[Double], CpC: DenseMatrix[Double], useREML: Boolean): IndexedSeq[(Double, Double)] = {

    val logmin = -10.0
    val logmax = 10.0
    val logres = 0.1

    val grid = (logmin to logmax by logres).map(math.exp)

    // up to constant shift and scale by 2

    def negLogLkhd(delta: Double, useREML: Boolean): Double = {
      val k = y.length
      val c = C.cols
      val D = S + delta
      val dy = y :/ D
      val dC = C(::, *) :/ D
      val ydy = y dot dy
      val Cdy = C.t * dy
      val CdC = C.t * dC
      val b = (CdC + CpC / delta) \ (Cdy + Cpy / delta)
      val r1 = ydy - (Cdy dot b)
      val r2 = (ypy - (Cpy dot b)) / delta
      val r = r1 + r2

      if (useREML)
        sum(breeze.numerics.log(D)) + (n - k) * math.log(delta) + (n - c) * math.log(r) + logdet(CdC + CpC / delta)._2
      else
        sum(breeze.numerics.log(D)) + (n - k) * math.log(delta) + n * math.log(r)
    }

    grid.map(delta => (delta, negLogLkhd(delta, useREML)))
  }

}

case class DiagLMMLowRank(
  n: Int,
  C: DenseMatrix[Double],
  y: DenseVector[Double],
  dy: DenseVector[Double],
  ydy: Double,
  ypy: Double,
  Cpy: DenseVector[Double],
  CpC: DenseMatrix[Double],
  nullB: DenseVector[Double],
  nullS2: Double,
  logNullS2: Double,
  delta: Double,
  invD: DenseVector[Double],
  useREML: Boolean) {

  def likelihoodRatioTest(x: DenseVector[Double], xpy: Double): LMMStat = {
    require(x.length == y.length)

    val c = C.cols

    val X = DenseMatrix.horzcat(x.asDenseMatrix.t, C)

    val XdX = X.t * (X(::, *) :* invD) // can precompute CdC
    val Xdy = X.t * dy

    val xpx = x dot x
    val Cpx = C.t * x
    val Xpy = DenseVector.vertcat(DenseVector(xpy), Cpy)

    val XpX = DenseMatrix.zeros[Double](c + 1, c + 1)  // ugly, and while loop is faster
    XpX(0, 0) = xpx
    for (i <- 1 to c) {
      XpX(0, i) = Cpx(i - 1)
      XpX(i, 0) = Cpx(i - 1)
      for (j <- 0 to c) {
        XpX(i, j) = CpC(i - 1, j - 1)
      }
    }

    val b = (XdX + XpX / delta) \ (Xdy + Xpy / delta)

    val r1 = ydy - (Xdy dot b)
    val r2 = (ypy - (Xpy dot b)) / delta
    val s2 = (r1 + r2) / (if (useREML) n - c else n)

    val chi2 = n * (logNullS2 - math.log(s2))
    val p = chiSquaredTail(1, chi2)

    LMMStat(b, s2, chi2, p)
  }
}

case class LMMResultLowRank(diagLMMLowRank: DiagLMMLowRank, stats: RDD[(Variant, LMMStat)])