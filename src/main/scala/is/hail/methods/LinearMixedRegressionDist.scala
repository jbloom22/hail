package is.hail.methods

import breeze.linalg._
import is.hail.annotations._
import is.hail.distributedmatrix.DistributedMatrix
import is.hail.expr._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.spark.mllib.linalg.{DenseMatrix => SparkDenseMatrix}
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow}
import is.hail.distributedmatrix.DistributedMatrix.implicits._

object LinearMixedRegressionDist {
  val dm = DistributedMatrix[BlockMatrix]
  import dm.ops._

  def applyEigenDist(
    vds: VariantDataset,
    eigenDist: EigendecompositionDist,
    yExpr: String,
    covExpr: Array[String],
    useML: Boolean,
    rootGA: String,
    rootVA: String,
    runAssoc: Boolean,
    optDelta: Option[Double],
    sparsityThreshold: Double,
    useDosages: Boolean): VariantDataset = {
    
    require(vds.wasSplit)

    val pathVA = Parser.parseAnnotationRoot(rootVA, Annotation.VARIANT_HEAD)
    Parser.validateAnnotationRoot(rootGA, Annotation.GLOBAL_HEAD)

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
    val completeSamplesSet = completeSamples.toSet
    val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray
    val completeSampleIndex = (0 until vds.nSamples)
      .filter(i => completeSamplesSet(vds.sampleIds(i)))
      .toArray

    optDelta.foreach(delta =>
      if (delta <= 0d)
        fatal(s"delta must be positive, got ${ delta }"))

    val covNames = "intercept" +: covExpr

    val n = y.length
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"lmmreg: $n samples and $k ${plural(k, "covariate")} including intercept implies $d degrees of freedom.")

    info(s"lmmreg: running lmmreg on $n samples with $k sample ${plural(k, "covariate")} including intercept...")

    optDelta match {
      case Some(del) => info(s"lmmreg: Delta of $del specified by user")
      case None => info(s"lmmreg: Estimating delta using ${ if (useML) "ML" else "REML" }... ")
    }

    // FIXME: need filtering on eigenDist
    val EigendecompositionDist(_, rowIds, evects, evals) = eigenDist
    
    if (! completeSamples.sameElements(rowIds))
      fatal("Bad stuff")
    
    val Ut = evects.transpose
    val S = evals
    val nEigs = S.length

    info(s"lmmreg: Using $nEigs")
    info(s"lmmreg: Evals 1 to ${math.min(20, nEigs)}: " + ((nEigs - 1) to math.max(0, nEigs - 20) by -1).map(S(_).formatted("%.5f")).mkString(", "))
    info(s"lmmreg: Evals $nEigs to ${math.max(1, nEigs - 20)}: " + (0 until math.min(nEigs, 20)).map(S(_).formatted("%.5f")).mkString(", "))

    val lmmConstants = LMMConstantsDist(y, cov, S, Ut)

    val diagLMM = DiagLMM(lmmConstants, optDelta, useML)

    val delta = diagLMM.delta
    val globalBetaMap = covNames.zip(diagLMM.globalB.toArray).toMap
    val globalSg2 = diagLMM.globalS2
    val globalSe2 = delta * globalSg2
    val h2 = 1 / (1 + delta)

    val header = "rank\teval"
    val evalString = (0 until nEigs).map(i => s"$i\t${ S(nEigs - i - 1) }").mkString("\n")
    log.info(s"\nlmmreg: table of eigenvalues\n$header\n$evalString\n")

    info(s"lmmreg: global model fit: beta = $globalBetaMap")
    info(s"lmmreg: global model fit: sigmaG2 = $globalSg2")
    info(s"lmmreg: global model fit: sigmaE2 = $globalSe2")
    info(s"lmmreg: global model fit: delta = $delta")
    info(s"lmmreg: global model fit: h2 = $h2")
    
    diagLMM.optGlobalFit.foreach { gf => info(s"lmmreg: global model fit: seH2 = ${ gf.sigmaH2 }") }

    val vds1 = vds.annotateGlobal(
      Annotation(useML, globalBetaMap, globalSg2, globalSe2, delta, h2, nEigs),
      TStruct(("useML", TBoolean), ("beta", TDict(TString, TDouble)), ("sigmaG2", TDouble), ("sigmaE2", TDouble),
        ("delta", TDouble), ("h2", TDouble), ("nEigs", TInt)), rootGA)

    val vds2 = diagLMM.optGlobalFit match {
      case Some(gf) =>
        val (logDeltaGrid, logLkhdVals) = gf.gridLogLkhd.unzip
        vds1.annotateGlobal(
          Annotation(gf.sigmaH2, gf.h2NormLkhd, gf.maxLogLkhd, logDeltaGrid, logLkhdVals, nEigs),
          TStruct(("seH2", TDouble), ("normLkhdH2", TArray(TDouble)), ("maxLogLkhd", TDouble),
            ("logDeltaGrid", TArray(TDouble)), ("logLkhdVals", TArray(TDouble))), rootGA + ".fit")
      case None =>
        assert(optDelta.isDefined)
        vds1
    }
    

    if (runAssoc) {

      val (newVAS, inserter) = vds2.insertVA(LinearMixedRegression.schema, pathVA)

      info(s"lmmreg: Computing statistics for each variant...")

      val useFullRank = nEigs == n

      val (scalarLMM, projection) = if (useFullRank) {
        val Qt = qr.reduced.justQ(diagLMM.TC).t
        val QtTy = Qt * diagLMM.Ty
        val TyQtTy = (diagLMM.Ty dot diagLMM.Ty) - (QtTy dot QtTy)
        (new FullRankScalarLMM(diagLMM.Ty, diagLMM.TyTy, Qt, QtTy, TyQtTy, diagLMM.logNullS2, useML),
          Ut :* diagLMM.sqrtInvD.toArray)
      } else
        (new LowRankScalarLMM(lmmConstants, delta, diagLMM.logNullS2, useML), Ut)

      val (variants, allG) = ToIndexedRowMatrix(vds, useDosages, sampleMask, completeSampleIndex)
      allG.rows.cache()
      val projG = (allG.toBlockMatrixDense() * projection.transpose).toIndexedRowMatrix()
      
      val sc = vds.sparkContext
      val variantsBc = sc.broadcast(variants)  
      
      val G0 = allG.rows.map { case IndexedRow(i, x) => (variantsBc.value(i.toInt), DenseVector(x.toArray)) }
      val projG0 = projG.rows.map { case IndexedRow(i, px) => (variantsBc.value(i.toInt), DenseVector(px.toArray)) }
      val rddG = G0.join(projG0).orderedRepartitionBy(vds.rdd.orderedPartitioner)
      val rdd = vds.rdd.orderedLeftJoinDistinct(rddG).asOrderedRDD
      
      val scalarLMMBc = sc.broadcast(scalarLMM)

      val blockSize = 128
      val newRDD = rdd.mapPartitions({it =>
        val missingSamples = new ArrayBuilder[Int]

        // columns are genotype vectors
        var X: DenseMatrix[Double] = null
        var projX: DenseMatrix[Double] = null
        
        it.grouped(blockSize)
          .flatMap ( git => {
            val block = git.toArray
            val blockLength = block.length

            if (X == null || X.cols != blockLength) {
              X = new DenseMatrix[Double](n, blockLength)
              projX = new DenseMatrix[Double](nEigs, blockLength)
            }
            
            var i = 0
            while (i < blockLength) {
              val (_, ((_, _), Some((x, px)))) = block(i)
                X(::, i) := x
                projX(::, i) := px
              i += 1
            }
            
            val annotations = scalarLMMBc.value match {
              case sclr: FullRankScalarLMM =>
                sclr.likelihoodRatioTestBlock(X)
              case sclr: LowRankScalarLMM =>
                sclr.likelihoodRatioTestBlockLowRank(X, projX)
            }

            (block, annotations).zipped.map { case ((v, ((va, gs), _)), a) => (v, (inserter(va, a), gs)) }
          } )
      }, preservesPartitioning = true)

      vds2.copy(
        rdd = newRDD.asOrderedRDD,
        vaSignature = newVAS)
    } else
      vds2
  }
}

object LMMConstantsDist {
  def apply(y: DenseVector[Double], C: DenseMatrix[Double],
            S: DenseVector[Double], Ut: BlockMatrix): LMMConstants = {
    
    val UtC = Helper.multiply(Ut, C)
    val Uty = Helper.multiply(Ut, y)
        
    val CtC = C.t * C
    val Cty = C.t * y
    val yty = y.t * y

    val n = y.length
    val d = C.cols

    LMMConstants(y, C, S, Uty, UtC, Cty, CtC, yty, n, d)
  }
}

object Helper {
  val dm = DistributedMatrix[BlockMatrix]
  import dm.ops._
  
  def multiply(bm: BlockMatrix, v: DenseVector[Double]): DenseVector[Double] =
    DenseVector((bm * v.asDenseMatrix.t.asSpark()).toLocalMatrix().asInstanceOf[SparkDenseMatrix].values)
  
  def multiply(bm: BlockMatrix, m: DenseMatrix[Double]): DenseMatrix[Double] =
    (bm * m.asSpark()).toLocalMatrix().asBreeze().asInstanceOf[DenseMatrix[Double]]
}