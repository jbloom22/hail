package is.hail.methods

import breeze.linalg._
import breeze.numerics.{sigmoid, sqrt}
import is.hail.annotations._
import is.hail.expr._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.commons.math3.analysis.UnivariateFunction
import org.apache.commons.math3.optim.MaxEval
import org.apache.commons.math3.optim.nonlinear.scalar.GoalType
import org.apache.commons.math3.optim.univariate.{BrentOptimizer, SearchInterval, UnivariateObjectiveFunction}
import org.apache.commons.math3.util.FastMath
import is.hail.distributedmatrix.DistributedMatrix
import is.hail.distributedmatrix.DistributedMatrix.implicits._
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow}
import org.apache.spark.mllib.linalg.{DenseMatrix => SparkDenseMatrix}
import is.hail.sparkextras.{OrderedKey, OrderedPartitioner, OrderedRDD}

import scala.reflect.ClassTag

object LinearMixedRegression {
  val schema: Type = TStruct(
    ("beta", TDouble),
    ("sigmaG2", TDouble),
    ("chi2", TDouble),
    ("pval", TDouble))

  def apply(
    vds: VariantDataset,
    kinshipMatrix: KinshipMatrix,
    yExpr: String,
    covExpr: Array[String],
    useML: Boolean,
    rootGA: String,
    rootVA: String,
    runAssoc: Boolean,
    optDelta: Option[Double],
    useDosages: Boolean,
    optNEigs: Option[Int],
    blockSize: Int): VariantDataset = {

    val nEigs = optNEigs.getOrElse(kinshipMatrix.sampleIds.length)
    val eigen = kinshipMatrix.eigen().takeTop(nEigs)
    
    applyEigen(vds, eigen, yExpr, covExpr, useML, rootGA, rootVA, runAssoc, optDelta, useDosages, blockSize)
  }

  def applyEigen(
    vds: VariantDataset,
    eigen: Eigen,
    yExpr: String,
    covExpr: Array[String],
    useML: Boolean,
    rootGA: String,
    rootVA: String,
    runAssoc: Boolean,
    optDelta: Option[Double],
    useDosages: Boolean,
    blockSize: Int): VariantDataset = {

    require(vds.wasSplit)

    val pathVA = Parser.parseAnnotationRoot(rootVA, Annotation.VARIANT_HEAD)
    Parser.validateAnnotationRoot(rootGA, Annotation.GLOBAL_HEAD)

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
    val C = cov
    val completeSamplesSet = completeSamples.toSet

    optDelta.foreach(delta =>
      if (delta <= 0d)
        fatal(s"delta must be positive, got ${ delta }"))

    val n = y.length
    val c = C.cols
    val d = n - c - 1

    if (d < 1)
      fatal(s"lmmreg: $n samples and $c ${ plural(c, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"lmmreg: Running lmmreg on $n samples with $c sample ${ plural(c, "covariate") } including intercept...")

    optDelta match {
      case Some(del) => info(s"lmmreg: Delta of $del specified by user")
      case None => info(s"lmmreg: Estimating delta using ${ if (useML) "ML" else "REML" }... ")
    }

    val Eigen(_, rowIds, evects, evals) = eigen.filterRows(vds.sSignature, completeSamplesSet)
    
    if (!completeSamples.sameElements(rowIds))
      fatal("Complete samples in the dataset must all be coordinates of the eigenvectors, and in the same order.")

    val nFiltered = rowIds.length - completeSamples.length
    if (nFiltered > 0)
      info(s"lmmreg: Filtered $nFiltered coordinates from each eigenvector, as the corresponding samples were not complete samples in the dataset.")

    val Ut = evects.t
    val S = evals
    val nEigs = S.length

    info(s"lmmreg: Using $nEigs")
    info(s"lmmreg: Evals 1 to ${ math.min(20, nEigs) }: " + ((nEigs - 1) to math.max(0, nEigs - 20) by -1).map(S(_).formatted("%.5f")).mkString(", "))
    info(s"lmmreg: Evals $nEigs to ${ math.max(1, nEigs - 20) }: " + (0 until math.min(nEigs, 20)).map(S(_).formatted("%.5f")).mkString(", "))

    val UtC = Ut * C
    val Uty = Ut * y
    val CtC = C.t * C
    val Cty = C.t * y
    val yty = y.t * y

    val lmmConstants = LMMConstants(y, C, S, Uty, UtC, Cty, CtC, yty, n, c)

    val diagLMM = DiagLMM(lmmConstants, optDelta, useML)

    val vds1 = globalFit(vds, diagLMM, covExpr, nEigs, S, rootGA, useML)

    if (runAssoc) {
      val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray
      val completeSampleIndex = (0 until vds.nSamples).filter(sampleMask).toArray

      val sc = vds1.sparkContext
      val sampleMaskBc = sc.broadcast(sampleMask)
      val completeSampleIndexBc = sc.broadcast(completeSampleIndex)

      val (newVAS, inserter) = vds1.insertVA(LinearMixedRegression.schema, pathVA)

      info(s"lmmreg: Computing statistics for each variant...")

      val blockSize = 128
      val useFullRank = nEigs == n

      val newRDD =
        if (useFullRank) {
          val Qt = qr.reduced.justQ(diagLMM.TC).t
          val QtTy = Qt * diagLMM.Ty
          val TyQtTy = (diagLMM.Ty dot diagLMM.Ty) - (QtTy dot QtTy)
          val scalarLMM = new FullRankScalarLMM(diagLMM.Ty, diagLMM.TyTy, Qt, QtTy, TyQtTy, diagLMM.logNullS2, useML)
          val projection = Ut(::, *) :* diagLMM.sqrtInvD

          val scalarLMMBc = sc.broadcast(scalarLMM)
          val projectionBc = sc.broadcast(projection)

          vds1.rdd.mapPartitions({ it =>
            val missingSamples = new ArrayBuilder[Int]

            // columns are genotype vectors
            var X: DenseMatrix[Double] = null

            it.grouped(blockSize)
              .flatMap(git => {
                val block = git.toArray
                val blockLength = block.length

                if (X == null || X.cols != blockLength)
                  X = new DenseMatrix[Double](n, blockLength)

                var i = 0
                while (i < blockLength) {
                  val (_, (_, gs)) = block(i)

                  if (useDosages)
                    RegressionUtils.dosages(X(::, i), gs, completeSampleIndexBc.value, missingSamples)
                  else
                    X(::, i) := RegressionUtils.hardCalls(gs, n, sampleMaskBc.value) // No special treatment of constant

                  i += 1
                }

                val annotations = scalarLMMBc.value.likelihoodRatioTestBlock(projectionBc.value * X)

                (block, annotations).zipped.map { case ((v, (va, gs)), a) => (v, (inserter(va, a), gs)) }
              })
          }, preservesPartitioning = true)
        } else {
          val scalarLMM = LowRankScalarLMM(lmmConstants, diagLMM.delta, diagLMM.logNullS2, useML)
          val scalarLMMBc = sc.broadcast(scalarLMM)
          val UtBc = sc.broadcast(Ut)

          vds1.rdd.mapPartitions({ it =>
            val sclr = scalarLMMBc.value

            val r1 = 1 to c

            val CtC = DenseMatrix.zeros[Double](c + 1, c + 1)
            CtC(r1, r1) := sclr.con.CtC

            val UtC = DenseMatrix.zeros[Double](nEigs, c + 1)
            UtC(::, r1) := sclr.con.UtC

            val Cty = DenseVector.zeros[Double](c + 1)
            Cty(r1) := sclr.con.Cty

            val CzC = DenseMatrix.zeros[Double](c + 1, c + 1)
            CzC(r1, r1) := sclr.UtcovZUtcov

            val missingSamples = new ArrayBuilder[Int]
            val x0 = DenseVector.zeros[Double](n)

            it.map { case (v, (va, gs)) =>
              val x: Vector[Double] =
                if (useDosages) {
                  RegressionUtils.dosages(x0, gs, completeSampleIndexBc.value, missingSamples)
                  x0
                } else
                  RegressionUtils.hardCalls(gs, n, sampleMaskBc.value) // No special treatment of constant
              
              val annotation = sclr.likelihoodRatioTestLowRank(x, UtBc.value * x, CtC, UtC, Cty, CzC)

              (v, (inserter(va, annotation), gs)) }
          }, preservesPartitioning = true)
        }

      vds1.copy(
        rdd = newRDD.asOrderedRDD,
        vaSignature = newVAS)
    } else
      vds1
  }

  def globalFit(vds: VariantDataset, diagLMM: DiagLMM, covExpr: Array[String], nEigs: Int,
    S: DenseVector[Double], rootGA: String, useML: Boolean): VariantDataset = {

    val delta = diagLMM.delta
    val covNames = "intercept" +: covExpr
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

    diagLMM.optGlobalFit match {
      case Some(gf) =>
        val (logDeltaGrid, logLkhdVals) = gf.gridLogLkhd.unzip
        vds1.annotateGlobal(
          Annotation(gf.sigmaH2, gf.h2NormLkhd, gf.maxLogLkhd, logDeltaGrid, logLkhdVals),
          TStruct(("seH2", TDouble), ("normLkhdH2", TArray(TDouble)), ("maxLogLkhd", TDouble),
            ("logDeltaGrid", TArray(TDouble)), ("logLkhdVals", TArray(TDouble))), rootGA + ".fit")
      case None => vds1
    }
  }
  
  val dm = DistributedMatrix[BlockMatrix]
  import dm.ops._

  def multiply(bm: BlockMatrix, v: DenseVector[Double]): DenseVector[Double] =
    DenseVector((bm * v.asDenseMatrix.t.asSpark()).toLocalMatrix().asInstanceOf[SparkDenseMatrix].values)

  def multiply(bm: BlockMatrix, m: DenseMatrix[Double]): DenseMatrix[Double] =
    (bm * m.asSpark()).toLocalMatrix().asBreeze().asInstanceOf[DenseMatrix[Double]]
  
  def writeProjection(path: String, vds: VariantDataset, eigenDist: EigenDistributed, yExpr: String, covExpr: Array[String], useDosages: Boolean) {
    val (_, _, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
    val completeSamplesSet = completeSamples.toSet
    val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray
    val completeSampleIndex = (0 until vds.nSamples).filter(sampleMask).toArray

    val EigenDistributed(_, rowIds, evects, evals) = eigenDist

    if (!completeSamples.sameElements(rowIds))
      fatal("Complete samples in the dataset must coincide with rows IDs of eigenvectors, in the same order.")
    
    val G = ToIndexedRowMatrix(vds, useDosages, sampleMask, completeSampleIndex)
    val projG = G.toBlockMatrixDense() * evects
    
    dm.write(projG, path)
  }
  
  def applyEigenDistributed(
    vds: VariantDataset,
    eigenDist: EigenDistributed,
    yExpr: String,
    covExpr: Array[String],
    useML: Boolean,
    rootGA: String,
    rootVA: String,
    runAssoc: Boolean,
    optDelta: Option[Double],
    useDosages: Boolean,
    pathToProjection: Option[String],
    blockSize: Int): VariantDataset = {

    require(vds.wasSplit)

    val pathVA = Parser.parseAnnotationRoot(rootVA, Annotation.VARIANT_HEAD)
    Parser.validateAnnotationRoot(rootGA, Annotation.GLOBAL_HEAD)

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
    val C = cov
    val completeSamplesSet = completeSamples.toSet

    optDelta.foreach(delta =>
      if (delta <= 0d)
        fatal(s"delta must be positive, got ${ delta }"))

    val covNames = "intercept" +: covExpr

    val n = y.length
    val c = C.cols
    val d = n - c - 1

    if (d < 1)
      fatal(s"lmmreg: $n samples and $c ${ plural(c, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"lmmreg: Running lmmreg on $n samples with $c sample ${ plural(c, "covariate") } including intercept...")

    optDelta match {
      case Some(del) => info(s"lmmreg: Delta of $del specified by user")
      case None => info(s"lmmreg: Estimating delta using ${ if (useML) "ML" else "REML" }... ")
    }

    val EigenDistributed(_, rowIds, evects, evals) = eigenDist

    if (!completeSamples.sameElements(rowIds))
      fatal("Complete samples in the dataset must coincide with rows IDs of eigenvectors, in the same order.")
    
    val Ut = evects.t
    val S = evals
    val nEigs = S.length

    info(s"lmmreg: Using $nEigs")
    info(s"lmmreg: Evals 1 to ${ math.min(20, nEigs) }: " + ((nEigs - 1) to math.max(0, nEigs - 20) by -1).map(S(_).formatted("%.5f")).mkString(", "))
    info(s"lmmreg: Evals $nEigs to ${ math.max(1, nEigs - 20) }: " + (0 until math.min(nEigs, 20)).map(S(_).formatted("%.5f")).mkString(", "))

    val UtC = multiply(Ut, C)
    val Uty = multiply(Ut, y)
    val CtC = C.t * C
    val Cty = C.t * y
    val yty = y.t * y

    val lmmConstants = LMMConstants(y, C, S, Uty, UtC, Cty, CtC, yty, n, c)

    val diagLMM = DiagLMM(lmmConstants, optDelta, useML)

    val vds1 = LinearMixedRegression.globalFit(vds, diagLMM, covExpr, nEigs, S, rootGA, useML)

    val vds2 = if (runAssoc) {
      val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray // FIXME
      val completeSampleIndex = (0 until vds.nSamples).filter(sampleMask).toArray
      
      val sc = vds1.sparkContext
      val sampleMaskBc = sc.broadcast(sampleMask)
      val completeSampleIndexBc = sc.broadcast(completeSampleIndex)

      val (newVAS, inserter) = vds1.insertVA(LinearMixedRegression.schema, pathVA)

      info(s"lmmreg: Computing statistics for each variant...")

      vds1.persist()
      
      def makeIntOrderedPartitioner(vds: VariantDataset): OrderedPartitioner[Int, Int] = {
        val partitionSizes = vds.rdd.mapPartitions( it => Iterator(it.length), preservesPartitioning = true).collect()
        val boundsPlus2 = partitionSizes.scanLeft(-1)(_ + _)
        val bounds = boundsPlus2.slice(1, boundsPlus2.length - 1)
        
        val intOrderedKey = new OrderedKey[Int, Int] {
          def project(key: Int): Int = key
          def kOrd: Ordering[Int] = Ordering.Int
          def pkOrd: Ordering[Int] = Ordering.Int
          def kct: ClassTag[Int] = implicitly[ClassTag[Int]]
          override def pkct: ClassTag[Int] = implicitly[ClassTag[Int]]
        }
        
        new OrderedPartitioner[Int, Int](bounds, partitionSizes.length)(intOrderedKey)     
      }
      
      val intOrderedPartitioner = makeIntOrderedPartitioner(vds1)
      val variantOrderedPartitioner = vds1.rdd.orderedPartitioner
      
      val useFullRank = nEigs == n
      val variants = vds1.variants.collect()
      val variantsBc = sc.broadcast(variants) // can we avoid by zipping with index?
      
      val newRDD =
        if (useFullRank) {
          val Qt = qr.reduced.justQ(diagLMM.TC).t
          val QtTy = Qt * diagLMM.Ty
          val TyQtTy = (diagLMM.Ty dot diagLMM.Ty) - (QtTy dot QtTy)
          val scalarLMM = new FullRankScalarLMM(diagLMM.Ty, diagLMM.TyTy, Qt, QtTy, TyQtTy, diagLMM.logNullS2, useML)
          val scalarLMMBc = sc.broadcast(scalarLMM)

          val G = ToIndexedRowMatrix(vds1, useDosages, sampleMask, completeSampleIndex)
          
          val projG = (G.toBlockMatrixDense() * (Ut :* diagLMM.sqrtInvD.toArray).t)
            .toIndexedRowMatrixOrderedPartitioner(intOrderedPartitioner)
            .rows
            .map { case IndexedRow(i, px) => (variantsBc.value(i.toInt), DenseVector(px.toArray)) }

          val projG2 = OrderedRDD(projG, variantOrderedPartitioner)

          vds1.rdd.orderedLeftJoinDistinct(projG2).asOrderedRDD.mapPartitions({ it =>
            val missingSamples = new ArrayBuilder[Int]

            // columns are projected genotype vectors
            var projX: DenseMatrix[Double] = null

            it.grouped(blockSize)
              .flatMap(git => {
                val block = git.toArray
                val blockLength = block.length

                if (projX == null || projX.cols != blockLength)
                  projX = new DenseMatrix[Double](nEigs, blockLength)

                var i = 0
                while (i < blockLength) {
                  val (_, ((_, _), Some(px))) = block(i)

                  projX(::, i) := px

                  i += 1
                }

                val annotations = scalarLMMBc.value.likelihoodRatioTestBlock(projX)

                (block, annotations).zipped.map { case ((v, ((va, gs), _)), a) => (v, (inserter(va, a), gs)) }
              })
          }, preservesPartitioning = true)
        } else {
          val scalarLMM = LowRankScalarLMM(lmmConstants, diagLMM.delta, diagLMM.logNullS2, useML)
          val scalarLMMBc = sc.broadcast(scalarLMM)

          val projG = pathToProjection match {
            case Some(path) => dm.read(vds1.hc, path)
            case None =>
              val G = ToIndexedRowMatrix(vds1, useDosages, sampleMask, completeSampleIndex)
              G.toBlockMatrixDense() * Ut.t
          }
          
          if (projG.numRows() != variants.length)
            fatal(s"Dimension mismatch: projection matches ${projG.numRows()} variants, but there are ${variants.length} variants.")
          if (projG.numCols() != eigenDist.nEvects)
            fatal(s"Dimension mismatch: projection matches ${projG.numCols()} eigenvectors, but there are ${eigenDist.nEvects} eigenvectors.")
           
          val projG1 = projG
            .toIndexedRowMatrixOrderedPartitioner(intOrderedPartitioner)
            .rows
            .map { case IndexedRow(i, px) => (variantsBc.value(i.toInt), DenseVector(px.toArray)) }

          val projG2 = OrderedRDD(projG1, variantOrderedPartitioner)
          
          vds1.rdd.orderedLeftJoinDistinct(projG2).asOrderedRDD.mapPartitions({ it =>
            val sclr = scalarLMMBc.value

            val r2 = 1 to c

            val CtC = DenseMatrix.zeros[Double](c + 1, c + 1)
            CtC(r2, r2) := sclr.con.CtC

            val UtC = DenseMatrix.zeros[Double](nEigs, c + 1)
            UtC(::, r2) := sclr.Utcov

            val Cty = DenseVector.zeros[Double](c + 1)
            Cty(r2) := sclr.con.Cty

            val CzC = DenseMatrix.zeros[Double](c + 1, c + 1)
            CzC(r2, r2) := sclr.UtcovZUtcov

            val missingSamples = new ArrayBuilder[Int]
            val x = DenseVector.zeros[Double](n)

            it.map { case (v, ((va, gs), Some(px))) =>
              if (useDosages)
                RegressionUtils.dosages(x, gs, completeSampleIndexBc.value, missingSamples)
              else
                x := RegressionUtils.hardCalls(gs, n, sampleMaskBc.value) // No special treatment of constant

              val annotation = sclr.likelihoodRatioTestLowRank(x, px, CtC, UtC, Cty, CzC)

              (v, (inserter(va, annotation), gs))
            }
          }, preservesPartitioning = true)
        }

      vds1.unpersist()

      vds1.copy(
        rdd = newRDD.asOrderedRDD,
        vaSignature = newVAS)
    } else
      vds1

    vds2
  }
}

object DiagLMM {
  def apply(
    lmmConstants: LMMConstants,
    optDelta: Option[Double] = None,
    useML: Boolean = false): DiagLMM = {

    val UtC = lmmConstants.UtC
    val Uty = lmmConstants.Uty

    val CtC = lmmConstants.CtC
    val Cty = lmmConstants.Cty
    val yty = lmmConstants.yty
    val S = lmmConstants.S

    val n = lmmConstants.n
    val c = lmmConstants.c

    def fitDelta(): (Double, GlobalFitLMM) = {

      object LogLkhdML extends UnivariateFunction {
        val shift = -0.5 * n * (1 + math.log(2 * math.Pi))

        def value(logDelta: Double): Double = {
          val delta = FastMath.exp(logDelta)
          val invDelta = 1 / delta
          val D = S + delta
          val dy = Uty :/ D
          val Z = D.map(1 / _ - invDelta)

          val ydy = invDelta * yty + (Uty dot (Uty :* Z))
          val Cdy = invDelta * Cty + (UtC.t * (Uty :* Z))
          val CdC = invDelta * CtC + (UtC.t * (UtC(::, *) :* Z))

          val b = CdC \ Cdy
          val sigma2 = (ydy - (Cdy dot b)) / n

          val logdetD = sum(breeze.numerics.log(D)) + (n - S.length) * logDelta

          -0.5 * (logdetD + n * math.log(sigma2)) + shift
        }
      }

      object LogLkhdREML extends UnivariateFunction {
        val shift = -0.5 * (n - c) * (1 + math.log(2 * math.Pi))

        def value(logDelta: Double): Double = {
          val delta = FastMath.exp(logDelta)
          val invDelta = 1 / delta
          val D = S + delta
          val dy = Uty :/ D
          val Z = D.map(1 / _ - invDelta)

          val ydy = invDelta * yty + (Uty dot (Uty :* Z))
          val Cdy = invDelta * Cty + (UtC.t * (Uty :* Z))
          val CdC = invDelta * CtC + (UtC.t * (UtC(::, *) :* Z))

          val b = CdC \ Cdy
          val sigma2 = (ydy - (Cdy dot b)) / (n - c)

          val logdetD = sum(breeze.numerics.log(D)) + (n - S.length) * logDelta
          val (_, logdetCdC) = logdet(CdC)
          val (_, logdetCtC) = logdet(CtC)

          -0.5 * (logdetD + logdetCdC - logdetCtC + (n - c) * math.log(sigma2)) + shift
        }
      }

      // number of points per unit of log space
      val pointsPerUnit = 100
      val minLogDelta = -8
      val maxLogDelta = 8

      // avoids rounding of (minLogDelta to logMax by logres)
      val grid = (minLogDelta * pointsPerUnit to maxLogDelta * pointsPerUnit).map(_.toDouble / pointsPerUnit)
      val logLkhdFunction = if (useML) LogLkhdML else LogLkhdREML

      val gridLogLkhd = grid.map(logDelta => (logDelta, logLkhdFunction.value(logDelta)))

      val header = "logDelta\tlogLkhd"
      val gridValsString = gridLogLkhd.map { case (d, nll) => s"$d\t$nll" }.mkString("\n")
      log.info(s"\nlmmreg: table of delta\n$header\n$gridValsString\n")

      val (approxLogDelta, _) = gridLogLkhd.maxBy(_._2)

      if (approxLogDelta == minLogDelta)
        fatal(s"lmmreg: failed to fit delta: ${ if (useML) "ML" else "REML" } realized at delta lower search boundary e^$minLogDelta = ${ FastMath.exp(minLogDelta) }, indicating negligible enviromental component of variance. The model is likely ill-specified.")
      else if (approxLogDelta == maxLogDelta)
        fatal(s"lmmreg: failed to fit delta: ${ if (useML) "ML" else "REML" } realized at delta upper search boundary e^$maxLogDelta = ${ FastMath.exp(maxLogDelta) }, indicating negligible genetic component of variance. Standard linear regression may be more appropriate.")

      val searchInterval = new SearchInterval(minLogDelta, maxLogDelta, approxLogDelta)
      val goal = GoalType.MAXIMIZE
      val objectiveFunction = new UnivariateObjectiveFunction(logLkhdFunction)
      // tol = 5e-8 * abs((ln(delta))) + 5e-7 <= 1e-6
      val brentOptimizer = new BrentOptimizer(5e-8, 5e-7)
      val logDeltaPointValuePair = brentOptimizer.optimize(objectiveFunction, goal, searchInterval, MaxEval.unlimited)

      val maxlogDelta = logDeltaPointValuePair.getPoint
      val maxLogLkhd = logDeltaPointValuePair.getValue

      if (math.abs(maxlogDelta - approxLogDelta) > 1d / pointsPerUnit) {
        warn(s"lmmreg: the difference between the optimal value $approxLogDelta of ln(delta) on the grid and" +
          s"the optimal value $maxlogDelta of ln(delta) by Brent's method exceeds the grid resolution" +
          s"of ${ 1d / pointsPerUnit }. Plot the values over the full grid to investigate.")
      }

      val epsilon = 1d / pointsPerUnit

      // three values of ln(delta) right of, at, and left of the MLE
      val z1 = maxlogDelta + epsilon
      val z2 = maxlogDelta
      val z3 = maxlogDelta - epsilon

      // three values of h2 = sigmoid(-ln(delta)) left of, at, and right of the MLE
      val x1 = sigmoid(-z1)
      val x2 = sigmoid(-z2)
      val x3 = sigmoid(-z3)

      // corresponding values of logLkhd
      val y1 = logLkhdFunction.value(z1)
      val y2 = maxLogLkhd
      val y3 = logLkhdFunction.value(z3)

      if (y1 >= y2 || y3 >= y2)
        fatal(s"Maximum likelihood estimate ${ math.exp(maxlogDelta) } for delta is not a global max. " +
          s"Plot the values over the full grid to investigate.")

      // fitting parabola logLkhd ~ a * x^2 + b * x + c near MLE by Lagrange interpolation gives
      // a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / ((x2 - x1) * (x1 - x3) * (x3 - x2))
      // comparing to normal approx: logLkhd ~ 1 / (-2 * sigma^2) * x^2 + lower order terms:
      val sigmaH2 =
      math.sqrt(((x2 - x1) * (x1 - x3) * (x3 - x2)) / (-2 * (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2))))

      val h2LogLkhd = (0.01 to 0.99 by 0.01).map(h2 => logLkhdFunction.value(math.log((1 - h2) / h2)))

      val h2Lkhd = h2LogLkhd.map(ll => math.exp(ll - maxLogLkhd))
      val h2LkhdSum = h2Lkhd.sum
      val h2NormLkhd = IndexedSeq(Double.NaN) ++ h2Lkhd.map(_ / h2LkhdSum) ++ IndexedSeq(Double.NaN)

      (FastMath.exp(maxlogDelta), GlobalFitLMM(maxLogLkhd, gridLogLkhd, sigmaH2, h2NormLkhd))
    }

    def fitUsingDelta(delta: Double, optGlobalFit: Option[GlobalFitLMM]): DiagLMM = {
      val invDelta = 1 / delta
      val invD = (S + delta).map(1 / _)
      val dy = Uty :* invD

      val Z = invD - invDelta

      val ydy = invDelta * yty + (Uty dot (Uty :* Z))
      val Cdy = invDelta * Cty + (UtC.t * (Uty :* Z))
      val CdC = invDelta * CtC + (UtC.t * (UtC(::, *) :* Z))

      val b = CdC \ Cdy
      val s2 = (ydy - (Cdy dot b)) / (if (useML) n else n - c)
      val sqrtInvD = sqrt(invD)
      val TC = UtC(::, *) :* sqrtInvD
      val Ty = Uty :* sqrtInvD
      val TyTy = Ty dot Ty

      DiagLMM(b, s2, math.log(s2), delta, optGlobalFit, sqrtInvD, TC, Ty, TyTy, useML)
    }

    val (delta, optGlobalFit) = optDelta match {
      case Some(delta0) => (delta0, None)
      case None =>
        info("lmmreg: Fitting delta...")
        val (delta0, gf) = printTime(fitDelta())
        (delta0, Some(gf))
    }

    fitUsingDelta(delta, optGlobalFit)
  }
}

case class LMMConstants(y: DenseVector[Double], C: DenseMatrix[Double], S: DenseVector[Double],
  Uty: DenseVector[Double], UtC: DenseMatrix[Double], Cty: DenseVector[Double],
  CtC: DenseMatrix[Double], yty: Double, n: Int, c: Int)

case class GlobalFitLMM(maxLogLkhd: Double, gridLogLkhd: IndexedSeq[(Double, Double)], sigmaH2: Double, h2NormLkhd: IndexedSeq[Double])

case class DiagLMM(
  globalB: DenseVector[Double],
  globalS2: Double,
  logNullS2: Double,
  delta: Double,
  optGlobalFit: Option[GlobalFitLMM],
  sqrtInvD: DenseVector[Double],
  TC: DenseMatrix[Double],
  Ty: DenseVector[Double],
  TyTy: Double,
  useML: Boolean)

// Handles full-rank case
class FullRankScalarLMM(
  y: DenseVector[Double],
  yy: Double,
  Qt: DenseMatrix[Double],
  Qty: DenseVector[Double],
  yQty: Double,
  logNullS2: Double,
  useML: Boolean) {

  val n = y.length
  val invDf = 1.0 / (if (useML) n else n - Qt.rows)

  def likelihoodRatioTestBlock(X: DenseMatrix[Double]): Array[Annotation] = {
    val n = y.length.toDouble
    val QtX = Qt * X
    val XQtX = X.t(*, ::).map(r => r dot r) - QtX.t(*, ::).map(r => r dot r)
    val XQty = X.t * y - QtX.t * Qty

    val b = XQty :/ XQtX
    val s2 = invDf * (yQty - (XQty :* b))
    val chi2 = n * (logNullS2 - breeze.numerics.log(s2))
    val p = chi2.map(c => chiSquaredTail(1, c))
    
    Array.tabulate(X.cols)(i => Annotation(b(i), s2(i), chi2(i), p(i)))
  }
}

// Handles low-rank case, but is slower than ScalarLMM on full-rank case
case class LowRankScalarLMM(con: LMMConstants, delta: Double, logNullS2: Double, useML: Boolean) {
  val n = con.n
  val Uty = con.Uty
  val Utcov = con.UtC

  val invDf = 1d / (if (useML) n else n - con.c)
  val invDelta = 1 / delta

  val Z = (con.S + delta).map(1 / _ - invDelta)
  val ydy = con.yty / delta + (Uty dot (Uty :* Z))
  val UtcovZUtcov = Utcov.t * (Utcov(::, *) :* Z)

  val r0 = 0 to 0
  val r1 = 1 to con.c

  def likelihoodRatioTestLowRank(x: Vector[Double], Utx: DenseVector[Double], CtC: DenseMatrix[Double], UtC: DenseMatrix[Double], Cty: DenseVector[Double], CzC: DenseMatrix[Double]): Annotation = {

    CtC(0, 0) = x dot x
    CtC(r0, r1) := con.C.t * x
    CtC(r1, r0) := CtC(r0, r1).t

    UtC(::, 0) := Utx

    Cty(0) = x dot con.y

    val Cdy = invDelta * Cty + (UtC.t * (Uty :* Z))

    val ZUtx = Utx :* Z

    CzC(0, 0) = Utx dot ZUtx
    CzC(r0, r1) := Utcov.t * ZUtx
    CzC(r1, r0) := CzC(r0, r1).t

    val CdC = invDelta * CtC + CzC

    try {
      val b = CdC \ Cdy
      val s2 = invDf * (ydy - (Cdy dot b))
      val chi2 = n * (logNullS2 - math.log(s2))
      val p = chiSquaredTail(1, chi2)

      Annotation(b(0), s2, chi2, p)
    } catch {
      case e: breeze.linalg.MatrixSingularException => null
    }
  }
}