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
    val C = cov
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
    val c = C.cols
    val d = n - c - 1

    if (d < 1)
      fatal(s"lmmreg: $n samples and $c ${ plural(c, "covariate") } including intercept implies $d degrees of freedom.")

    info(s"lmmreg: Running lmmreg on $n samples with $c sample ${ plural(c, "covariate") } including intercept...")

    optDelta match {
      case Some(del) => info(s"lmmreg: Delta of $del specified by user")
      case None => info(s"lmmreg: Estimating delta using ${ if (useML) "ML" else "REML" }... ")
    }

    // FIXME: need filtering on eigenDist
    val EigendecompositionDist(_, rowIds, evects, evals) = eigenDist

    if (!completeSamples.sameElements(rowIds))
      fatal("Bad stuff")

    val Ut = evects.transpose
    val S = evals
    val nEigs = S.length

    info(s"lmmreg: Using $nEigs")
    info(s"lmmreg: Evals 1 to ${ math.min(20, nEigs) }: " + ((nEigs - 1) to math.max(0, nEigs - 20) by -1).map(S(_).formatted("%.5f")).mkString(", "))
    info(s"lmmreg: Evals $nEigs to ${ math.max(1, nEigs - 20) }: " + (0 until math.min(nEigs, 20)).map(S(_).formatted("%.5f")).mkString(", "))

    val UtC = Helper.multiply(Ut, C)
    val Uty = Helper.multiply(Ut, y)
    val CtC = C.t * C
    val Cty = C.t * y
    val yty = y.t * y

    val lmmConstants = LMMConstants(y, C, S, Uty, UtC, Cty, CtC, yty, n, c)

    val diagLMM = DiagLMM(lmmConstants, optDelta, useML)

    val vds1 = LinearMixedRegression.globalFit(vds, diagLMM, covExpr, nEigs, S, rootGA, useML)

    val vds2 = if (runAssoc) {
      val sc = vds1.sparkContext
      val sampleMaskBc = sc.broadcast(sampleMask)
      val completeSampleIndexBc = sc.broadcast(completeSampleIndex)

      val (newVAS, inserter) = vds1.insertVA(LinearMixedRegression.schema, pathVA)

      info(s"lmmreg: Computing statistics for each variant...")

      val blockSize = 128
      val useFullRank = nEigs == n

      vds1.persist()

      val (variants, allG) = ToIndexedRowMatrix(vds1, useDosages, sampleMask, completeSampleIndex)
      val variantsBc = sc.broadcast(variants)

      val newRDD =
        if (useFullRank) {
          val Qt = qr.reduced.justQ(diagLMM.TC).t
          val QtTy = Qt * diagLMM.Ty
          val TyQtTy = (diagLMM.Ty dot diagLMM.Ty) - (QtTy dot QtTy)
          val scalarLMM = new FullRankScalarLMM(diagLMM.Ty, diagLMM.TyTy, Qt, QtTy, TyQtTy, diagLMM.logNullS2, useML)

          val scalarLMMBc = sc.broadcast(scalarLMM)

          val projG = (allG.toBlockMatrixDense() * (Ut :* diagLMM.sqrtInvD.toArray).transpose)
            .toIndexedRowMatrix()
            .rows
            .map { case IndexedRow(i, px) => (variantsBc.value(i.toInt), DenseVector(px.toArray)) }
            .orderedRepartitionBy(vds1.rdd.orderedPartitioner)

          vds1.rdd.orderedLeftJoinDistinct(projG).asOrderedRDD.mapPartitions({ it =>
            val missingSamples = new ArrayBuilder[Int]

            // columns are genotype vectors
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

          val projG = (allG.toBlockMatrixDense() * Ut.transpose)
            .toIndexedRowMatrix()
            .rows
            .map { case IndexedRow(i, px) => (variantsBc.value(i.toInt), DenseVector(px.toArray)) }
            .orderedRepartitionBy(vds1.rdd.orderedPartitioner)

          vds1.rdd.orderedLeftJoinDistinct(projG).asOrderedRDD.mapPartitions({ it =>
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

object Helper {
  val dm = DistributedMatrix[BlockMatrix]

  import dm.ops._

  def multiply(bm: BlockMatrix, v: DenseVector[Double]): DenseVector[Double] =
    DenseVector((bm * v.asDenseMatrix.t.asSpark()).toLocalMatrix().asInstanceOf[SparkDenseMatrix].values)

  def multiply(bm: BlockMatrix, m: DenseMatrix[Double]): DenseMatrix[Double] =
    (bm * m.asSpark()).toLocalMatrix().asBreeze().asInstanceOf[DenseMatrix[Double]]
}