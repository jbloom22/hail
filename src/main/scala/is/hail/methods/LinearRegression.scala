package is.hail.methods

import breeze.linalg._
import is.hail.annotations.Annotation
import is.hail.expr._
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._

object LinearRegression {
  def apply(vds: VariantDataset, ysExpr: Array[String], xsExpr: Array[String], covsExpr: Array[String], root: String, variantBlockSize: Int): VariantDataset = {
    require(vds.wasSplit)

    val (ys, covs, completeSamples) = RegressionUtils.getPhenosCovCompleteSamples(vds, ysExpr, covsExpr)
    val completeSamplesSet = completeSamples.toSet
    val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray
    val completeSampleIndex = (0 until vds.nSamples).filter(i => completeSamplesSet(vds.sampleIds(i))).toArray // change to mask?
        
    val n = ys.rows // nCompleteSamples
    val nys = ys.cols
    val nxs = xsExpr.length
    val ncovs = covs.cols
    val d = n - nxs - ncovs
    val dRec = 1d / d

    if (nxs == 0) // modify to annotate global only when no field present
      fatal("Must have at least one field")
    
    if (d < 1)
      fatal(s"$n samples with $nxs ${ plural(nxs, "field") } and $ncovs ${ plural(ncovs, "covariate") } implies $d degrees of freedom.")

    info(s"Running linear regression for ${ nys } ${ plural(nys, "phenotype") } on $n samples with $nxs ${ plural(nxs, "field") } and $ncovs ${ plural(ncovs, "covariate") }...")

    val Qt = qr.reduced.justQ(covs).t
    val Qty = Qt * ys

    val sc = vds.sparkContext
    val sampleMaskBc = sc.broadcast(sampleMask)
    val completeSampleIndexBc = sc.broadcast(completeSampleIndex)
    val yBc = sc.broadcast(ys)
    val QtBc = sc.broadcast(Qt)
    val QtyBc = sc.broadcast(Qty)
    val yypBc = sc.broadcast(ys.t(*, ::).map(r => r dot r) - Qty.t(*, ::).map(r => r dot r))

    val pathVA = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    val (newVAS, inserter) = vds.insertVA(LinearRegressionModel.schemaNew, pathVA)

    val vas = vds.vaSignature
    val sas = vds.saSignature
    
    // relation to typ?
    val symTab = Map(
      "v"      -> (0, TVariant),
      "va"     -> (1, vas),
      "s"      -> (2, TString),
      "sa"     -> (3, sas),
      "g"      -> (4, TGenotype),
      "global" -> (5, vds.globalSignature))

    val ec = EvalContext(symTab)
    ec.set(5, vds.globalAnnotation)

    val samplesIds = vds.sampleIds // filter before broadcast?
    val sampleAnnotations = vds.sampleAnnotations

    val sampleIdsBc = sc.broadcast(samplesIds)
    val sampleAnnotationsBc = sc.broadcast(sampleAnnotations)

    val (types, xs) = Parser.parseExprs(xsExpr.mkString(","), ec)
   
    val aToDouble = (types, xsExpr).zipped.map(RegressionUtils.toDouble)
   
    
    if (nxs == 0) {
      vds
    } else if (nxs == 1) {
      val newRDD = vds.rdd.mapPartitions( { it =>
        val missingSamples = new ArrayBuilder[Int]
  
        // columns are genotype vectors
        var X: DenseMatrix[Double] = null
  
        it.grouped(variantBlockSize)
          .flatMap { git =>
            val block = git.toArray
            val blockLength = block.length
  
            if (X == null || X.cols != blockLength)
              X = new DenseMatrix[Double](n, blockLength)
  
            var i = 0
            while (i < blockLength) {
              val (_, (_, gs)) = block(i)
              X(::, i) := RegressionUtils.hardCalls(gs, n, sampleMaskBc.value) // FIXME: replace
              i += 1
            }
  
            val stats = LinearRegressionModel.fitBlock(X, yBc.value, yypBc.value, QtBc.value, QtyBc.value, d: Int, blockLength: Int)
                                    
            (block, stats).zipped.map { case ((v, (va, gs)), stat) => (v, (inserter(va, stat.toAnnotation), gs)) }
          }
      }, preservesPartitioning = true) 
      
      vds.copy(
        rdd = newRDD.asOrderedRDD,
        vaSignature = newVAS)
    } else {
      vds.mapAnnotations{ case (v, va, gs) =>
        ec.set(0, v)
        ec.set(1, va)
  
        val sampleMask = sampleMaskBc.value
        val cols = (0 until nxs).toArray
        val data = Array.ofDim[Double](n * nxs)
        val sums = Array.ofDim[Double](nxs)
        val nMissings = Array.ofDim[Int](nxs)
        val gsIter = gs.iterator
  
        val missingRows = new ArrayBuilder[Int]()
        val missingCols = new ArrayBuilder[Int]()
  
        
        var r = 0
        var i = 0
        while (i < sampleMask.length) {
          val g = gsIter.next()
          if (sampleMask(i)) {
            ec.set(2, sampleIdsBc.value(i))
            ec.set(3, sampleAnnotationsBc.value(i))
            ec.set(4, g)

            (xs(), aToDouble, cols).zipped.map { (e, td, c) =>
              if (e != null) {
                val de = td(e)
                sums(c) += de
                data(c * n + r) = de
              } else {
                nMissings(c) += 1
                missingRows += r
                missingCols += c
              }
            }
            r += 1
          }
          i += 1
        }
        
        val means = (sums, nMissings).zipped.map { case (sum, nMissing) => sum / (n - nMissing) }
        i = 0
        while (i < missingRows.length) {
          val c = missingCols(i)
          data(c * n + missingRows(i)) = means(c)
          i += 1
        }
  
        val X = new DenseMatrix[Double](n, nxs, data)
        val stat = LinearRegressionModel.fit(X, yBc.value, yypBc.value, QtBc.value, QtyBc.value, d)
        
        inserter(va, stat.map(_.toAnnotation).orNull)
      }.copy(vaSignature = newVAS)
    }
  }
}
