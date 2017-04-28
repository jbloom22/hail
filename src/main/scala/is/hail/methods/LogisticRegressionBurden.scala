package is.hail.methods

import breeze.linalg._
import is.hail.expr._
import is.hail.keytable.KeyTable
import is.hail.stats._
import is.hail.utils._
import is.hail.variant._
import org.apache.spark.sql.Row

object LogisticRegressionBurden {

  def apply(vds: VariantDataset,
    keyName: String,
    variantKeys: String,
    aggExpr: String,
    test: String,
    yExpr: String,
    covExpr: Array[String],
    singleKey: Boolean): (KeyTable, KeyTable) = {

    def tests = Map("wald" -> WaldTest, "lrt" -> LikelihoodRatioTest, "score" -> ScoreTest, "firth" -> FirthTest)

    if (!tests.isDefinedAt(test))
      fatal(s"Supported tests are ${ tests.keys.mkString(", ") }, got: $test")

    val logRegTest = tests(test)

    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
    val completeSamplesSet = completeSamples.toSet
    val sampleMask = vds.sampleIds.map(completeSamplesSet).toArray

    if (!y.forall(yi => yi == 0d || yi == 1d))
      fatal(s"For logistic regression, phenotype must be Boolean or numeric with all values equal to 0 or 1")

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")


    info(s"Running $test logistic regression, aggregated by key $keyName on $n samples with $k ${ plural(k, "covariate") } including intercept...")

    val nullModel = new LogisticRegressionModel(cov, y)
    val nullFit = nullModel.fit()

    if (!nullFit.converged)
      fatal("Failed to fit (unregulatized) logistic regression null model (covariates only): " + (
        if (nullFit.exploded)
          s"exploded at Newton iteration ${ nullFit.nIter }"
        else
          "Newton iteration failed to converge"))

    if (completeSamplesSet(keyName))
      fatal(s"Sample name conflicts with the key name $keyName")

    def sampleKT = vds.filterSamples((s, sa) => completeSamplesSet(s))
      .aggregateBySamplePerVariantKey(keyName, variantKeys, aggExpr, singleKey)

    val keyType = sampleKT.fields(0).typ

    println(keyType)

    // d > 0 implies at least 1 sample
    val numericType = sampleKT.fields(1).typ

    if (!numericType.isInstanceOf[TNumeric])
      fatal(s"aggregate_expr type must be numeric, found $numericType")

    val sc = sampleKT.hc.sc
    val sampleMaskBc = sc.broadcast(sampleMask)
    val yBc = sc.broadcast(y)
    val XBc = sc.broadcast(new DenseMatrix[Double](n, k + 1, cov.toArray ++ Array.ofDim[Double](n)))
    val nullFitBc = sc.broadcast(nullFit)
    val logRegTestBc = sc.broadcast(logRegTest)

    val nSamplesBeforeMask = vds.nSamples
    val emptyStats = logRegTest.emptyStats

    val logregRDD = sampleKT.rdd.mapPartitions({ it =>
      val X = XBc.value.copy
      it.map { keyedRow =>
        val key = keyedRow.get(0)

        if (RegressionUtils.setLastColumnBurden(X, keyedRow))
          logRegTestBc.value.test(X, yBc.value, nullFitBc.value).toAnnotation(emptyStats).asInstanceOf[Row]
        else
          Row(key +: emptyStats)
      }
    })

    def logregSignature = TStruct(keyName -> keyType).merge(logRegTest.schema.asInstanceOf[TStruct])._1

    val logregKT = new KeyTable(sampleKT.hc, logregRDD, signature = logregSignature, keyNames = Array(keyName))

    (logregKT, sampleKT)
  }
}