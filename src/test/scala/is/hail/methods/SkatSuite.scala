package is.hail.methods

import is.hail.expr._
import is.hail.SparkSuite
import is.hail.io.annotators.IntervalList
import is.hail.variant.VariantDataset
import org.testng.annotations.Test

class SkatSuite extends SparkSuite {
  def covariates = hc.importTable("src/test/resources/regressionLinear.cov",
    types = Map("Cov1" -> TDouble, "Cov2" -> TDouble)).keyBy("Sample")

  def phenotypes = hc.importTable("src/test/resources/regressionLinear.pheno",
    types = Map("Pheno" -> TDouble), missing = "0").keyBy("Sample")

  def intervals = IntervalList.read(hc, "src/test/resources/regressionLinear.interval_list")

  def vdsBurden: VariantDataset = hc.importVCF("src/test/resources/regressionLinear.vcf")
    .annotateVariantsTable(intervals, root="va.genes", product=true)
    .annotateVariantsExpr("va.weight = v.start.toDouble")
    .annotateSamplesTable(phenotypes, root="sa.pheno0")
    .annotateSamplesTable(covariates, root="sa.cov")
    .annotateSamplesExpr("sa.pheno = if (sa.pheno0 == 1.0) false else if (sa.pheno0 == 2.0) true else NA: Boolean")

  @Test def test() {
    println(None.isEmpty)
    vdsBurden.skat("gene", "va.genes", singleKey = false, "va.weight", "sa.pheno",
                   covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))


  }
}
