package is.hail.methods

import is.hail.SparkSuite
import is.hail.expr.TDouble
import is.hail.keytable.KeyTable
import is.hail.utils._
import is.hail.variant.VariantDataset
import org.apache.spark.sql.Row
import org.testng.annotations.Test

class LogisticRegressionBurdenSuite extends SparkSuite {

  /*
  vdsBurden is shared by testWeightedSum, testWeightedSumWithImputation, and testMax.
  Three genes overlap each other in the first 3 positions:

              Position
        1 2 3 4 5 6 7 8 9 10
  Gene1 ---
  Gene2   -
  Gene3 -----
  */

  val vdsBurden: VariantDataset = hc.importVCF("src/test/resources/regressionLinear.vcf")
    .annotateVariantsIntervals("src/test/resources/regressionLinear.interval_list", "va.genes", all=true)
    .annotateVariantsExpr("va.weight = v.start.toDouble")
    .annotateSamplesTable("src/test/resources/regressionLinear.pheno",
      "Sample",
      root = Some("sa.pheno0"),
      config = TextTableConfiguration(types = Map("Pheno" -> TDouble), missing = "0"))
    .annotateSamplesExpr("sa.pheno.Pheno = if (sa.pheno0.Pheno == 1.0) false else if (sa.pheno0.Pheno == 2.0) true else NA: Boolean")
    .annotateSamplesTable("src/test/resources/regressionLinear.cov",
      "Sample",
      root = Some("sa.cov"),
      config = TextTableConfiguration(types = Map("Cov1" -> TDouble, "Cov2" -> TDouble)))

  def makeMap(sampleKT: KeyTable): Map[Any, IndexedSeq[java.lang.Double]] = sampleKT.collect().map { r =>
    val s = r.asInstanceOf[Row].toSeq
    s.head -> s.tail.map(_.asInstanceOf[java.lang.Double]).toIndexedSeq }
    .toMap

  def assertMap(mapR: Map[Int, IndexedSeq[java.lang.Double]], map: Map[Any, IndexedSeq[java.lang.Double]], tol: Double) {
    (1 to 3).foreach { i =>
      (mapR(i), map(s"Gene$i")).zipped.foreach( (x, y) =>
        assert((x == null && y == null) || D_==(x.doubleValue(), y.doubleValue(), tolerance = tol))
      )
    }
  }

  @Test def testWeightedSum() {
    val (logregKT, sampleKT) = vdsBurden.logregBurden("gene", "va.genes",
      "gs.map(g => va.weight * g.gt).sum()", "wald", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))

    println(logregKT.collect().toIndexedSeq)
    println(sampleKT.collect().toIndexedSeq)

//    val linregMap = makeMap(logregKT)
//    val sampleMap = makeMap(sampleKT)

    /*
    Variant A       B       C       D       E       F       Weight
    1       0.0     1.0     0.0     0.0     0.0     1.0     1.0
    2         .     2.0       .     2.0     0.0     0.0     2.0
    3       0.0       .     1.0     1.0     1.0       .     3.0

    Sample  Cov1    Cov2    Pheno   Gene1   Gene2   Gene3
    A       0.0     -1.0    1.0     0.0     0.0     0.0
    B       2.0     3.0     1.0     5.0     4.0     5.0
    C       1.0     5.0     2.0     0.0     0.0     3.0
    D       -2.0    0.0     2.0     4.0     4.0     7.0
    E       -2.0    -4.0    2.0     0.0     0.0     3.0
    F       4.0     3.0     2.0     1.0     0.0     1.0
    */

    // Values from R: fit <- lm(Pheno ~ Gene1 + Cov1 + Cov2, data=df)
//    val linregMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
//      1 -> IndexedSeq(-0.08164, 0.15339, -0.532, 0.6478),
//      2 -> IndexedSeq(-0.09900, 0.17211, -0.575, 0.6233),
//      3 -> IndexedSeq(0.01558, 0.18323, 0.085, 0.940))
//
//    val sampleMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
//      1 -> IndexedSeq(0.0, 5.0, 0.0, 4.0, 0.0, 1.0),
//      2 -> IndexedSeq(0.0, 4.0, 0.0, 4.0, 0.0, 0.0),
//      3 -> IndexedSeq(0.0, 5.0, 3.0, 7.0, 3.0, 1.0))
//
//    assertMap(linregMapR, linregMap, 1e-3)
//    assertMap(sampleMapR, sampleMap, 1e-6)
  }

  /*
  @Test def testWeightedSumWithImputation() {

    val (linregKT, sampleKT) = vdsBurden
      .filterSamplesExpr("isDefined(sa.pheno.Pheno) && isDefined(sa.cov.Cov1) && isDefined(sa.cov.Cov2)")
      .variantQC()
      .linregBurden("gene", "va.genes",
      "gs.map(g => va.weight * orElse(g.gt.toDouble, 2 * va.qc.AF)).sum()",
      "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val linregMap = makeMap(linregKT)
    val sampleMap = makeMap(sampleKT)

    /*
    Variant A       B       C       D       E       F       Weight
    1       0.0     1.0     0.0     0.0     0.0     1.0     1.0
    2       1.0     2.0     1.0     2.0     0.0     0.0     2.0
    3       0.0     .75     1.0     1.0     1.0     .75     3.0

    Sample  Cov1    Cov2    Pheno   Gene1   Gene2   Gene3
    A       0.0     -1.0    1.0     2.0     2.0     2.0
    B       2.0     3.0     1.0     5.0     4.0     7.25
    C       1.0     5.0     2.0     2.0     2.0     5.0
    D       -2.0    0.0     2.0     4.0     4.0     7.0
    E       -2.0    -4.0    2.0     0.0     0.0     3.0
    F       4.0     3.0     2.0     1.0     0.0     3.25
    */

    // Values from R: fit <- lm(Pheno ~ Gene1 + Cov1 + Cov2, data=df)
    val linregMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(-0.2177, 0.1621, -1.343, 0.3115),
      2 -> IndexedSeq(-0.2709, 0.1675, -1.617, 0.2473),
      3 -> IndexedSeq(-0.05710, 0.21078, -0.271, 0.812))

    val sampleMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(2.0, 5.0, 2.0, 4.0, 0.0, 1.0),
      2 -> IndexedSeq(2.0, 4.0, 2.0, 4.0, 0.0, 0.0),
      3 -> IndexedSeq(2.0, 7.25, 5.0, 7.0, 3.0, 3.25))


    assertMap(linregMapR, linregMap, 1e-3)
    assertMap(sampleMapR, sampleMap, 1e-6)
    }

  @Test def testMax() {
    val (linregKT, sampleKT) = vdsBurden.linregBurden("gene", "va.genes",
      "gs.map(g => g.gt.toDouble).max()", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1", "sa.cov.Cov2"))

    val linregMap = makeMap(linregKT)
    val sampleMap = makeMap(sampleKT)

    /*
    Variant A       B       C       D       E       F
    1       0.0     1.0     0.0     0.0     0.0     1.0
    2         .     2.0       .     2.0     0.0     0.0
    3       0.0       .     1.0     1.0     1.0       .

    Sample  Cov1    Cov2    Pheno   Gene1   Gene2   Gene3
    A       0.0     -1.0    1.0     0.0       .     0.0
    B       2.0     3.0     1.0     2.0     2.0     2.0
    C       1.0     5.0     2.0     0.0       .     1.0
    D       -2.0    0.0     2.0     2.0     2.0     2.0
    E       -2.0    -4.0    2.0     0.0     0.0     1.0
    F       4.0     3.0     2.0     1.0     0.0     1.0
    */

    // Values from R: fit <- lm(Pheno ~ Gene1 + Cov1 + Cov2, data=df)
    val linregMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(-0.08368, 0.36841, -0.227, 0.8414),
      2 -> IndexedSeq(-0.5418, 0.3351, -1.617, 0.2473),
      3 -> IndexedSeq(0.07474, 0.51528, 0.145, 0.898))

    val sampleMapR: Map[Int, IndexedSeq[java.lang.Double]] = Map(
      1 -> IndexedSeq(0.0, 2.0, 0.0, 2.0, 0.0, 1.0),
      2 -> IndexedSeq(null, 2.0, null, 2.0, 0.0, 0.0),
      3 -> IndexedSeq(0.0, 2.0, 1.0, 2.0, 1.0, 1.0))


    assertMap(linregMapR, linregMap, 1e-3)
    assertMap(sampleMapR, sampleMap, 1e-6)
  }

  @Test def testSingleVsArray() {
    /*
    Three disjoint genes are in the first three positions.

                Position
          1 2 3 4 5 6 7 8 9 10
    Gene1 -
    Gene2   -
    Gene3     -
    */

    val vdsBurdenNoOverlap: VariantDataset = hc.importVCF("src/test/resources/regressionLinear.vcf")
      .annotateVariantsIntervals("src/test/resources/regressionLinearNoOverlap.interval_list", "va.gene")
      .annotateVariantsExpr("va.weight = v.start.toDouble")
      .annotateVariantsExpr("""va.genes2 = if (isDefined(va.gene)) [va.gene] else range(0).map(x => "")""")
      .annotateVariantsExpr("va.genes3 = if (isDefined(va.gene)) [va.gene] else NA: Array[String]")
      .annotateVariantsExpr("va.genes4 = if (isDefined(va.gene)) [va.gene, va.gene].toSet else NA: Set[String]")
      .annotateVariantsExpr("va.genes5 = if (isDefined(va.gene)) [va.gene, va.gene] else NA: Array[String]")
      .annotateVariantsExpr("va.genes6 = NA: Set[String]")
      .annotateVariantsExpr("va.genes7 = v.start")
      .annotateVariantsExpr("va.genes8 = v.start.toDouble")
      .annotateSamplesTable("src/test/resources/regressionLinear.pheno",
        "Sample",
        root = Some("sa.pheno"),
        config = TextTableConfiguration(types = Map("Pheno" -> TDouble), missing = "0"))
      .annotateSamplesTable("src/test/resources/regressionLinear.cov",
        "Sample",
        root = Some("sa.cov"),
        config = TextTableConfiguration(types = Map("Cov1" -> TDouble, "Cov2" -> TDouble)))

    val (linregKT, sampleKT) = vdsBurdenNoOverlap.linregBurden("gene", "va.gene",
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1"),
      singleKey = true)

    val (linregKT2, sampleKT2) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes2",
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1"))

    val (linregKT3, sampleKT3) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes3",
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1"))

    val (linregKT4, sampleKT4) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes4",
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1"))

    val (linregKT5, sampleKT5) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes5",
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1"))

    val (linregKT6, sampleKT6) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes6",
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1"))

    val (linregKT7, sampleKT7) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes7",
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1"),
      singleKey = true)

    val (linregKT8, sampleKT8) = vdsBurdenNoOverlap.linregBurden("gene", "va.genes8",
      "gs.map(g => va.weight * g.gt).sum()", "sa.pheno.Pheno", covariates = Array("sa.cov.Cov1"),
      singleKey = true)

    assert(linregKT same linregKT2)
    assert(sampleKT same sampleKT2)

    assert(linregKT same linregKT3)
    assert(sampleKT same sampleKT3)

    assert(linregKT same linregKT4)
    assert(sampleKT same sampleKT4)

    val twiceSampleMap = makeMap(sampleKT).mapValues(_.map(2 * _))
    val onceSampleMap5 = makeMap(sampleKT5).mapValues(_.map(1 * _))
    assert(twiceSampleMap == onceSampleMap5)

    assert(linregKT6.nRows == 0)
    assert(sampleKT6.nRows == 0)

    val sampleMap7 = makeMap(sampleKT7)
    assert(sampleMap7.size == 10)
    assert(sampleMap7.forall { case (key, value) => value.forall(_ % key.asInstanceOf[Int] == 0) })

    val sampleMap8 = makeMap(sampleKT8)
    assert(sampleMap8.size == 10)
    assert(sampleMap8.forall { case (key, value) => value.forall(_ % key.asInstanceOf[Double] == 0) })
  }
  */
}
