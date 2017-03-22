package is.hail.variant

import java.nio.ByteBuffer
import java.util
import scala.collection.JavaConverters._

import is.hail.annotations._
import is.hail.check.Gen
import is.hail.expr.{EvalContext, TAggregable, _}
import is.hail.io._
import is.hail.io.annotators.{BedAnnotator, IntervalListAnnotator}
import is.hail.io.plink.{FamFileConfig, PlinkLoader}
import is.hail.keytable.KeyTable
import is.hail.methods.{Aggregators, DuplicateReport, Filter, VEP}
import is.hail.sparkextras._
import is.hail.utils._
import is.hail.variant.Variant.orderedKey
import is.hail.{HailContext, utils}
import org.apache.hadoop
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkContext, SparkEnv}
import org.json4s._
import org.json4s.jackson.Serialization

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.io.Source
import scala.language.implicitConversions
import scala.reflect.ClassTag

object VariantSampleMatrix {
  final val fileVersion: Int = 4

  def apply[T](hc: HailContext, metadata: VariantMetadata,
    rdd: OrderedRDD[Locus, Variant, (Annotation, Iterable[T])])(implicit tct: ClassTag[T]): VariantSampleMatrix[T] = {
    new VariantSampleMatrix(hc, metadata, rdd)
  }

  def writePartitioning(sqlContext: SQLContext, dirname: String): Unit = {
    val sc = sqlContext.sparkContext
    val hConf = sc.hadoopConfiguration

    if (hConf.exists(dirname + "/partitioner.json.gz")) {
      warn("write partitioning: partitioner.json.gz already exists, nothing to do")
      return
    }

    val parquetFile = dirname + "/rdd.parquet"

    val fastKeys = sqlContext.readParquetSorted(parquetFile, Some(Array("variant")))
      .map(_.getVariant(0))
    val kvRDD = fastKeys.map(k => (k, ()))

    val ordered = kvRDD.toOrderedRDD(fastKeys)

    hConf.writeTextFile(dirname + "/partitioner.json.gz") { out =>
      Serialization.write(ordered.orderedPartitioner.toJSON, out)
    }
  }

  def gen[T](hc: HailContext,
    gen: VSMSubgen[T])(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] =
    gen.gen(hc)

  def genGeneric(hc: HailContext): Gen[VariantSampleMatrix[Annotation]] =
    for (tSig <- Type.genArb.resize(3);
      vsm <- VSMSubgen[Annotation](
        sampleIdGen = Gen.distinctBuildableOf[IndexedSeq, String](Gen.identifier),
        saSigGen = Type.genArb,
        vaSigGen = Type.genArb,
        globalSigGen = Type.genArb,
        tSig = tSig,
        saGen = (t: Type) => t.genValue,
        vaGen = (t: Type) => t.genValue,
        globalGen = (t: Type) => t.genValue,
        vGen = Variant.gen,
        tGen = (v: Variant) => tSig.genValue.resize(20),
        isGenericGenotype = true).gen(hc)
    ) yield vsm
}

case class VSMSubgen[T](
  sampleIdGen: Gen[IndexedSeq[String]],
  saSigGen: Gen[Type],
  vaSigGen: Gen[Type],
  globalSigGen: Gen[Type],
  tSig: Type,
  saGen: (Type) => Gen[Annotation],
  vaGen: (Type) => Gen[Annotation],
  globalGen: (Type) => Gen[Annotation],
  vGen: Gen[Variant],
  tGen: (Variant) => Gen[T],
  isDosage: Boolean = false,
  wasSplit: Boolean = false,
  isGenericGenotype: Boolean = false) {

  def gen(hc: HailContext)(implicit tct: ClassTag[T]): Gen[VariantSampleMatrix[T]] =
    for (size <- Gen.size;
      subsizes <- Gen.partitionSize(5).resize(size / 10);
      vaSig <- vaSigGen.resize(subsizes(0));
      saSig <- saSigGen.resize(subsizes(1));
      globalSig <- globalSigGen.resize(subsizes(2));
      global <- globalGen(globalSig).resize(subsizes(3));
      nPartitions <- Gen.choose(1, 10);

      (l, w) <- Gen.squareOfAreaAtMostSize.resize((size / 10) * 9);

      sampleIds <- sampleIdGen.resize(w);
      nSamples = sampleIds.length;
      saValues <- Gen.buildableOfN[IndexedSeq, Annotation](nSamples, saGen(saSig)).resize(subsizes(4));
      rows <- Gen.distinctBuildableOf[Seq, (Variant, (Annotation, Iterable[T]))](
        for (subsubsizes <- Gen.partitionSize(3);
          v <- vGen.resize(subsubsizes(0));
          va <- vaGen(vaSig).resize(subsubsizes(1));
          ts <- Gen.buildableOfN[Iterable, T](nSamples, tGen(v)).resize(subsubsizes(2)))
          yield (v, (va, ts))).resize(l))
      yield {
        VariantSampleMatrix[T](hc, VariantMetadata(sampleIds, saValues, global, saSig, vaSig, globalSig, tSig, wasSplit = wasSplit, isDosage = isDosage, isGenericGenotype = isGenericGenotype),
          hc.sc.parallelize(rows, nPartitions).toOrderedRDD)
      }
}

object VSMSubgen {
  val random = VSMSubgen[Genotype](
    sampleIdGen = Gen.distinctBuildableOf[IndexedSeq, String](Gen.identifier),
    saSigGen = Type.genArb,
    vaSigGen = Type.genArb,
    globalSigGen = Type.genArb,
    tSig = TGenotype,
    saGen = (t: Type) => t.genValue,
    vaGen = (t: Type) => t.genValue,
    globalGen = (t: Type) => t.genValue,
    vGen = Variant.gen,
    tGen = Genotype.genExtreme)

  val plinkSafeBiallelic = random.copy(
    sampleIdGen = Gen.distinctBuildableOf[IndexedSeq, String](Gen.plinkSafeIdentifier),
    vGen = VariantSubgen.plinkCompatible.copy(nAllelesGen = Gen.const(2)).gen,
    wasSplit = true)

  val realistic = random.copy(
    tGen = Genotype.genRealistic)

  val dosage = random.copy(
    tGen = Genotype.genDosage, isDosage = true)
}

class VariantSampleMatrix[T](val hc: HailContext, val metadata: VariantMetadata,
  val rdd: OrderedRDD[Locus, Variant, (Annotation, Iterable[T])])(implicit tct: ClassTag[T]) extends JoinAnnotator {

  lazy val sampleIdsBc = sparkContext.broadcast(sampleIds)

  lazy val sampleAnnotationsBc = sparkContext.broadcast(sampleAnnotations)

  def genotypeSignature = metadata.genotypeSignature

  def isGenericGenotype: Boolean = metadata.isGenericGenotype

  /**
    * Aggregate by user-defined key and aggregation expressions.
    *
    * Equivalent of a group-by operation in SQL.
    *
    * @param keyExpr Named expression(s) for which fields are keys
    * @param aggExpr Named aggregation expression(s)
    */
  def aggregateByKey(keyExpr: String, aggExpr: String): KeyTable = {
    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vaSignature),
      "s" -> (3, TSample),
      "sa" -> (4, saSignature),
      "g" -> (5, genotypeSignature))

    val ec = EvalContext(aggregationST.map { case (name, (i, t)) => name -> (i, TAggregable(t, aggregationST)) })

    val keyEC = EvalContext(Map(
      "global" -> (0, globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vaSignature),
      "s" -> (3, TSample),
      "sa" -> (4, saSignature),
      "g" -> (5, genotypeSignature)))

    val (keyNames, keyTypes, keyF) = Parser.parseNamedExprs(keyExpr, keyEC)
    val (aggNames, aggTypes, aggF) = Parser.parseNamedExprs(aggExpr, ec)

    val signature = TStruct((keyNames ++ aggNames, keyTypes ++ aggTypes).zipped.toSeq: _*)

    val (zVals, seqOp, combOp, resultOp) = Aggregators.makeFunctions[Annotation](ec, { case (ec, a) =>
      ec.setAllFromRow(a.asInstanceOf[Row])
    })

    val localGlobalAnnotation = globalAnnotation

    val ktRDD = mapPartitionsWithAll { it =>
      it.map { case (v, va, s, sa, g) =>
        keyEC.setAll(localGlobalAnnotation, v, va, s, sa, g)
        val key = Annotation.fromSeq(keyF())
        (key, Annotation(localGlobalAnnotation, v, va, s, sa, g))
      }
    }.aggregateByKey(zVals)(seqOp, combOp)
      .map { case (k, agg) =>
        resultOp(agg)
        Annotation.fromSeq(k.asInstanceOf[Row].toSeq ++ aggF())
      }

    KeyTable(hc, ktRDD, signature, keyNames)
  }

  def aggregateBySample[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(String, U)] =
    aggregateBySampleWithKeys(zeroValue)((e, v, s, g) => seqOp(e, g), combOp)

  def aggregateBySampleWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, String, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(String, U)] = {
    aggregateBySampleWithAll(zeroValue)((e, v, va, s, sa, g) => seqOp(e, v, s, g), combOp)
  }

  def aggregateBySampleWithAll[U](zeroValue: U)(
    seqOp: (U, Variant, Annotation, String, Annotation, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(String, U)] = {

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .mapPartitions { (it: Iterator[(Variant, (Annotation, Iterable[T]))]) =>
        val serializer = SparkEnv.get.serializer.newInstance()

        def copyZeroValue() = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

        val arrayZeroValue = Array.fill[U](localSampleIdsBc.value.length)(copyZeroValue())

        localSampleIdsBc.value.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, (va, gs))) =>
            for ((g, i) <- gs.iterator.zipWithIndex) {
              acc(i) = seqOp(acc(i), v, va,
                localSampleIdsBc.value(i), localSampleAnnotationsBc.value(i), g)
            }
            acc
          }.iterator)
      }.foldByKey(zeroValue)(combOp)
  }

  def aggregateByVariant[U](zeroValue: U)(
    seqOp: (U, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Variant, U)] =
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, sa, g) => seqOp(e, g), combOp)

  def aggregateByVariantWithAll[U](zeroValue: U)(
    seqOp: (U, Variant, Annotation, String, Annotation, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Variant, U)] = {

    // Serialize the zero value to a byte array so that we can apply a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .mapPartitions({ (it: Iterator[(Variant, (Annotation, Iterable[T]))]) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        it.map { case (v, (va, gs)) =>
          val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))
          (v, gs.iterator.zipWithIndex.map { case (g, i) => (localSampleIdsBc.value(i), localSampleAnnotationsBc.value(i), g) }
            .foldLeft(zeroValue) { case (acc, (s, sa, g)) =>
              seqOp(acc, v, va, s, sa, g)
            })
        }
      }, preservesPartitioning = true)

    /*
        rdd
          .map { case (v, gs) =>
            val serializer = SparkEnv.get.serializer.newInstance()
            val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

            (v, gs.zipWithIndex.foldLeft(zeroValue) { case (acc, (g, i)) =>
              seqOp(acc, v, localSamplesBc.value(i), g)
            })
          }
    */
  }

  def aggregateByVariantWithKeys[U](zeroValue: U)(
    seqOp: (U, Variant, String, T) => U,
    combOp: (U, U) => U)(implicit uct: ClassTag[U]): RDD[(Variant, U)] = {
    aggregateByVariantWithAll(zeroValue)((e, v, va, s, sa, g) => seqOp(e, v, s, g), combOp)
  }

  /**
    * Aggregate over intervals and export.
    *
    * @param intervalList Input interval list file
    * @param expr Export expression
    * @param out Output file path
    */
  def aggregateIntervals(intervalList: String, expr: String, out: String) {

    val vas = vaSignature
    val sas = saSignature
    val localGlobalAnnotation = globalAnnotation

    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "interval" -> (1, TInterval),
      "v" -> (2, TVariant),
      "va" -> (3, vas))
    val symTab = Map(
      "global" -> (0, globalSignature),
      "interval" -> (1, TInterval),
      "variants" -> (2, TAggregable(TVariant, aggregationST)))

    val ec = EvalContext(symTab)
    ec.set(1, globalAnnotation)

    val (names, _, f) = Parser.parseExportExprs(expr, ec)

    val definedNames = names match {
      case Some(arr) => arr
      case None => fatal("this module requires one or more named export expressions")
    }

    val (zVals, seqOp, combOp, resultOp) =
      Aggregators.makeFunctions[(Interval[Locus], Variant, Annotation)](ec, { case (ec, (i, v, va)) =>
        ec.setAll(localGlobalAnnotation, i, v, va)
      })

    val iList = IntervalListAnnotator.read(intervalList, hc.hadoopConf)
    val iListBc = sparkContext.broadcast(iList)

    val results = variantsAndAnnotations.flatMap { case (v, va) =>
      iListBc.value.query(v.locus).map { i => (i, (i, v, va)) }
    }
      .aggregateByKey(zVals)(seqOp, combOp)
      .collectAsMap()

    hc.hadoopConf.writeTextFile(out) { out =>
      val sb = new StringBuilder
      sb.append("Contig")
      sb += '\t'
      sb.append("Start")
      sb += '\t'
      sb.append("End")
      definedNames.foreach { col =>
        sb += '\t'
        sb.append(col)
      }
      sb += '\n'

      iList.toIterator
        .foreachBetween { interval =>

          sb.append(interval.start.contig)
          sb += '\t'
          sb.append(interval.start.position)
          sb += '\t'
          sb.append(interval.end.position)
          val res = results.getOrElse(interval, zVals)
          resultOp(res)

          ec.setAll(localGlobalAnnotation, interval)
          f().foreach { field =>
            sb += '\t'
            sb.append(field)
          }
        }(sb += '\n')

      out.write(sb.result())
    }
  }

  def annotateGlobal(a: Annotation, t: Type, code: String): VariantSampleMatrix[T] = {
    val (newT, i) = insertGlobal(t, Parser.parseAnnotationRoot(code, Annotation.GLOBAL_HEAD))
    copy(globalSignature = newT, globalAnnotation = i(globalAnnotation, a))
  }

  /**
    * Create and destroy global annotations with expression language.
    *
    * @param expr Annotation expression
    */
  def annotateGlobalExpr(expr: String): VariantSampleMatrix[T] = {
    val ec = EvalContext(Map(
      "global" -> (0, globalSignature)))

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Option(Annotation.GLOBAL_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]

    val finalType = (paths, types).zipped.foldLeft(globalSignature) { case (v, (ids, signature)) =>
      val (s, i) = v.insert(signature, ids)
      inserterBuilder += i
      s
    }

    val inserters = inserterBuilder.result()

    ec.set(0, globalAnnotation)
    val ga = inserters
      .zip(f())
      .foldLeft(globalAnnotation) { case (a, (ins, res)) =>
        ins(a, res)
      }

    copy(globalAnnotation = ga,
      globalSignature = finalType)
  }

  /**
    * Load text file into global annotations as Array[String] or
    *   Set[String].
    *
    * @param path Input text file
    * @param root Global annotation path to store text file
    * @param asSet If true, load text file as Set[String],
    *   otherwise, load as Array[String]
    */
  def annotateGlobalList(path: String, root: String, asSet: Boolean = false): VariantSampleMatrix[T] = {
    val textList = hc.hadoopConf.readFile(path) { in =>
      Source.fromInputStream(in)
        .getLines()
        .toArray
    }

    val (sig, toInsert) =
      if (asSet)
        (TSet(TString), textList.toSet)
      else
        (TArray(TString), textList: IndexedSeq[String])

    val rootPath = Parser.parseAnnotationRoot(root, "global")

    val (newGlobalSig, inserter) = insertGlobal(sig, rootPath)

    copy(
      globalAnnotation = inserter(globalAnnotation, toInsert),
      globalSignature = newGlobalSig)
  }

  def globalAnnotation: Annotation = metadata.globalAnnotation

  def insertGlobal(sig: Type, path: List[String]): (Type, Inserter) = {
    globalSignature.insert(sig, path)
  }

  def globalSignature: Type = metadata.globalSignature

  /**
    * Load delimited text file (text table) into global annotations as
    *   Array[Struct].
    *
    * @param path Input text file
    * @param root Global annotation path to store text table
    * @param config Configuration options for importing text files
    */
  def annotateGlobalTable(path: String, root: String,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {
    val annotationPath = Parser.parseAnnotationRoot(root, Annotation.GLOBAL_HEAD)

    val (struct, rdd) = TextTableReader.read(sparkContext)(Array(path), config)
    val arrayType = TArray(struct)

    val (finalType, inserter) = insertGlobal(arrayType, annotationPath)

    val table = rdd
      .map(_.value)
      .collect(): IndexedSeq[Annotation]

    copy(
      globalAnnotation = inserter(globalAnnotation, table),
      globalSignature = finalType)
  }

  def annotateIntervals(is: IntervalTree[Locus],
    path: List[String]): VariantSampleMatrix[T] = {
    val isBc = sparkContext.broadcast(is)
    val (newSignature, inserter) = insertVA(TBoolean, path)
    copy(rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
      (inserter(va, isBc.value.contains(v.locus)), gs)
    }.asOrderedRDD,
      vaSignature = newSignature)
  }

  def annotateIntervals(is: IntervalTree[Locus],
    t: Type,
    m: Map[Interval[Locus], List[String]],
    all: Boolean,
    path: List[String]): VariantSampleMatrix[T] = {
    val isBc = sparkContext.broadcast(is)

    val mBc = sparkContext.broadcast(m)
    val (newSignature, inserter) = insertVA(
      if (all) TSet(t) else t,
      path)
    copy(rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
      val queries = isBc.value.query(v.locus)
      val toIns = if (all)
        queries.flatMap(mBc.value)
      else {
        queries.flatMap(mBc.value).headOption.orNull
      }
      (inserter(va, toIns), gs)
    }.asOrderedRDD,
      vaSignature = newSignature)
  }

  def annotateSamples(signature: Type, path: List[String], annotation: (String) => Annotation): VariantSampleMatrix[T] = {
    val (t, i) = insertSA(signature, path)
    annotateSamples(annotation, t, i)
  }

  def annotateSamplesExpr(expr: String): VariantSampleMatrix[T] = {
    val ec = sampleEC

    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.SAMPLE_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(saSignature) { case (sas, (ids, signature)) =>
      val (s, i) = sas.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val sampleAggregationOption = Aggregators.buildSampleAggregations(this, ec)

    ec.set(0, globalAnnotation)
    val newAnnotations = sampleIdsAndAnnotations.map { case (s, sa) =>
      sampleAggregationOption.foreach(f => f.apply(s))
      ec.set(1, s)
      ec.set(2, sa)
      f().zip(inserters)
        .foldLeft(sa) { case (sa, (v, inserter)) =>
          inserter(sa, v)
        }
    }

    copy(
      sampleAnnotations = newAnnotations,
      saSignature = finalType
    )
  }

  /**
    * Import PLINK .fam file into sample annotations.
    *
    * @param path Path to .fam file
    * @param root Sample annotation path at which to store .fam file
    * @param config .fam file configuration options
    */
  def annotateSamplesFam(path: String, root: String = "sa.fam",
    config: FamFileConfig = FamFileConfig()): VariantSampleMatrix[T] = {
    if (!path.endsWith(".fam"))
      fatal("input file must end in .fam")

    val (info, signature) = PlinkLoader.parseFam(path, config, hc.hadoopConf)

    val duplicateIds = info.map(_._1).duplicates().toArray
    if (duplicateIds.nonEmpty) {
      val n = duplicateIds.length
      fatal(
        s"""found $n duplicate sample ${ plural(n, "id") }:
           |  @1""".stripMargin, duplicateIds)
    }

    annotateSamples(info.toMap, signature, root)
  }

  def annotateSamplesList(path: String, root: String): VariantSampleMatrix[T] = {

    val samplesInList = hc.hadoopConf.readLines(path) { lines =>
      if (lines.isEmpty)
        warn(s"Empty annotation file given: $path")

      lines.map(_.value).toSet
    }

    val sampleAnnotations = sampleIds.map { s => (s, samplesInList.contains(s)) }.toMap
    annotateSamples(sampleAnnotations, TBoolean, root)
  }

  def annotateSamples(annotations: Map[String, Annotation], signature: Type, code: String): VariantSampleMatrix[T] = {
    val (t, i) = insertSA(signature, Parser.parseAnnotationRoot(code, Annotation.SAMPLE_HEAD))
    annotateSamples(s => annotations.getOrElse(s, null), t, i)
  }

  def annotateSamplesTable(path: String, sampleExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (struct, rdd) = TextTableReader.read(sparkContext)(Array(path), config)

    val (finalType, inserter): (Type, (Annotation, Annotation) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "sa" -> (0, saSignature),
          "table" -> (1, struct)))
        Annotation.buildInserter(annotationExpr, saSignature, ec, Annotation.SAMPLE_HEAD)
      } else
        insertSA(struct, Parser.parseAnnotationRoot(annotationExpr, Annotation.SAMPLE_HEAD))

    val sampleQuery = struct.parseInStructScope[String](sampleExpr)

    val map = rdd
      .map {
        _.map { a =>
          (sampleQuery(a), a)
        }.value
      }
      .filter { case (s, a) => s != null }
      .collect()
      .toMap

    val vdsKeys = sampleIds.toSet
    val tableKeys = map.keySet
    val onlyVds = vdsKeys -- tableKeys
    val onlyTable = tableKeys -- vdsKeys
    if (onlyVds.nonEmpty) {
      warn(s"There were ${ onlyVds.size } samples present in the VDS but not in the table.")
    }
    if (onlyTable.nonEmpty) {
      warn(s"There were ${ onlyTable.size } samples present in the table but not in the VDS.")
    }

    annotateSamples(s => map.getOrElse(s, null), finalType, inserter)
  }

  def annotateSamplesVDS(other: VariantSampleMatrix[_],
    root: Option[String] = None,
    code: Option[String] = None): VariantSampleMatrix[T] = {

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Annotation) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "sa" -> (0, saSignature),
          "vds" -> (1, other.saSignature)))
        Annotation.buildInserter(annotationExpr, saSignature, ec, Annotation.SAMPLE_HEAD)
      } else
        insertSA(other.saSignature, Parser.parseAnnotationRoot(annotationExpr, Annotation.SAMPLE_HEAD))

    val m = other.sampleIdsAndAnnotations.toMap
    annotateSamples(s => m.getOrElse(s, null), finalType, inserter)
  }

  def annotateSamples(annotation: (String) => Annotation, newSignature: Type, inserter: Inserter): VariantSampleMatrix[T] = {
    val newAnnotations = sampleIds.zipWithIndex.map { case (id, i) =>
      val sa = sampleAnnotations(i)
      val newAnnotation = inserter(sa, annotation(id))
      newSignature.typeCheck(newAnnotation)
      newAnnotation
    }

    copy(sampleAnnotations = newAnnotations, saSignature = newSignature)
  }

  def annotateVariants(otherRDD: OrderedRDD[Locus, Variant, Annotation], signature: Type,
    code: String): VariantSampleMatrix[T] = {
    val (newSignature, ins) = insertVA(signature, Parser.parseAnnotationRoot(code, Annotation.VARIANT_HEAD))
    annotateVariants(otherRDD, newSignature, ins)
  }

  def annotateVariantsBED(path: String, root: String, all: Boolean = false): VariantSampleMatrix[T] = {
    val annotationPath = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)
    BedAnnotator(path, hc.hadoopConf) match {
      case (is, None) =>
        annotateIntervals(is, annotationPath)

      case (is, Some((t, m))) =>
        annotateIntervals(is, t, m, all = all, annotationPath)
    }
  }

  def annotateVariantsExpr(expr: String): VariantSampleMatrix[T] = {
    val localGlobalAnnotation = globalAnnotation

    val ec = variantEC
    val (paths, types, f) = Parser.parseAnnotationExprs(expr, ec, Some(Annotation.VARIANT_HEAD))

    val inserterBuilder = mutable.ArrayBuilder.make[Inserter]
    val finalType = (paths, types).zipped.foldLeft(vaSignature) { case (vas, (ids, signature)) =>
      val (s, i) = vas.insert(signature, ids)
      inserterBuilder += i
      s
    }
    val inserters = inserterBuilder.result()

    val aggregateOption = Aggregators.buildVariantAggregations(this, ec)

    mapAnnotations { case (v, va, gs) =>
      ec.setAll(localGlobalAnnotation, v, va)

      aggregateOption.foreach(f => f(v, va, gs))
      f().zip(inserters)
        .foldLeft(va) { case (va, (v, inserter)) =>
          inserter(va, v)
        }
    }.copy(vaSignature = finalType)
  }

  def annotateVariantsIntervals(path: String, root: String, all: Boolean = false): VariantSampleMatrix[T] = {
    val annotationPath = Parser.parseAnnotationRoot(root, Annotation.VARIANT_HEAD)

    IntervalListAnnotator(path, hc.hadoopConf) match {
      case (is, Some((m, t))) =>
        annotateIntervals(is, m, t, all = all, annotationPath)

      case (is, None) =>
        annotateIntervals(is, annotationPath)
    }
  }

  def annotateVariantsKeyTable(kt: KeyTable, code: String): VariantSampleMatrix[T] = {
    val ktKeyTypes = kt.keyFields.map(_.typ)

    if (ktKeyTypes.length != 1 || ktKeyTypes(0) != TVariant)
      fatal(s"Key signature of KeyTable must be 1 field with type `Variant'. Found `${ ktKeyTypes.mkString(", ") }'")

    val ktSig = kt.signature

    val inserterEc = EvalContext(Map("va" -> (0, vaSignature), "table" -> (1, ktSig)))

    val (finalType, inserter) =
      buildInserter(code, vaSignature, inserterEc, Annotation.VARIANT_HEAD)

    val keyedRDD = kt.keyedRDD().map { case (k: Row, a) => (k(0).asInstanceOf[Variant], a) }

    val ordRdd = OrderedRDD(keyedRDD, None, None)

    annotateVariants(ordRdd, finalType, inserter)
  }

  def annotateVariantsKeyTable(kt: KeyTable, vdsKey: java.util.ArrayList[String], code: String): VariantSampleMatrix[T] =
    annotateVariantsKeyTable(kt, vdsKey.asScala, code)

  def annotateVariantsKeyTable(kt: KeyTable, vdsKey: Seq[String], code: String): VariantSampleMatrix[T] = {
    val vdsKeyEc = EvalContext(Map("v" -> (0, TVariant), "va" -> (1, vaSignature)))

    val (vdsKeyType, vdsKeyFs) = vdsKey.map(Parser.parseExpr(_, vdsKeyEc)).unzip

    val keyTypes = kt.keyFields.map(_.typ)
    if (!(keyTypes sameElements vdsKeyType))
      fatal(s"Key signature of KeyTable, `${ keyTypes.mkString(", ") }', must match type of computed key, `${ vdsKeyType.mkString(", ") }'.")

    val ktSig = kt.signature

    val inserterEc = EvalContext(Map("va" -> (0, vaSignature), "table" -> (1, ktSig)))

    val (finalType, inserter) =
      buildInserter(code, vaSignature, inserterEc, Annotation.VARIANT_HEAD)

    val ktRdd = kt.keyedRDD()

    val thisRdd = rdd.map { case (v, (va, gs)) =>
      vdsKeyEc.setAll(v, va)
      (Annotation.fromSeq(vdsKeyFs.map(_ ())), (v, va))
    }

    val variantKeyedRdd = ktRdd.join(thisRdd)
      .map { case (_, (table, (v, va))) => (v, inserter(va, table)) }

    val ordRdd = OrderedRDD(variantKeyedRdd, None, None)

    val newRdd = rdd.orderedLeftJoinDistinct(ordRdd)
      .mapValues { case ((va, gs), optVa) => (optVa.getOrElse(va), gs) }
      .asOrderedRDD

    copy(rdd = newRdd, vaSignature = finalType)
  }

  def annotateVariantsLoci(path: String, locusExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {
    annotateVariantsLociAll(List(path), locusExpr, root, code, config)
  }

  def annotateVariantsLociAll(paths: Seq[String], locusExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {
    val files = hc.hadoopConf.globAll(paths)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val (struct, locusRDD) = TextTableReader.read(sparkContext)(files, config, nPartitions)

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Annotation) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "va" -> (0, vaSignature),
          "table" -> (1, struct)))
        Annotation.buildInserter(annotationExpr, vaSignature, ec, Annotation.VARIANT_HEAD)
      } else insertVA(struct, Parser.parseAnnotationRoot(annotationExpr, Annotation.VARIANT_HEAD))

    val locusQuery = struct.parseInStructScope[Locus](locusExpr)


    import is.hail.variant.LocusImplicits.orderedKey
    val lociRDD = locusRDD.map {
      _.map { a =>
        (locusQuery(a), a)
      }.value
    }
      .filter { case (l, a) => l != null }
      .toOrderedRDD(rdd.orderedPartitioner.mapMonotonic)

    annotateLoci(lociRDD, finalType, inserter)
  }

  def annotateLoci(lociRDD: OrderedRDD[Locus, Locus, Annotation], newSignature: Type, inserter: Inserter): VariantSampleMatrix[T] = {

    import LocusImplicits.orderedKey

    val newRDD = rdd
      .mapMonotonic(OrderedKeyFunction(_.locus), { case (v, vags) => (v, vags) })
      .orderedLeftJoinDistinct(lociRDD)
      .map { case (l, ((v, (va, gs)), annotation)) => (v, (inserter(va, annotation.orNull), gs)) }

    // we safely use the non-shuffling apply method of OrderedRDD because orderedLeftJoinDistinct preserves the
    // (Variant) ordering of the left RDD
    val orderedRDD = OrderedRDD(newRDD, rdd.orderedPartitioner)
    copy(rdd = orderedRDD, vaSignature = newSignature)
  }

  def nPartitions: Int = rdd.partitions.length

  def annotateVariantsTable(path: String, variantExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {
    annotateVariantsTables(List(path), variantExpr, root, code, config)
  }

  def annotateVariantsTables(paths: Seq[String], variantExpr: String,
    root: Option[String] = None, code: Option[String] = None,
    config: TextTableConfiguration = TextTableConfiguration()): VariantSampleMatrix[T] = {
    val files = hc.hadoopConf.globAll(paths)
    if (files.isEmpty)
      fatal("Arguments referred to no files")

    val (struct, variantRDD) = TextTableReader.read(sparkContext)(files, config, nPartitions)

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Annotation) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "va" -> (0, vaSignature),
          "table" -> (1, struct)))
        Annotation.buildInserter(annotationExpr, vaSignature, ec, Annotation.VARIANT_HEAD)
      } else
        insertVA(struct, Parser.parseAnnotationRoot(annotationExpr, Annotation.VARIANT_HEAD))

    val variantQuery = struct.parseInStructScope[Variant](variantExpr)

    val keyedRDD = variantRDD.map {
      _.map { a =>
        (variantQuery(a), a)
      }.value
    }
      .filter { case (v, a) => v != null }
      .toOrderedRDD(rdd.orderedPartitioner)

    annotateVariants(keyedRDD, finalType, inserter)
  }

  def annotateVariants(otherRDD: OrderedRDD[Locus, Variant, Annotation], newSignature: Type,
    inserter: Inserter): VariantSampleMatrix[T] = {
    val newRDD = rdd.orderedLeftJoinDistinct(otherRDD)
      .mapValues { case ((va, gs), annotation) =>
        (inserter(va, annotation.orNull), gs)
      }.asOrderedRDD
    copy(rdd = newRDD, vaSignature = newSignature)
  }

  def annotateVariantsVDS(other: VariantSampleMatrix[_],
    root: Option[String] = None, code: Option[String] = None): VariantSampleMatrix[T] = {

    val (isCode, annotationExpr) = (root, code) match {
      case (Some(r), None) => (false, r)
      case (None, Some(c)) => (true, c)
      case _ => fatal("this module requires one of `root' or 'code', but not both")
    }

    val (finalType, inserter): (Type, (Annotation, Annotation) => Annotation) =
      if (isCode) {
        val ec = EvalContext(Map(
          "va" -> (0, vaSignature),
          "vds" -> (1, other.vaSignature)))
        Annotation.buildInserter(annotationExpr, vaSignature, ec, Annotation.VARIANT_HEAD)
      } else insertVA(other.vaSignature, Parser.parseAnnotationRoot(annotationExpr, Annotation.VARIANT_HEAD))

    annotateVariants(other.variantsAndAnnotations, finalType, inserter)
  }

  def countVariants(): Long = variants.count()

  def variants: RDD[Variant] = rdd.keys

  def deduplicate(): VariantSampleMatrix[T] = {
    DuplicateReport.initialize()

    val acc = DuplicateReport.accumulator
    copy(rdd = rdd.mapPartitions({ it =>
      new SortedDistinctPairIterator(it, (v: Variant) => acc += v)
    }, preservesPartitioning = true).asOrderedRDD)
  }

  def deleteGlobal(args: String*): (Type, Deleter) = deleteGlobal(args.toList)

  def deleteGlobal(path: List[String]): (Type, Deleter) = globalSignature.delete(path)

  def deleteSA(args: String*): (Type, Deleter) = deleteSA(args.toList)

  def deleteSA(path: List[String]): (Type, Deleter) = saSignature.delete(path)

  def deleteVA(args: String*): (Type, Deleter) = deleteVA(args.toList)

  def deleteVA(path: List[String]): (Type, Deleter) = vaSignature.delete(path)

  def downsampleVariants(keep: Long): VariantSampleMatrix[T] = {
    sampleVariants(keep.toDouble / countVariants())
  }

  def dropSamples(): VariantSampleMatrix[T] =
    copy(sampleIds = IndexedSeq.empty[String],
      sampleAnnotations = IndexedSeq.empty[Annotation],
      rdd = rdd.mapValues { case (va, gs) => (va, Iterable.empty[T]) }
        .asOrderedRDD)

  def dropVariants(): VariantSampleMatrix[T] = copy(rdd = OrderedRDD.empty(sparkContext))

  def expand(): RDD[(Variant, String, T)] =
    mapWithKeys[(Variant, String, T)]((v, s, g) => (v, s, g))

  def expandWithAll(): RDD[(Variant, Annotation, String, Annotation, T)] =
    mapWithAll[(Variant, Annotation, String, Annotation, T)]((v, va, s, sa, g) => (v, va, s, sa, g))

  def mapWithAll[U](f: (Variant, Annotation, String, Annotation, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd
      .flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith2[Annotation, T, U](localSampleAnnotationsBc.value, gs, {
          case (s, sa, g) => f(v, va, s, sa, g)
        })
      }
  }

  def exportSamples(path: String, expr: String, typeFile: Boolean = false) {
    val localGlobalAnnotation = globalAnnotation

    val ec = sampleEC

    val (names, types, f) = Parser.parseExportExprs(expr, ec)
    val hadoopConf = hc.hadoopConf
    if (typeFile) {
      hadoopConf.delete(path + ".types", recursive = false)
      val typeInfo = names
        .getOrElse(types.indices.map(i => s"_$i").toArray)
        .zip(types)
      exportTypes(path + ".types", hadoopConf, typeInfo)
    }

    val sampleAggregationOption = Aggregators.buildSampleAggregations(this, ec)

    hadoopConf.delete(path, recursive = true)

    val sb = new StringBuilder()
    val lines = for ((s, sa) <- sampleIdsAndAnnotations) yield {
      sampleAggregationOption.foreach(f => f.apply(s))
      sb.clear()
      ec.setAll(localGlobalAnnotation, s, sa)
      f().foreachBetween(x => sb.append(x))(sb += '\t')
      sb.result()
    }

    hadoopConf.writeTable(path, lines, names.map(_.mkString("\t")))
  }

  def exportVariants(path: String, expr: String, typeFile: Boolean = false) {
    val vas = vaSignature
    val hConf = hc.hadoopConf

    val localGlobalAnnotations = globalAnnotation
    val ec = variantEC

    val (names, types, f) = Parser.parseExportExprs(expr, ec)

    val hadoopConf = hc.hadoopConf
    if (typeFile) {
      hadoopConf.delete(path + ".types", recursive = false)
      val typeInfo = names
        .getOrElse(types.indices.map(i => s"_$i").toArray)
        .zip(types)
      exportTypes(path + ".types", hadoopConf, typeInfo)
    }

    val variantAggregations = Aggregators.buildVariantAggregations(this, ec)

    hadoopConf.delete(path, recursive = true)

    rdd
      .mapPartitions { it =>
        val sb = new StringBuilder()
        it.map { case (v, (va, gs)) =>
          variantAggregations.foreach { f => f(v, va, gs) }
          ec.setAll(localGlobalAnnotations, v, va)
          sb.clear()
          f().foreachBetween(x => sb.append(x))(sb += '\t')
          sb.result()
        }
      }.writeTable(path, hc.tmpDir, names.map(_.mkString("\t")))
  }

  def filterIntervals(path: String, keep: Boolean): VariantSampleMatrix[T] = {
    filterIntervals(IntervalListAnnotator.read(path, sparkContext.hadoopConfiguration, prune = true), keep)
  }

  def filterIntervals(iList: IntervalTree[Locus], keep: Boolean): VariantSampleMatrix[T] = {
    if (keep)
      copy(rdd = rdd.filterIntervals(iList))
    else {
      val iListBc = sparkContext.broadcast(iList)
      filterVariants { (v, va, gs) => !iListBc.value.contains(v.locus)
      }
    }
  }

  def filterVariants(p: (Variant, Annotation, Iterable[T]) => Boolean): VariantSampleMatrix[T] =
    copy(rdd = rdd.filter { case (v, (va, gs)) => p(v, va, gs) }.asOrderedRDD)

  // FIXME see if we can remove broadcasts elsewhere in the code
  def filterSamples(p: (String, Annotation) => Boolean): VariantSampleMatrix[T] = {
    val mask = sampleIdsAndAnnotations.map { case (s, sa) => p(s, sa) }
    val maskBc = sparkContext.broadcast(mask)
    val localtct = tct
    copy[T](sampleIds = sampleIds.zipWithIndex
      .filter { case (s, i) => mask(i) }
      .map(_._1),
      sampleAnnotations = sampleAnnotations.zipWithIndex
        .filter { case (sa, i) => mask(i) }
        .map(_._1),
      rdd = rdd.mapValues { case (va, gs) =>
        (va, gs.lazyFilterWith(maskBc.value, (g: T, m: Boolean) => m))
      }.asOrderedRDD)
  }

  /**
    * Filter samples using the Hail expression language.
    *
    * @param filterExpr Filter expression involving `s' (sample) and `sa' (sample annotations)
    * @param keep keep where filterExpr evaluates to true
    */
  def filterSamplesExpr(filterExpr: String, keep: Boolean = true): VariantSampleMatrix[T] = {
    val localGlobalAnnotation = globalAnnotation

    val sas = saSignature

    val ec = sampleEC

    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](filterExpr, ec)

    val sampleAggregationOption = Aggregators.buildSampleAggregations(this, ec)

    val localKeep = keep
//    val sampleIds = vsampleIds
    val p = (s: String, sa: Annotation) => {
      sampleAggregationOption.foreach(f => f.apply(s))
      ec.setAll(localGlobalAnnotation, s, sa)
      Filter.boxedKeepThis(f(), localKeep)
    }

    filterSamples(p)
  }

  /**
    * Filter samples using a text file containing sample IDs
    * @param path path to sample list file
    * @param keep keep listed samples
    */
  def filterSamplesList(path: String, keep: Boolean = true): VariantSampleMatrix[T] = {
    val samples = hc.hadoopConf.readFile(path) { reader =>
      Source.fromInputStream(reader)
        .getLines()
        .filter(line => !line.isEmpty)
        .toSet
    }
    val p = (s: String, sa: Annotation) => Filter.keepThis(samples.contains(s), keep)

    filterSamples(p)
  }

  /**
    * Filter variants using the Hail expression language.
    * @param filterExpr filter expression
    * @param keep keep variants where filterExpr evaluates to true
    * @return
    */
  def filterVariantsExpr(filterExpr: String, keep: Boolean = true): VariantSampleMatrix[T] = {
    val localGlobalAnnotation = globalAnnotation
    val ec = variantEC

    val f: () => java.lang.Boolean = Parser.parseTypedExpr[java.lang.Boolean](filterExpr, ec)

    val aggregatorOption = Aggregators.buildVariantAggregations(this, ec)

    val localKeep = keep
    val p = (v: Variant, va: Annotation, gs: Iterable[T]) => {
      aggregatorOption.foreach(f => f(v, va, gs))

      ec.setAll(localGlobalAnnotation, v, va)
      Filter.boxedKeepThis(f(), localKeep)
    }

    filterVariants(p)
  }

  def filterVariantsList(input: String, keep: Boolean): VariantSampleMatrix[T] = {
    copy(
      rdd = rdd
        .orderedLeftJoinDistinct(Variant.variantUnitRdd(sparkContext, input).toOrderedRDD)
        .mapPartitions({ it =>
          it.flatMap { case (v, ((va, gs), o)) =>
            o match {
              case Some(_) =>
                if (keep) Some((v, (va, gs))) else None
              case None =>
                if (keep) None else Some((v, (va, gs)))
            }
          }
        }, preservesPartitioning = true)
        .asOrderedRDD
    )
  }

  def sparkContext: SparkContext = hc.sc

  def flatMap[U](f: T => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] =
    flatMapWithKeys((v, s, g) => f(g))

  def flatMapWithKeys[U](f: (Variant, String, T) => TraversableOnce[U])(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc

    rdd
      .flatMap { case (v, (va, gs)) => localSampleIdsBc.value.lazyFlatMapWith(gs,
        (s: String, g: T) => f(v, s, g))
      }
  }

  /**
    * The function {@code f} must be monotonic with respect to the ordering on {@code Locus}
    */
  def flatMapVariants(f: (Variant, Annotation, Iterable[T]) => TraversableOnce[(Variant, (Annotation, Iterable[T]))]): VariantSampleMatrix[T] =
    copy(rdd = rdd.flatMapMonotonic[(Annotation, Iterable[T])] { case (v, (va, gs)) => f(v, va, gs) })

  def foldBySample(zeroValue: T)(combOp: (T, T) => T): RDD[(String, T)] = {

    val localtct = tct

    val serializer = SparkEnv.get.serializer.newInstance()
    val zeroBuffer = serializer.serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    val localSampleIdsBc = sampleIdsBc

    rdd
      .mapPartitions { (it: Iterator[(Variant, (Annotation, Iterable[T]))]) =>
        val serializer = SparkEnv.get.serializer.newInstance()

        def copyZeroValue() = serializer.deserialize[T](ByteBuffer.wrap(zeroArray))(localtct)

        val arrayZeroValue = Array.fill[T](localSampleIdsBc.value.length)(copyZeroValue())
        localSampleIdsBc.value.iterator
          .zip(it.foldLeft(arrayZeroValue) { case (acc, (v, (va, gs))) =>
            for ((g, i) <- gs.iterator.zipWithIndex)
              acc(i) = combOp(acc(i), g)
            acc
          }.iterator)
      }.foldByKey(zeroValue)(combOp)
  }

  def foldByVariant(zeroValue: T)(combOp: (T, T) => T): RDD[(Variant, T)] =
    rdd.mapValues { case (va, gs) => gs.foldLeft(zeroValue)((acc, g) => combOp(acc, g)) }

  def hadoopConf: hadoop.conf.Configuration = hc.hadoopConf

  def insertGlobal(sig: Type, args: String*): (Type, Inserter) = insertGlobal(sig, args.toList)

  def insertSA(sig: Type, args: String*): (Type, Inserter) = insertSA(sig, args.toList)

  def insertSA(sig: Type, path: List[String]): (Type, Inserter) = saSignature.insert(sig, path)

  def insertVA(sig: Type, args: String*): (Type, Inserter) = insertVA(sig, args.toList)

  def insertVA(sig: Type, path: List[String]): (Type, Inserter) = {
    vaSignature.insert(sig, path)
  }

  def isDosage: Boolean = metadata.isDosage

  /**
    *
    * @param right right-hand dataset with which to join
    */
  def join(right: VariantSampleMatrix[T]): VariantSampleMatrix[T] = {
    if (wasSplit != right.wasSplit) {
      warn(
        s"""cannot join split and unsplit datasets
           |  left was split: ${ wasSplit }
           |  light was split: ${ right.wasSplit }""".stripMargin)
    }

    if (genotypeSignature != right.genotypeSignature) {
      fatal(
        s"""cannot join datasets with different genotype schemata
           |  left sample schema: @1
           |  right sample schema: @2""".stripMargin,
        genotypeSignature.toPrettyString(compact = true, printAttrs = true),
        right.genotypeSignature.toPrettyString(compact = true, printAttrs = true))
    }

    if (saSignature != right.saSignature) {
      fatal(
        s"""cannot join datasets with different sample schemata
           |  left sample schema: @1
           |  right sample schema: @2""".stripMargin,
        saSignature.toPrettyString(compact = true, printAttrs = true),
        right.saSignature.toPrettyString(compact = true, printAttrs = true))
    }

    val newSampleIds = sampleIds ++ right.sampleIds
    val duplicates = newSampleIds.duplicates()
    if (duplicates.nonEmpty)
      fatal("duplicate sample IDs: @1", duplicates)

    val joined = rdd.orderedInnerJoinDistinct(right.rdd)
      .mapValues { case ((lva, lgs), (rva, rgs)) =>
        (lva, lgs ++ rgs)
      }.asOrderedRDD

    copy(
      sampleIds = newSampleIds,
      sampleAnnotations = sampleAnnotations ++ right.sampleAnnotations,
      rdd = joined)
  }

  def makeKT(variantCondition: String, genotypeCondition: String, keyNames: Array[String]): KeyTable = {
    val vSymTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vaSignature))
    val vEC = EvalContext(vSymTab)
    val vA = vEC.a

    val (vNames, vTypes, vf) = Parser.parseNamedExprs(variantCondition, vEC)

    val gSymTab = Map(
      "v" -> (0, TVariant),
      "va" -> (1, vaSignature),
      "s" -> (2, TSample),
      "sa" -> (3, saSignature),
      "g" -> (4, genotypeSignature))
    val gEC = EvalContext(gSymTab)
    val gA = gEC.a

    val (gNames, gTypes, gf) = Parser.parseNamedExprs(genotypeCondition, gEC)

    val sig = TStruct(((vNames, vTypes).zipped ++
      sampleIds.flatMap { s =>
        (gNames, gTypes).zipped.map { case (n, t) =>
          (if (n.isEmpty)
            s
          else
            s + "." + n, t)
        }
      }).toSeq: _*)

    val localNSamples = nSamples
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    KeyTable(hc,
      rdd.mapPartitions { it =>
        val n = vNames.length + gNames.length * localNSamples

        it.map { case (v, (va, gs)) =>
          val a = new Array[Any](n)

          var j = 0
          vEC.setAll(v, va)
          vf().foreach { x =>
            a(j) = x
            j += 1
          }

          gs.iterator.zipWithIndex.foreach { case (g, i) =>
            val s = localSampleIdsBc.value(i)
            val sa = localSampleAnnotationsBc.value(i)
            gEC.setAll(v, va, s, sa, g)
            gf().foreach { x =>
              a(j) = x
              j += 1
            }
          }

          assert(j == n)
          Row.fromSeq(a): Annotation
        }
      },
      sig,
      keyNames)
  }

  def map[U](f: T => U)(implicit uct: ClassTag[U]): RDD[U] =
    mapWithKeys((v, s, g) => f(g))

  def mapWithKeys[U](f: (Variant, String, T) => U)(implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc

    rdd
      .flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith[T, U](gs,
          (s, g) => f(v, s, g))
      }
  }

  def mapAnnotations(f: (Variant, Annotation, Iterable[T]) => Annotation): VariantSampleMatrix[T] =
    copy[T](rdd = rdd.mapValuesWithKey { case (v, (va, gs)) => (f(v, va, gs), gs) }.asOrderedRDD)

  def mapAnnotationsWithAggregate[U](zeroValue: U, newVAS: Type)(
    seqOp: (U, Variant, Annotation, String, Annotation, T) => U,
    combOp: (U, U) => U,
    mapOp: (Annotation, U) => Annotation)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[T] = {

    // Serialize the zero value to a byte array so that we can apply a new clone of it on each key
    val zeroBuffer = SparkEnv.get.serializer.newInstance().serialize(zeroValue)
    val zeroArray = new Array[Byte](zeroBuffer.limit)
    zeroBuffer.get(zeroArray)

    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    copy(vaSignature = newVAS,
      rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
        val serializer = SparkEnv.get.serializer.newInstance()
        val zeroValue = serializer.deserialize[U](ByteBuffer.wrap(zeroArray))

        (mapOp(va, gs.iterator
          .zip(localSampleIdsBc.value.iterator
            .zip(localSampleAnnotationsBc.value.iterator)).foldLeft(zeroValue) {
          case (acc, (g, (s, sa))) =>
            seqOp(acc, v, va, s, sa, g)
        }), gs)
      }.asOrderedRDD)
  }

  def mapPartitionsWithAll[U](f: Iterator[(Variant, Annotation, String, Annotation, T)] => Iterator[U])
    (implicit uct: ClassTag[U]): RDD[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    rdd.mapPartitions { it =>
      f(it.flatMap { case (v, (va, gs)) =>
        localSampleIdsBc.value.lazyMapWith2[Annotation, T, (Variant, Annotation, String, Annotation, T)](
          localSampleAnnotationsBc.value, gs, { case (s, sa, g) => (v, va, s, sa, g) })
      })
    }
  }

  def mapValues[U](f: (T) => U)(implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithAll((v, va, s, sa, g) => f(g))
  }

  def mapValuesWithKeys[U](f: (Variant, String, T) => U)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    mapValuesWithAll((v, va, s, sa, g) => f(v, s, g))
  }

  def mapValuesWithAll[U](f: (Variant, Annotation, String, Annotation, T) => U)
    (implicit uct: ClassTag[U]): VariantSampleMatrix[U] = {
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc
    copy(rdd = rdd.mapValuesWithKey { case (v, (va, gs)) =>
      (va, localSampleIdsBc.value.lazyMapWith2[Annotation, T, U](localSampleAnnotationsBc.value, gs, {
        case (s, sa, g) => f(v, va, s, sa, g)
      }))
    }.asOrderedRDD)
  }

  def minrep(maxShift: Int = 100): VariantSampleMatrix[T] = {
    require(maxShift > 0, s"invalid value for maxShift: $maxShift. Parameter must be a positive integer.")
    val minrepped = rdd.map {
      case (v, (va, gs)) =>
        (v.minrep, (va, gs))
    }
    copy(rdd = minrepped.smartShuffleAndSort(rdd.orderedPartitioner, maxShift))
  }

  def queryGenotypes(expr: String): (Annotation, Type) = {
    val qv = queryGenotypes(Array(expr))
    assert(qv.length == 1)
    qv.head
  }

  def queryGenotypes(exprs: Array[String]): Array[(Annotation, Type)] = {
    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "g" -> (1, genotypeSignature),
      "v" -> (2, TVariant),
      "va" -> (3, vaSignature),
      "s" -> (4, TSample),
      "sa" -> (5, saSignature))
    val ec = EvalContext(Map(
      "global" -> (0, globalSignature),
      "gs" -> (1, TAggregable(genotypeSignature, aggregationST))))

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val localGlobalAnnotation = globalAnnotation
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[T](ec, { case (ec, g) =>
      ec.set(1, g)
    })

    val globalBc = sparkContext.broadcast(globalAnnotation)
    val localSampleIdsBc = sampleIdsBc
    val localSampleAnnotationsBc = sampleAnnotationsBc

    val result = rdd.mapPartitions { it =>
      val zv = zVal.map(_.copy())
      ec.set(0, globalBc.value)
      it.foreach { case (v, (va, gs)) =>
        var i = 0
        ec.set(2, v)
        ec.set(3, va)
        gs.foreach { g =>
          ec.set(4, localSampleIdsBc.value(i))
          ec.set(5, localSampleAnnotationsBc.value(i))
          seqOp(zv, g)
          i += 1
        }
      }
      Iterator(zv)
    }.fold(zVal.map(_.copy()))(combOp)
    resOp(result)

    ec.set(0, localGlobalAnnotation)
    ts.map { case (t, f) => (f(), t) }
  }

  def queryGlobal(path: String): (Type, Annotation) = {
    val st = Map(Annotation.GLOBAL_HEAD -> (0, globalSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(path, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2(globalAnnotation))
  }

  def querySA(code: String): (Type, Querier) = {

    val st = Map(Annotation.SAMPLE_HEAD -> (0, saSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def querySamples(expr: String): (Annotation, Type) = {
    val qs = querySamples(Array(expr))
    assert(qs.length == 1)
    qs.head
  }

  def querySamples(exprs: Array[String]): Array[(Annotation, Type)] = {
    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "s" -> (1, TSample),
      "sa" -> (2, saSignature))
    val ec = EvalContext(Map(
      "global" -> (0, globalSignature),
      "samples" -> (1, TAggregable(TSample, aggregationST))))

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val localGlobalAnnotation = globalAnnotation
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[(String, Annotation)](ec, { case (ec, (s, sa)) =>
      ec.setAll(localGlobalAnnotation, s, sa)
    })

    val results = sampleIdsAndAnnotations
      .aggregate(zVal)(seqOp, combOp)
    resOp(results)
    ec.set(0, localGlobalAnnotation)

    ts.map { case (t, f) => (f(), t) }.toArray
  }

  def queryVA(code: String): (Type, Querier) = {

    val st = Map(Annotation.VARIANT_HEAD -> (0, vaSignature))
    val ec = EvalContext(st)
    val a = ec.a

    val (t, f) = Parser.parseExpr(code, ec)

    val f2: Annotation => Any = { annotation =>
      a(0) = annotation
      f()
    }

    (t, f2)
  }

  def vaSignature: Type = metadata.vaSignature

  def queryVariants(expr: String): (Annotation, Type) = {
    val qv = queryVariants(Array(expr))
    assert(qv.length == 1)
    qv.head
  }

  def queryVariants(exprs: Array[String]): Array[(Annotation, Type)] = {

    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vaSignature))
    val ec = EvalContext(Map(
      "global" -> (0, globalSignature),
      "variants" -> (1, TAggregable(TVariant, aggregationST))))

    val ts = exprs.map(e => Parser.parseExpr(e, ec))

    val localGlobalAnnotation = globalAnnotation
    val (zVal, seqOp, combOp, resOp) = Aggregators.makeFunctions[(Variant, Annotation)](ec, { case (ec, (v, va)) =>
      ec.setAll(localGlobalAnnotation, v, va)
    })

    val result = variantsAndAnnotations
      .treeAggregate(zVal)(seqOp, combOp, depth = treeAggDepth(hc, nPartitions))
    resOp(result)

    ec.setAll(localGlobalAnnotation)
    ts.map { case (t, f) => (f(), t) }.toArray
  }

  def renameSamples(mapping: java.util.Map[String, String]): VariantSampleMatrix[T] =
    renameSamples(mapping.asScala.toMap)

  def renameSamples(mapping: Map[String, String]): VariantSampleMatrix[T] = {
    val newSamples = mutable.Set.empty[String]
    val newSampleIds = sampleIds
      .map { s =>
        val news = mapping.getOrElse(s, s)
        if (newSamples.contains(news))
          fatal(s"duplicate sample ID `$news' after rename")
        newSamples += news
        news
      }
    copy(sampleIds = newSampleIds)
  }

  def same(that: VariantSampleMatrix[T], tolerance: Double = utils.defaultTolerance): Boolean = {
    var metadataSame = true
    if (vaSignature != that.vaSignature) {
      metadataSame = false
      println(
        s"""different va signature:
           |  left:  ${ vaSignature.toPrettyString(compact = true) }
           |  right: ${ that.vaSignature.toPrettyString(compact = true) }""".stripMargin)
    }
    if (saSignature != that.saSignature) {
      metadataSame = false
      println(
        s"""different sa signature:
           |  left:  ${ saSignature.toPrettyString(compact = true) }
           |  right: ${ that.saSignature.toPrettyString(compact = true) }""".stripMargin)
    }
    if (globalSignature != that.globalSignature) {
      metadataSame = false
      println(
        s"""different global signature:
           |  left:  ${ globalSignature.toPrettyString(compact = true) }
           |  right: ${ that.globalSignature.toPrettyString(compact = true) }""".stripMargin)
    }
    if (sampleIds != that.sampleIds) {
      metadataSame = false
      println(
        s"""different sample ids:
           |  left:  $sampleIds
           |  right: ${ that.sampleIds }""".stripMargin)
    }
    if (!sampleAnnotationsSimilar(that, tolerance)) {
      metadataSame = false
      println(
        s"""different sample annotations:
           |  left:  $sampleAnnotations
           |  right: ${ that.sampleAnnotations }""".stripMargin)
    }
    if (sampleIds != that.sampleIds) {
      metadataSame = false
      println(
        s"""different global annotation:
           |  left:  $globalAnnotation
           |  right: ${ that.globalAnnotation }""".stripMargin)
    }
    if (wasSplit != that.wasSplit) {
      metadataSame = false
      println(
        s"""different was split:
           |  left:  $wasSplit
           |  right: ${ that.wasSplit }""".stripMargin)
    }
    if (!metadataSame)
      println("metadata were not the same")
    val vaSignatureBc = sparkContext.broadcast(vaSignature)
    val gSignatureBc = sparkContext.broadcast(genotypeSignature)
    var printed = false
    metadataSame &&
      rdd
        .fullOuterJoin(that.rdd)
        .forall {
          case (v, (Some((va1, it1)), Some((va2, it2)))) =>
            val annotationsSame = vaSignatureBc.value.valuesSimilar(va1, va2, tolerance)
            if (!annotationsSame && !printed) {
              println(
                s"""at variant `$v', annotations were not the same:
                   |  $va1
                   |  $va2
                 """.stripMargin)
              printed = true
            }
            val genotypesSame = (it1, it2).zipped.forall { case (g1, g2) =>
              val gSame = gSignatureBc.value.valuesSimilar(g1, g2, tolerance)
              if (!gSame)
                println(s"genotypes $g1, $g2 were not the same")
              gSame
            }
            annotationsSame && genotypesSame
          case (v, _) =>
            println(s"Found unmatched variant $v")
            false
        }
  }

  def sampleEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "s" -> (1, TSample),
      "va" -> (2, saSignature),
      "g" -> (3, genotypeSignature),
      "v" -> (4, TVariant),
      "va" -> (5, vaSignature))
    EvalContext(Map(
      "global" -> (0, globalSignature),
      "s" -> (1, TSample),
      "sa" -> (2, saSignature),
      "gs" -> (3, TAggregable(genotypeSignature, aggregationST))))
  }

  def sampleIds: IndexedSeq[String] = metadata.sampleIds

  def saSignature: Type = metadata.saSignature

  def sampleAnnotations: IndexedSeq[Annotation] = metadata.sampleAnnotations

  def wasSplit: Boolean = metadata.wasSplit

  def sampleAnnotationsSimilar(that: VariantSampleMatrix[T], tolerance: Double = utils.defaultTolerance): Boolean = {
    require(saSignature == that.saSignature)
    sampleAnnotations.zip(that.sampleAnnotations)
      .forall { case (s1, s2) => saSignature.valuesSimilar(s1, s2, tolerance) }
  }

  def sampleVariants(fraction: Double): VariantSampleMatrix[T] =
    copy(rdd = rdd.sample(withReplacement = false, fraction, 1).asOrderedRDD)

  def copy[U](rdd: OrderedRDD[Locus, Variant, (Annotation, Iterable[U])] = rdd,
    sampleIds: IndexedSeq[String] = sampleIds,
    sampleAnnotations: IndexedSeq[Annotation] = sampleAnnotations,
    globalAnnotation: Annotation = globalAnnotation,
    saSignature: Type = saSignature,
    vaSignature: Type = vaSignature,
    globalSignature: Type = globalSignature,
    genotypeSignature: Type = genotypeSignature,
    wasSplit: Boolean = wasSplit,
    isDosage: Boolean = isDosage,
    isGenericGenotype: Boolean = isGenericGenotype)
    (implicit tct: ClassTag[U]): VariantSampleMatrix[U] =
    new VariantSampleMatrix[U](hc,
      VariantMetadata(sampleIds, sampleAnnotations, globalAnnotation,
        saSignature, vaSignature, globalSignature, genotypeSignature, wasSplit, isDosage, isGenericGenotype), rdd)

  def samplesKT(): KeyTable = {
    KeyTable(hc, sparkContext.parallelize(sampleIdsAndAnnotations)
      .map { case (s, sa) =>
        Annotation(s, sa)
      },
      TStruct(
        "s" -> TSample,
        "sa" -> saSignature),
      Array("s"))
  }

  def storageLevel: String = rdd.getStorageLevel.toReadableString()

  def setVAattribute(path: String, key: String, value: String): VariantSampleMatrix[T] = {
    setVAattributes(path, Map(key -> value))
  }

  def setVAattribute(path: List[String], key: String, value: String): VariantSampleMatrix[T] = {
    setVAattributes(path, Map(key -> value))
  }

  def setVAattributes(path: String, kv: Map[String, String]): VariantSampleMatrix[T] = {
    setVAattributes(Parser.parseAnnotationRoot(path, Annotation.VARIANT_HEAD), kv)
  }

  def setVAattributes(path: List[String], kv: Map[String, String]): VariantSampleMatrix[T] = {
    this.copy(vaSignature = vaSignature.asInstanceOf[TStruct].setFieldAttributes(path, kv))
  }

  def deleteVAattribute(path: String, attribute: String): VariantSampleMatrix[T] = {
    deleteVAattribute(Parser.parseAnnotationRoot(path, Annotation.VARIANT_HEAD), attribute)
  }

  def deleteVAattribute(path: List[String], attribute: String): VariantSampleMatrix[T] = {
    this.copy(vaSignature = vaSignature.asInstanceOf[TStruct].deleteFieldAttributes(path, attribute))
  }

  def getVAattributes(path: String): Map[String, String] = {
    vaSignature.asInstanceOf[TStruct]
      .fieldOption(Parser.parseAnnotationRoot(path, Annotation.VARIANT_HEAD))
      .map(f => f.attrs)
      .getOrElse(Map[String, String]())
  }

  def getVAattributesAsJava(path: String): util.Map[String, String] = {
    getVAattributes(path).asJava
  }

  override def toString = s"VariantSampleMatrix(metadata=$metadata, rdd=$rdd, sampleIds=$sampleIds, nSamples=$nSamples, vaSignature=$vaSignature, saSignature=$saSignature, globalSignature=$globalSignature, sampleAnnotations=$sampleAnnotations, sampleIdsAndAnnotations=$sampleIdsAndAnnotations, globalAnnotation=$globalAnnotation, wasSplit=$wasSplit)"

  def nSamples: Int = metadata.sampleIds.length

  def typecheck() {
    var foundError = false
    if (!globalSignature.typeCheck(globalAnnotation)) {
      warn(
        s"""found violation in global annotation
           |Schema: ${ globalSignature.toPrettyString() }
           |
            |Annotation: ${ Annotation.printAnnotation(globalAnnotation) }""".stripMargin)
    }

    sampleIdsAndAnnotations.find { case (_, sa) => !saSignature.typeCheck(sa) }
      .foreach { case (s, sa) =>
        foundError = true
        warn(
          s"""found violation in sample annotations for sample $s
             |Schema: ${ saSignature.toPrettyString() }
             |
              |Annotation: ${ Annotation.printAnnotation(sa) }""".stripMargin)
      }

    val localVaSignature = vaSignature
    variantsAndAnnotations.find { case (_, va) => !localVaSignature.typeCheck(va) }
      .foreach { case (v, va) =>
        foundError = true
        warn(
          s"""found violation in variant annotations for variant $v
             |Schema: ${ localVaSignature.toPrettyString() }
             |
              |Annotation: ${ Annotation.printAnnotation(va) }""".stripMargin)
      }

    if (foundError)
      fatal("found one or more type check errors")
  }

  def sampleIdsAndAnnotations: IndexedSeq[(String, Annotation)] = sampleIds.zip(sampleAnnotations)

  def variantsAndAnnotations: OrderedRDD[Locus, Variant, Annotation] = rdd.mapValuesWithKey { case (v, (va, gs)) => va }.asOrderedRDD

  def variantEC: EvalContext = {
    val aggregationST = Map(
      "global" -> (0, globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vaSignature),
      "g" -> (3, genotypeSignature),
      "s" -> (4, TSample),
      "sa" -> (5, saSignature))
    EvalContext(Map(
      "global" -> (0, globalSignature),
      "v" -> (1, TVariant),
      "va" -> (2, vaSignature),
      "gs" -> (3, TAggregable(genotypeSignature, aggregationST))))
  }

  def variantsKT(): KeyTable = {
    val localVASignature = vaSignature
    KeyTable(hc, rdd.map { case (v, (va, gs)) =>
      Annotation(v, va)
    },
      TStruct(
        "v" -> TVariant,
        "va" -> vaSignature),
      Array("v"))
  }

  /**
    *
    * @param config VEP configuration file
    * @param root Variant annotation path to store VEP output
    * @param csq Annotates with the VCF CSQ field as a string, rather than the full nested struct schema
    * @param blockSize Variants per VEP invocation
    */
  def vep(config: String, root: String = "va.vep", csq: Boolean = false,
    blockSize: Int = 1000): VariantSampleMatrix[T] = {
    VEP.annotate(this, config, root, csq, blockSize)
  }

  def writeMetadata(dirname: String, parquetGenotypes: Boolean) {
    if (!dirname.endsWith(".vds") && !dirname.endsWith(".vds/"))
      fatal(s"output path ending in `.vds' required, found `$dirname'")

    val sqlContext = hc.sqlContext
    val hConf = hc.hadoopConf
    hConf.mkDir(dirname)

    val sb = new StringBuilder

    saSignature.pretty(sb, printAttrs = true, compact = true)
    val saSchemaString = sb.result()

    sb.clear()
    vaSignature.pretty(sb, printAttrs = true, compact = true)
    val vaSchemaString = sb.result()

    sb.clear()
    globalSignature.pretty(sb, printAttrs = true, compact = true)
    val globalSchemaString = sb.result()

    sb.clear()
    genotypeSignature.pretty(sb, printAttrs = true, compact = true)
    val genotypeSchemaString = sb.result()

    val sampleInfoJson = JArray(
      sampleIdsAndAnnotations
        .map { case (id, annotation) =>
          JObject(List(("id", JString(id)), ("annotation", JSONAnnotationImpex.exportAnnotation(annotation, saSignature))))
        }
        .toList
    )

    val json = JObject(
      ("version", JInt(VariantSampleMatrix.fileVersion)),
      ("split", JBool(wasSplit)),
      ("isDosage", JBool(isDosage)),
      ("isGenericGenotype", JBool(isGenericGenotype)),
      ("parquetGenotypes", JBool(parquetGenotypes)),
      ("sample_annotation_schema", JString(saSchemaString)),
      ("variant_annotation_schema", JString(vaSchemaString)),
      ("global_annotation_schema", JString(globalSchemaString)),
      ("genotype_schema", JString(genotypeSchemaString)),
      ("sample_annotations", sampleInfoJson),
      ("global_annotation", JSONAnnotationImpex.exportAnnotation(globalAnnotation, globalSignature))
    )

    hConf.writeTextFile(dirname + "/metadata.json.gz")(Serialization.writePretty(json, _))
  }
}
