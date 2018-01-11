package is.hail.distributedmatrix

import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.expr.{JSONAnnotationImpex, Parser}
import is.hail.expr.types.Type
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.storage.StorageLevel
import org.json4s.jackson.Serialization
import org.json4s.{JArray, JValue}
import org.json4s.jackson.JsonMethods.parse

case class KeysImpex(typeStr: String, valuesJ: JValue)

object Keys {
  def unify(left: Option[Keys], right: Option[Keys], msg: String): Option[Keys] = {
    (left, right) match {
      case (Some(l), Some(r)) =>
        l.assertSame(r, msg)
        left
      case (Some(_), _) => left
      case _ => right
    }
  }
  
  def read(sc: SparkContext, uri: String): Keys = {
    val KeysImpex(typeStr, JArray(arr)) =  sc.hadoopConfiguration.readTextFile(uri)(in =>
      try {
        val json = parse(in)
        json.extract[KeysImpex]
      } catch {
        case e: Exception => fatal(
          s"""corrupt or outdated matrix key data.
             |  Recreate with current version of Hail.
             |  Detailed exception:
             |  ${ e.getMessage }""".stripMargin)
      })
    
    val typ = Parser.parseType(typeStr)
    val values = arr.map(JSONAnnotationImpex.importAnnotation(_, typ)).toArray
    
    new Keys(typ, values)
  }
}

class Keys(val typ: Type, val values: Array[Annotation]) {
  def length: Int = values.length
  
  def assertSame(that: Keys, msg: String = "") {
    if (typ != that.typ)
      fatal(msg + s"Keys have different types: $typ, ${ that.typ }")

    if (length != that.length)
      fatal(msg + s"Differing number of keys: $length, ${ that.length }")

    var i = 0
    while (i < length) {
      if (values(i) != that.values(i))
        fatal(msg + s"Key mismatch at index $i: ${typ.str(values(i))}, ${typ.str(that.values(i))}")
      i += 1
    }
  }
  
  def filter(pred: Annotation => Boolean): Keys = new Keys(typ, values.filter(pred))
  
  // keep must be sorted, strictly increasing, in [0, length)
  def filter(keep: Array[Int]): Keys = new Keys(typ, keep.map(values))
  
  def filterAndIndex(pred: Annotation => Boolean): (Keys, Array[Int]) = {
    val (newValues, indices) = values.zipWithIndex.filter(pred).unzip
    (new Keys(typ, newValues), indices)
  }
  
  def write(sc: SparkContext, uri: String) {
    val typeStr = typ.toPrettyString(compact = true)
    val valuesJ = JArray(values.map(typ.toJSON).toList)

    sc.hadoopConfiguration.writeTextFile(uri)(out => Serialization.write(KeysImpex(typeStr, valuesJ), out))
  }
}

object KeyedBlockMatrix {
  def from(bm: BlockMatrix): KeyedBlockMatrix =
    new KeyedBlockMatrix(bm, None, None)
  
  def read(hc: HailContext, uri: String): KeyedBlockMatrix = {
    val hadoop = hc.hadoopConf
    
    val rowFile = uri + "/rowkeys"
    val rowKeys =
      if (hadoop.exists(rowFile))
        Some(Keys.read(hc.sc, rowFile))
      else
        None
    
    val colFile = uri + "/colkeys"
    val colKeys =
      if (hadoop.exists(colFile))
        Some(Keys.read(hc.sc, colFile))
      else
        None

    val bm = BlockMatrix.read(hc, uri + "/blockmatrix")

    new KeyedBlockMatrix(bm, rowKeys, colKeys)
  }
}

class KeyedBlockMatrix(val bm: BlockMatrix, val rowKeys: Option[Keys], val colKeys: Option[Keys]) extends Serializable {
  require(rowKeys.forall(_.length == bm.nRows))
  require(colKeys.forall(_.length == bm.nCols))
  
  def copy(bm: BlockMatrix = bm,
    rowKeys: Option[Keys] = rowKeys,
    colKeys: Option[Keys] = colKeys): KeyedBlockMatrix = new KeyedBlockMatrix(bm, rowKeys, colKeys)

  def cache(): KeyedBlockMatrix = copy(bm.cache())
  
  def persist(storageLevel: StorageLevel): KeyedBlockMatrix = copy(bm.persist(storageLevel))
  
  def setRowKeys(keys: Keys): KeyedBlockMatrix = {
    if (keys.length != bm.nRows)
      fatal(s"Differing number of keys and rows: $keys.length, ${bm.nRows}")    

    copy(rowKeys = Some(keys))
  }
  
  def setColKeys(keys: Keys): KeyedBlockMatrix = {
    if (keys.length != bm.nCols)
      fatal(s"Differing number of keys and cows: $keys.length, ${bm.nCols}")    

    copy(colKeys = Some(keys))
  }
  
  def dropRowKeys(): KeyedBlockMatrix = copy(rowKeys = None)

  def dropColKeys(): KeyedBlockMatrix = copy(colKeys = None)
  
  def unifyRowKeys(that: KeyedBlockMatrix): Option[Keys] = Keys.unify(rowKeys, that.rowKeys, "Inconsistent row keys. ")
  
  def unifyColKeys(that: KeyedBlockMatrix): Option[Keys] = Keys.unify(colKeys, that.colKeys, "Inconsistent col keys. ")
  
  def add(that: KeyedBlockMatrix): KeyedBlockMatrix =
    new KeyedBlockMatrix(bm.add(that.bm), unifyRowKeys(that), unifyColKeys(that))
  
  def subtract(that: KeyedBlockMatrix): KeyedBlockMatrix =
      new KeyedBlockMatrix(bm.subtract(that.bm), unifyRowKeys(that), unifyColKeys(that))


  def multiply(that: KeyedBlockMatrix): KeyedBlockMatrix = {
    Keys.unify(colKeys, that.rowKeys, "Column keys on left do not match row keys on right. ")
    new KeyedBlockMatrix(bm.multiply(that.bm), rowKeys, that.colKeys)
  }

  def multiply(bm: BlockMatrix): KeyedBlockMatrix = multiply(KeyedBlockMatrix.from(bm))

  def scalarAdd(i: Double): KeyedBlockMatrix = copy(bm.scalarAdd(i))
  
  def scalarSubtract(i: Double): KeyedBlockMatrix = copy(bm.scalarSubtract(i))
  
  def scalarMultiply(i: Double): KeyedBlockMatrix = copy(bm.scalarMultiply(i))

  def scalarDivide(i: Double): KeyedBlockMatrix = copy(bm.scalarDivide(i))

  def pointwiseMultiply(that: KeyedBlockMatrix): KeyedBlockMatrix =
    new KeyedBlockMatrix(bm.pointwiseMultiply(that.bm), unifyRowKeys(that), unifyColKeys(that))

  def pointwiseDivide(that: KeyedBlockMatrix): KeyedBlockMatrix =
    new KeyedBlockMatrix(bm.pointwiseDivide(that.bm), unifyRowKeys(that), unifyColKeys(that))
    
  def vectorAddToEveryRow(v: Array[Double]): KeyedBlockMatrix = copy(bm.vectorAddToEveryRow(v))

  def vectorAddToEveryColumn(v: Array[Double]): KeyedBlockMatrix = copy(bm.vectorAddToEveryColumn(v))

  def vectorPointwiseMultiplyEveryRow(v: Array[Double]): KeyedBlockMatrix = copy(bm.vectorPointwiseMultiplyEveryRow(v))

  def vectorPointwiseMultiplyEveryColumn(v: Array[Double]): KeyedBlockMatrix = copy(bm.vectorPointwiseMultiplyEveryColumn(v))
  
  def transpose(): KeyedBlockMatrix = new KeyedBlockMatrix(bm.transpose(), colKeys, rowKeys)

  def filterRows(pred: Annotation => Boolean): KeyedBlockMatrix =
    rowKeys match {
      case Some(rk) =>
        val (newRK, keep) = rk.filterAndIndex(pred)
        new KeyedBlockMatrix(bm.filterRows(keep.map(_.toLong)), Some(newRK), colKeys)
      case None =>
        fatal("Cannot filter rows using predicate: no row keys")
    }
  
  def filterRows(keep: Array[Int]): KeyedBlockMatrix =
    new KeyedBlockMatrix(
      bm.filterRows(keep.map(_.toLong)),
      rowKeys.map(_.filter(keep)),
      colKeys)

  def filterCols(pred: Annotation => Boolean): KeyedBlockMatrix =
    colKeys match {
      case Some(ck) =>
        val (newCK, keep) = ck.filterAndIndex(pred)
        new KeyedBlockMatrix(bm.filterCols(keep.map(_.toLong)), rowKeys, Some(newCK))
      case None =>
        fatal("Cannot filter rows using predicate: no col keys")
    }
  
  def filterCols(keep: Array[Int]): KeyedBlockMatrix =
      new KeyedBlockMatrix(
        bm.filterCols(keep.map(_.toLong)),
        rowKeys,
        colKeys.map(_.filter(keep)))

  def filter(rowPred: Annotation => Boolean, colPred: Annotation => Boolean): KeyedBlockMatrix =
    (rowKeys, colKeys) match {
      case (Some(rk), Some(ck)) =>
        val (newRK, keepRows) = rk.filterAndIndex(rowPred)
        val (newCK, keepCols) = ck.filterAndIndex(colPred)
        new KeyedBlockMatrix(
          bm.filter(keepRows.map(_.toLong), keepCols.map(_.toLong)),
          Some(newRK),
          Some(newCK))
      case (None, Some(_)) => fatal("Cannot filter predicate: no row keys")
      case (Some(_), None) => fatal("Cannot filter using predicate: no col keys")
      case _ => fatal("Cannot filter using predicates: no row keys, no col keys")
    }
  
  def filter(keepRows: Array[Int], keepCols: Array[Int]): KeyedBlockMatrix =
        new KeyedBlockMatrix(
          bm.filter(keepRows.map(_.toLong), keepCols.map(_.toLong)),
          rowKeys.map(_.filter(keepRows)),
          colKeys.map(_.filter(keepCols)))
  
  def write(uri: String, forceRowMajor: Boolean = false) {
    val sc = bm.blocks.sparkContext
    sc.hadoopConfiguration.mkDir(uri)

    rowKeys.foreach(_.write(sc, uri + "/rowkeys"))
    colKeys.foreach(_.write(sc, uri + "/colkeys"))
    bm.write(uri + "/blockmatrix", forceRowMajor)
  }
}
