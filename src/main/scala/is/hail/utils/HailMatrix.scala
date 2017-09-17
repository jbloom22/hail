package is.hail.utils

import breeze.linalg.{*, DenseMatrix, DenseVector, inv}
import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.expr.Type
import is.hail.utils.richUtils.RichDenseMatrixDouble
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRowMatrix}

object HailVector {
  def read(hc: HailContext, uri: String): HailVector = fatal("Not implemented")
}

// do we handle reading and writing of vectors?
// should we just wrap Array[Double] if we're not supporting slicing? what about sparse vectors?
case class HailVector(v: DenseVector[Double]) {
  def length: Int = v.length
  
  def apply(i: Int): Double = v(i)
  
  def add(hv: HailVector): HailVector = HailVector(v + hv.v)
  
  def scalarAdd(e: Double): HailVector = HailVector(v + e)
  def scalarMultiply(e: Double): HailVector = HailVector(v * e)  
  
  def write(hc: HailContext, uri: String): Unit = fatal("Not implemented")
  
  def toHailMatrix: HailMatrix = HailLocalMatrix(v.toDenseMatrix)
  def asHailMatrix: HailMatrix = HailLocalMatrix(v.asDenseMatrix)
  
  def toArray: Array[Double] = v.toArray
  def asArray: Array[Double] =
    if (v.offset == 0 && v.stride == 1)
      v.data
    else
      v.toArray
}

object HailMatrix {
  def read(hc: HailContext, uri: String, hailMatrixType: Int): HailMatrix = { // FIXME: change on disk formats to include type, then remove parameter
    hailMatrixType match {
      case 1 => HailLocalMatrix.read(hc, uri)
      case 2 => HailBlockMatrix.read(hc, uri)
      case 3 => HailIndexedRowMatrix.read(hc, uri)
      case _ => fatal(s"Matrix type $hailMatrixType invalid")
    }
  }
}

sealed trait HailMatrix {
  def nRows: Int // or Long?
  def nCols: Int

  def transpose: HailMatrix
  
  def add(hm: HailMatrix): HailMatrix
  def subtract(hm: HailMatrix): HailMatrix
  def multiply(hm: HailMatrix): HailMatrix
  
  def pointwiseMultiply(hm: HailMatrix): HailMatrix
 
  def multiply(hv: HailVector): HailVector
  
  def vectorPointwiseAddEveryRow(hv: HailVector): HailMatrix
  def vectorPointwiseAddEveryColumn(hv: HailVector): HailMatrix
  
  def vectorPointwiseMultiplyEveryRow(hv: HailVector): HailMatrix
  def vectorPointwiseMultiplyEveryColumn(hv: HailVector): HailMatrix

  def scalarAdd(e: Double): HailMatrix
  def scalarMultiply(e: Double): HailMatrix

  def filterRows(pred: Int => Boolean): Option[HailMatrix]
  def filterCols(pred: Int => Boolean): Option[HailMatrix]
  
  def write(hc: HailContext, uri: String): Unit // hc unnecessary of block and irm...should it be built into local?
  
  def toHailLocalMatrix: HailLocalMatrix
  def toHailBlockMatrix: HailBlockMatrix
  def toHailIndexedRowMatrix: HailIndexedRowMatrix
}

object HailLocalMatrix {
  def read(hc: HailContext, uri: String): HailLocalMatrix = {
    val m = RichDenseMatrixDouble.read(hc, uri)

    HailLocalMatrix(m)
  }
}

case class HailLocalMatrix(m: DenseMatrix[Double]) extends HailMatrix {
  def nRows: Int = m.rows
  def nCols: Int = m.cols
  
  def transpose = HailLocalMatrix(m.t)
  
  def add(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => HailLocalMatrix(m + m2)
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  } 
  def subtract(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => HailLocalMatrix(m - m2)
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  }
  def multiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => HailLocalMatrix(m * m2)
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  }
  
  def pointwiseMultiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => HailLocalMatrix(m :* m2)
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  }

  def vectorPointwiseAddEveryRow(hv: HailVector): HailMatrix = HailLocalMatrix(m(::, *) :+ hv.v)
  def vectorPointwiseAddEveryColumn(hv: HailVector): HailMatrix = HailLocalMatrix(m(*, ::) :+ hv.v)
  
  def vectorPointwiseMultiplyEveryRow(hv: HailVector) = HailLocalMatrix(m(::, *) :* hv.v)
  def vectorPointwiseMultiplyEveryColumn(hv: HailVector) = HailLocalMatrix(m(*, ::) :* hv.v)
  
  def scalarAdd(e: Double): HailMatrix = HailLocalMatrix(m + e)
  def scalarSubtract(e: Double): HailMatrix = HailLocalMatrix(m - e)
  def scalarMultiply(e: Double): HailMatrix = HailLocalMatrix(m * e)
  
  def multiply(hv: HailVector): HailVector = HailVector(m * hv.v)
    
  def filterRows(pred: Int => Boolean): Option[HailLocalMatrix] = m.filterRows(pred).map(HailLocalMatrix(_))
  def filterCols(pred: Int => Boolean): Option[HailLocalMatrix] = m.filterCols(pred).map(HailLocalMatrix(_))
  
  def write(hc: HailContext, uri: String): Unit = m.write(hc, uri)

  def toHailLocalMatrix: HailLocalMatrix = this
  def toHailBlockMatrix: HailBlockMatrix = fatal("Not implemented")
  def toHailIndexedRowMatrix: HailIndexedRowMatrix = fatal("Not implemented")
  
  // Specific to HailLocalMatrix
  def apply(i: Int, j: Int): Double = m(i, j)
  
  def inverse: HailMatrix = HailLocalMatrix(inv(m))
  def solve(hv: HailVector): HailVector = HailVector(m \ hv.v)
  
  def toArray: Array[Double] = m.toArray
  def asArray: Array[Double] = m.asArray
}

// will use Dan's implementation of BlockMatrix
object HailBlockMatrix {
  def read(hc: HailContext, uri: String): HailBlockMatrix = fatal("Not implemented")
}

case class HailBlockMatrix(m: BlockMatrix) extends HailMatrix {
  def nRows: Int = fatal("Not implemented")
  def nCols: Int = fatal("Not implemented")

  def transpose: HailMatrix = fatal("Not implemented")
  
  def add(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  }
  def subtract(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  }
  def multiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  }

  def pointwiseMultiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  }
  
  def multiply(hv: HailVector): HailVector = fatal("Not implemented")

  def vectorPointwiseAddEveryRow(hv: HailVector): HailMatrix = fatal("Not implemented")
  def vectorPointwiseAddEveryColumn(hv: HailVector): HailMatrix = fatal("Not implemented")  
  
  def vectorPointwiseMultiplyEveryRow(hv: HailVector): HailMatrix = fatal("Not implemented")
  def vectorPointwiseMultiplyEveryColumn(hv: HailVector): HailMatrix = fatal("Not implemented")

  def scalarAdd(e: Double): HailMatrix = fatal("Not implemented")
  def scalarMultiply(e: Double): HailMatrix = fatal("Not implemented")

  def filterRows(pred: Int => Boolean): Option[HailMatrix] = fatal("Not implemented")
  def filterCols(pred: Int => Boolean): Option[HailMatrix] = fatal("Not implemented")
  
  def write(hc: HailContext, uri: String): Unit = fatal("Not implemented")
  
  def toHailLocalMatrix: HailLocalMatrix = fatal("Not implemented")
  def toHailBlockMatrix: HailBlockMatrix = this
  def toHailIndexedRowMatrix: HailIndexedRowMatrix = fatal("Not implemented")
}

object HailIndexedRowMatrix {
  def read(hc: HailContext, uri: String): HailIndexedRowMatrix = fatal("Not implemented")
}

// will reimplement and use reflection to get SVD
case class HailIndexedRowMatrix(m: IndexedRowMatrix) extends HailMatrix {
  def nRows: Int = fatal("Not implemented")
  def nCols: Int = fatal("Not implemented")

  def transpose: HailMatrix = fatal("Not implemented")
  
  def add(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  }
  def subtract(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  }
  def multiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  }

  def pointwiseMultiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case HailBlockMatrix(m2) => fatal("Not implemented")
    case HailIndexedRowMatrix(m2) => fatal("Not implemented")
  }
  
  def multiply(hv: HailVector): HailVector = fatal("Not implemented")

  def vectorPointwiseAddEveryRow(hv: HailVector): HailMatrix = fatal("Not implemented")
  def vectorPointwiseAddEveryColumn(hv: HailVector): HailMatrix = fatal("Not implemented")  
  
  def vectorPointwiseMultiplyEveryRow(hv: HailVector): HailMatrix = fatal("Not implemented")
  def vectorPointwiseMultiplyEveryColumn(hv: HailVector): HailMatrix = fatal("Not implemented")

  def scalarAdd(e: Double): HailMatrix = fatal("Not implemented")
  def scalarMultiply(e: Double): HailMatrix = fatal("Not implemented")

  def filterRows(pred: Int => Boolean): Option[HailMatrix] = fatal("Not implemented")
  def filterCols(pred: Int => Boolean): Option[HailMatrix] = fatal("Not implemented")
  
  def write(hc: HailContext, uri: String): Unit = fatal("Not implemented")
  
  def toHailLocalMatrix: HailLocalMatrix = fatal("Not implemented")
  def toHailBlockMatrix: HailBlockMatrix = fatal("Not implemented")
  def toHailIndexedRowMatrix: HailIndexedRowMatrix = this
}



object Keys {
  def read(hc: HailContext, uri: String): Keys = fatal("Not implemented")
}

case class Keys(signature: Type, values: IndexedSeq[Annotation]) {
//  override def equals(that: Keys): Boolean = signature == that.signature && (values sameElements that.values)

  def length: Int = values.length
  
  def apply(i: Int): Annotation = {
    if (i >= 0 && i < values.length) // should this be a require?
      fatal(s"Key index $i out of range with $length keys")
    
    values(i)
  }
  
  def isEmpty: Boolean = values.isEmpty
  
  def filter(pred: Annotation => Boolean): Keys = Keys(signature, values.filter(pred))
  
  def typeCheck(): Unit = values.foreach(signature.typeCheck) // sensible?
  
  def write(hc: HailContext, uri: String): Unit = {
    
  }
}


// Do we need a KeyedVector?


object KeyedMatrix {
  private val rowKeysRelativePath = "/row_keys"
  private val colKeysRelativePath = "/col_keys"
  private val hailMatrixRelativePath = "/hail_matrix"

  def read(hc: HailContext, uri: String): KeyedMatrix = {
    val rowKeys = Keys.read(hc, uri + rowKeysRelativePath)
    val colKeys = Keys.read(hc, uri + rowKeysRelativePath)
    val hm = HailMatrix.read(hc, uri + hailMatrixRelativePath, 1)
    
    KeyedMatrix(rowKeys, colKeys, hm)
  }
}

case class KeyedMatrix(rowKeys: Keys, colKeys: Keys, hm: HailMatrix) {
  require(rowKeys.length == hm.nRows && colKeys.length == hm.nCols)
  
  def nRows: Int = hm.nRows
  def nCols: Int = hm.nCols

  def transpose: KeyedMatrix = KeyedMatrix(colKeys, rowKeys, hm.transpose)
  
  def add(km: KeyedMatrix): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.add(km.hm)) // what checks to do on keys?
  def subtract(km: KeyedMatrix): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.subtract(km.hm))
  def multiply(km: KeyedMatrix): KeyedMatrix = {
    if (colKeys != km.rowKeys)
      fatal("Column keys of left matrix must match row keys of right matrix")    
    KeyedMatrix(rowKeys, km.colKeys, hm.multiply(km.hm))
  }
  
  def pointwiseMultiply(km: KeyedMatrix): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.pointwiseMultiply(km.hm))
  
  def multiply(hv: HailVector): HailVector = hm.multiply(hv)

  def vectorPointwiseAddEveryRow(hv: HailVector): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.vectorPointwiseAddEveryRow(hv))
  def vectorPointwiseAddEveryColumn(hv: HailVector): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.vectorPointwiseAddEveryColumn(hv))
  
  def vectorPointwiseMultiplyEveryRow(hv: HailVector): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.vectorPointwiseMultiplyEveryRow(hv))
  def vectorPointwiseMultiplyEveryColumn(hv: HailVector): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.vectorPointwiseMultiplyEveryColumn(hv))

  def scalarAdd(e: Double): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.scalarAdd(e))
  def scalarMultiply(e: Double): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.scalarMultiply(e))

  // FIXME: don't evaluate pred twice
  def filterRows(pred: Annotation => Boolean): Option[KeyedMatrix] =
    hm.filterRows(rowKeys.values.map(pred)).map(KeyedMatrix(rowKeys.filter(pred), colKeys, _))
    
  def filterCols(pred: Annotation => Boolean): Option[KeyedMatrix] =
    hm.filterCols(colKeys.values.map(pred)).map(KeyedMatrix(rowKeys, colKeys.filter(pred), _))
  
  def write(hc: HailContext, uri: String): Unit = {
    import KeyedMatrix._
    
    rowKeys.write(hc, uri + rowKeysRelativePath)
    colKeys.write(hc, uri + rowKeysRelativePath)
    hm.write(hc, uri + hailMatrixRelativePath)
  }
  
  def toKeyedHailLocalMatrix: KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.toHailLocalMatrix)
  def toKeyedHailBlockMatrix: KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.toHailBlockMatrix)
  def toKeyedHailIndexedRowMatrix: KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.toHailIndexedRowMatrix)

  // This does not test for symmetry...should it? Or should we kill?
  def toSymmetricKeyedMatrix: SymmetricKeyedMatrix = {
    if (rowKeys != colKeys)
      fatal("Row keys must match column keys")
    
    SymmetricKeyedMatrix(rowKeys, hm)
  }
  
  def grammian(): SymmetricKeyedMatrix = SymmetricKeyedMatrix(rowKeys, hm.multiply(hm.transpose))
}




object SymmetricKeyedMatrix {
  private val keysRelativePath = "/keys"
  private val hailMatrixRelativePath = "/hail_matrix"

  def read(hc: HailContext, uri: String): SymmetricKeyedMatrix = {
    val keys = Keys.read(hc, uri + keysRelativePath)
    val hm = HailMatrix.read(hc, uri + hailMatrixRelativePath, 1)
    
    SymmetricKeyedMatrix(keys, hm)
  }
}

case class SymmetricKeyedMatrix(keys: Keys, hm: HailMatrix) {
  def nRows: Int = hm.nRows
  def nCols: Int = hm.nCols

  def transpose: SymmetricKeyedMatrix = SymmetricKeyedMatrix(keys, hm.transpose)
  
  def add(skm: SymmetricKeyedMatrix): SymmetricKeyedMatrix = SymmetricKeyedMatrix(keys, hm.add(skm.hm)) // what checks to do on keys?
  def subtract(skm: SymmetricKeyedMatrix): SymmetricKeyedMatrix = SymmetricKeyedMatrix(keys, hm.subtract(skm.hm))
  def multiply(skm: SymmetricKeyedMatrix): SymmetricKeyedMatrix = {
    if (keys != skm.keys)
      fatal("Keys of left matrix must match keys of right matrix")    
    SymmetricKeyedMatrix(keys, hm.multiply(skm.hm))
  }
  
  def pointwiseMultiply(skm: SymmetricKeyedMatrix): SymmetricKeyedMatrix = SymmetricKeyedMatrix(keys, hm.pointwiseMultiply(skm.hm))
  
  def multiply(hv: HailVector): HailVector = hm.multiply(hv)
  
  def scalarAdd(e: Double): SymmetricKeyedMatrix = SymmetricKeyedMatrix(keys, hm.scalarAdd(e))
  def scalarMultiply(e: Double): SymmetricKeyedMatrix = SymmetricKeyedMatrix(keys, hm.scalarMultiply(e))

  // FIXME: don't evaluate pred twice and filter more intelligently than two passes
  def filter(pred: Annotation => Boolean): Option[SymmetricKeyedMatrix] = {
    val indexPred = keys.values.map(pred)

    hm.filterRows(indexPred)
      .flatMap(_.filterCols(indexPred))
      .map(SymmetricKeyedMatrix(keys.filter(pred), _))
  }
  
  def write(hc: HailContext, uri: String): Unit = {
    import SymmetricKeyedMatrix._
    
    keys.write(hc, uri + keysRelativePath)
    hm.write(hc, uri + hailMatrixRelativePath)
  }

  def toSymmetricKeyedHailLocalMatrix: SymmetricKeyedMatrix = SymmetricKeyedMatrix(keys, hm.toHailLocalMatrix)
  def toSymmetricKeyedHailBlockMatrix: SymmetricKeyedMatrix = SymmetricKeyedMatrix(keys, hm.toHailBlockMatrix)
  def toSymmetricKeyedHailIndexedRowMatrix: SymmetricKeyedMatrix = SymmetricKeyedMatrix(keys, hm.toHailIndexedRowMatrix)
  
  def toKeyedMatrix: KeyedMatrix = KeyedMatrix(keys, keys, hm)
    
  def eigen: Eigen = fatal("Not implemented")
}






object Eigen {
  private val keyedMatrixRelativePath = "/keyed_matrix"
  private val eigenvaluesRelativePath = "/evals"
  
  def read(hc: HailContext, uri: String): Eigen = {
    val km = KeyedMatrix.read(hc, uri + keyedMatrixRelativePath)
    val eigenvalues = HailVector.read(hc, uri + eigenvaluesRelativePath)
    
    Eigen(km, eigenvalues)
  }
}

case class Eigen(km: KeyedMatrix, eigenvalues: HailVector) {
  
  def dropThreshold: Eigen = fatal("Not implemented")
  
  def dropProportion: Eigen = fatal("Not implemented")
  
  def takeTop: Eigen = fatal("Not implemented")
  
  def write(hc: HailContext, uri: String): Unit = {
    import Eigen._
    
    km.write(hc: HailContext, uri + keyedMatrixRelativePath)
    eigenvalues.write(hc: HailContext, uri + eigenvaluesRelativePath)
  }
}