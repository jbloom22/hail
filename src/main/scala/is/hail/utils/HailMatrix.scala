package is.hail.utils

import breeze.linalg.{*, DenseMatrix, DenseVector, inv}
import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.expr.{TInt32, Type}
import is.hail.stats.eigSymD
import is.hail.utils.richUtils.RichDenseMatrixDouble
import org.apache.spark.rdd.RDD

object HailVector {
  def read(hc: HailContext, uri: String): HailVector = fatal("Not implemented")
}

// FIXME: should we wrap Array[Double] instead? Breeze adds a lot of complexity to manage slicing
case class HailVector(v: DenseVector[Double]) {
  def length: Int = v.length
  
  def apply(i: Int): Double = v(i)
  
  def slice(i: Int, j: Int): HailVector = HailVector(v(i until j))
    
  def add(hv: HailVector): HailVector = HailVector(v + hv.v)
  def dot(hv: HailVector): Double = v dot hv.v
  
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
  def read(hc: HailContext, uri: String): HailMatrix = {
    val hailMatrixType = 0 // FIXME: change on disk formats to include this type so can read type and dispatch
    
    hailMatrixType match {
      case 1 => HailLocalMatrix.read(hc, uri)
      case 2 => HailBlockMatrix.read(hc, uri)
      case 3 => HailIndexedRowMatrix.read(hc, uri)
      case _ => fatal(s"Matrix type $hailMatrixType invalid")
    }
  }
}

sealed trait HailMatrix {
  def nRows: Long // FIXME: I've made these Long but they cannot currently exceed Int when keyed
  def nCols: Long

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

  def filterRows(pred: Int => Boolean): HailMatrix
  def filterCols(pred: Int => Boolean): HailMatrix
  
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
  def nRows: Long = m.rows
  def nCols: Long = m.cols
  
  def transpose = HailLocalMatrix(m.t)
  
  def add(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => HailLocalMatrix(m + m2)
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  } 
  def subtract(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => HailLocalMatrix(m - m2)
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  }
  def multiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => HailLocalMatrix(m * m2)
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  }
  
  def pointwiseMultiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => HailLocalMatrix(m :* m2)
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  }

  def vectorPointwiseAddEveryRow(hv: HailVector): HailLocalMatrix = HailLocalMatrix(m(::, *) :+ hv.v)
  def vectorPointwiseAddEveryColumn(hv: HailVector): HailLocalMatrix = HailLocalMatrix(m(*, ::) :+ hv.v)
  
  def vectorPointwiseMultiplyEveryRow(hv: HailVector) = HailLocalMatrix(m(::, *) :* hv.v)
  def vectorPointwiseMultiplyEveryColumn(hv: HailVector) = HailLocalMatrix(m(*, ::) :* hv.v)
  
  def scalarAdd(e: Double): HailLocalMatrix = HailLocalMatrix(m + e)
  def scalarSubtract(e: Double): HailLocalMatrix = HailLocalMatrix(m - e)
  def scalarMultiply(e: Double): HailLocalMatrix = HailLocalMatrix(m * e)
  
  def multiply(hv: HailVector): HailVector = HailVector(m * hv.v)
    
  def filterRows(pred: Int => Boolean): HailLocalMatrix = HailLocalMatrix(m.filterRows(pred))
  def filterCols(pred: Int => Boolean): HailLocalMatrix = HailLocalMatrix(m.filterCols(pred))
  
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

// FIXME: use Dan's implementation of BlockMatrix
object HailBlockMatrix {
  def read(hc: HailContext, uri: String): HailBlockMatrix = fatal("Not implemented")
}

class HailBlockMatrix extends HailMatrix {
  def nRows: Long = fatal("Not implemented")
  def nCols: Long = fatal("Not implemented")

  def transpose: HailMatrix = fatal("Not implemented")
  
  def add(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  }
  def subtract(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  }
  def multiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  }

  def pointwiseMultiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  }
  
  def multiply(hv: HailVector): HailVector = fatal("Not implemented")

  def vectorPointwiseAddEveryRow(hv: HailVector): HailMatrix = fatal("Not implemented")
  def vectorPointwiseAddEveryColumn(hv: HailVector): HailMatrix = fatal("Not implemented")  
  
  def vectorPointwiseMultiplyEveryRow(hv: HailVector): HailMatrix = fatal("Not implemented")
  def vectorPointwiseMultiplyEveryColumn(hv: HailVector): HailMatrix = fatal("Not implemented")

  def scalarAdd(e: Double): HailMatrix = fatal("Not implemented")
  def scalarMultiply(e: Double): HailMatrix = fatal("Not implemented")

  def filterRows(pred: Int => Boolean): HailMatrix = fatal("Not implemented")
  def filterCols(pred: Int => Boolean): HailMatrix = fatal("Not implemented")
  
  def write(hc: HailContext, uri: String): Unit = fatal("Not implemented")
  
  def toHailLocalMatrix: HailLocalMatrix = fatal("Not implemented")
  def toHailBlockMatrix: HailBlockMatrix = this
  def toHailIndexedRowMatrix: HailIndexedRowMatrix = fatal("Not implemented")
}

object HailIndexedRowMatrix {
  def read(hc: HailContext, uri: String): HailIndexedRowMatrix = fatal("Not implemented")
}

case class HailIndexedRow(index: Long, vector: DenseVector[Double])

// FIXME: Implement.
class HailIndexedRowMatrix(val rows: RDD[HailIndexedRow], val numRows: Long, val numCols: Int) extends HailMatrix {
  def nRows: Long = fatal("Not implemented")
  def nCols: Long = fatal("Not implemented")

  def transpose: HailMatrix = fatal("Not implemented")
  
  def add(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  }
  def subtract(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  }
  def multiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  }

  def pointwiseMultiply(hm: HailMatrix): HailMatrix = hm match {
    case HailLocalMatrix(m2) => fatal("Not implemented")
    case _: HailBlockMatrix => fatal("Not implemented")
    case _: HailIndexedRowMatrix => fatal("Not implemented")
  }
  
  def multiply(hv: HailVector): HailVector = fatal("Not implemented")

  def vectorPointwiseAddEveryRow(hv: HailVector): HailMatrix = fatal("Not implemented")
  def vectorPointwiseAddEveryColumn(hv: HailVector): HailMatrix = fatal("Not implemented")  
  
  def vectorPointwiseMultiplyEveryRow(hv: HailVector): HailMatrix = fatal("Not implemented")
  def vectorPointwiseMultiplyEveryColumn(hv: HailVector): HailMatrix = fatal("Not implemented")

  def scalarAdd(e: Double): HailMatrix = fatal("Not implemented")
  def scalarMultiply(e: Double): HailMatrix = fatal("Not implemented")

  def filterRows(pred: Int => Boolean): HailMatrix = fatal("Not implemented")
  def filterCols(pred: Int => Boolean): HailMatrix = fatal("Not implemented")
  
  def write(hc: HailContext, uri: String): Unit = fatal("Not implemented")
  
  def toHailLocalMatrix: HailLocalMatrix = fatal("Not implemented")
  def toHailBlockMatrix: HailBlockMatrix = fatal("Not implemented")
  def toHailIndexedRowMatrix: HailIndexedRowMatrix = this
}



object Keys {
  def read(hc: HailContext, uri: String): Keys = fatal("Not implemented")
  
  def ofDim(length: Int): Keys = Keys(TInt32, (0 until length).toArray)
}

case class Keys(signature: Type, values: Array[Annotation]) {
  override def equals(that: Any): Boolean = that match {
    case k: Keys => signature == k.signature && values.sameElements(k.values)
    case _ => false
  }
  
  def length: Int = values.length
  
  def apply(i: Int): Annotation = {
    if (i >= 0 && i < values.length) // should this be a require?
      fatal(s"Key index $i out of range with $length keys")
    
    values(i)
  }
  
  def filter(pred: Annotation => Boolean): Keys = Keys(signature, values.filter(pred))
  
  def toSet: Set[Annotation] = values.toSet // e.g., for use in lhs.multiply(rhs.filterRows(lhs.colKeys.toSet))
  
  def typeCheck(): Unit = values.foreach(signature.typeCheck)
  
  def write(hc: HailContext, uri: String): Unit = fatal("Not implemented")
}


object KeyedMatrix {
  private val rowKeysRelativePath = "/row_keys"
  private val colKeysRelativePath = "/col_keys"
  private val hailMatrixRelativePath = "/hail_matrix"

  def read(hc: HailContext, uri: String): KeyedMatrix = {
    val rowKeys = Keys.read(hc, uri + rowKeysRelativePath)
    val colKeys = Keys.read(hc, uri + rowKeysRelativePath)
    val hm = HailMatrix.read(hc, uri + hailMatrixRelativePath)
    
    KeyedMatrix(rowKeys, colKeys, hm)
  }
}

case class KeyedMatrix(rowKeys: Keys, colKeys: Keys, hm: HailMatrix) {
  require(rowKeys.length == hm.nRows && colKeys.length == hm.nCols) // also enforces nRows and nCols < Int.MaxValue
  
  def nRows: Long = hm.nRows
  def nCols: Long = hm.nCols

  def transpose: KeyedMatrix = KeyedMatrix(colKeys, rowKeys, hm.transpose)
  
  // FIXME: when to check that keys match? e.g. here I'm checking on multiplication, but not on addition
  def add(km: KeyedMatrix): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.add(km.hm))
  def subtract(km: KeyedMatrix): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.subtract(km.hm))
  def multiply(km: KeyedMatrix): KeyedMatrix = {
    if (colKeys != km.rowKeys)
      fatal("Column keys of left matrix must match row keys of right matrix")    
    KeyedMatrix(rowKeys, km.colKeys, hm.multiply(km.hm))
  }
  
  def pointwiseMultiply(km: KeyedMatrix): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.pointwiseMultiply(km.hm))
  
  // FIXME: for checking / filtering / indexing by key, it seems natural to also have a KeyedVector
  def multiply(hv: HailVector): HailVector = hm.multiply(hv)

  def vectorPointwiseAddEveryRow(hv: HailVector): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.vectorPointwiseAddEveryRow(hv))
  def vectorPointwiseAddEveryColumn(hv: HailVector): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.vectorPointwiseAddEveryColumn(hv))
  
  def vectorPointwiseMultiplyEveryRow(hv: HailVector): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.vectorPointwiseMultiplyEveryRow(hv))
  def vectorPointwiseMultiplyEveryColumn(hv: HailVector): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.vectorPointwiseMultiplyEveryColumn(hv))

  def scalarAdd(e: Double): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.scalarAdd(e))
  def scalarMultiply(e: Double): KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.scalarMultiply(e))

  // FIXME: don't evaluate pred twice
  def filterRows(pred: Annotation => Boolean): KeyedMatrix = KeyedMatrix(rowKeys.filter(pred), colKeys, hm.filterRows(rowKeys.values.map(pred)))
  def filterCols(pred: Annotation => Boolean): KeyedMatrix = KeyedMatrix(rowKeys, colKeys.filter(pred), hm.filterCols(colKeys.values.map(pred)))
  
  // def filterRowIndices(pred: Int => Boolean): KeyedMatrix ???
  // def filterColIndices(pred: Int => Boolean): KeyedMatrix ???

  def write(hc: HailContext, uri: String): Unit = {
    import KeyedMatrix._
    
    rowKeys.write(hc, uri + rowKeysRelativePath)
    colKeys.write(hc, uri + rowKeysRelativePath)
    hm.write(hc, uri + hailMatrixRelativePath)
  }
  
  def toKeyedHailLocalMatrix: KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.toHailLocalMatrix)
  def toKeyedHailBlockMatrix: KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.toHailBlockMatrix)
  def toKeyedHailIndexedRowMatrix: KeyedMatrix = KeyedMatrix(rowKeys, colKeys, hm.toHailIndexedRowMatrix)

  // FIXME: Does not test for symmetry...should it? Or should we kill it?
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
    val hm = HailMatrix.read(hc, uri + hailMatrixRelativePath)
    
    SymmetricKeyedMatrix(keys, hm)
  }
}

case class SymmetricKeyedMatrix(keys: Keys, hm: HailMatrix) {
  def nRows: Long = hm.nRows
  def nCols: Long = hm.nCols

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

  // FIXME: don't evaluate pred twice. filter in one pass
  def filter(pred: Annotation => Boolean): SymmetricKeyedMatrix = {
    val indexPred = keys.values.map(pred)
    SymmetricKeyedMatrix(keys.filter(pred), hm.filterRows(indexPred).filterCols(indexPred))
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
    
  def eigen: Eigen = hm match {
    case HailLocalMatrix(m) =>
      info(s"Computing eigendecomposition of $nRows x $nCols matrix...")
      val eig = printTime(eigSymD(m))
      val km = KeyedMatrix(keys, Keys.ofDim(keys.length), HailLocalMatrix(eig.eigenvectors))
      val hv = HailVector(eig.eigenvalues)
      Eigen(km, hv)
    case _: HailBlockMatrix => fatal("Eigendecomposition requires local matrix")
    case _: HailIndexedRowMatrix => fatal("Eigendecomposition requires local matrix")
  }
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

case class Eigen(eigenvectors: KeyedMatrix, eigenvalues: HailVector) {
  require(eigenvectors.nCols == eigenvalues.length) // also (re)enforces nCols < Int.MaxValue
  
  def nRows: Long  = eigenvectors.nRows
  def nCols: Int = eigenvectors.nCols.toInt
  def nEigs: Int = nCols

  // FIXME: don't evaluate pred twice
  def filterEigenvalues(pred: Double => Boolean): Eigen = {
    val evects = eigenvectors.filterCols(i => pred(eigenvalues(i)))
    val evals = HailVector(DenseVector(eigenvalues.v.toArray.filter(pred)))
    Eigen(evects, evals)
  }

  def takeTop(k: Int): Eigen = { // relies on increasing order
    if (! (k > 0 && k < nCols))
      fatal(s"k must be strictly between 0 and the number of eigenvectors $nEigs, got $k")
    
    val evects = eigenvectors.filterCols(i => i.asInstanceOf[Int] >= nCols - k) // can be made more efficient
    val evals = eigenvalues.slice(nCols - k, nCols)
    
    Eigen(evects, evals)
  }
   
  def write(hc: HailContext, uri: String): Unit = {
    import Eigen._
    
    eigenvectors.write(hc: HailContext, uri + keyedMatrixRelativePath)
    eigenvalues.write(hc: HailContext, uri + eigenvaluesRelativePath)
  }
}