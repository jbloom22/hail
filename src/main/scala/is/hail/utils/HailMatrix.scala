package is.hail.utils

import breeze.linalg.{*, DenseMatrix, DenseVector}
import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.expr.Type
import org.apache.spark.mllib.linalg.distributed.BlockMatrix

trait HailMatrix[M] {
  def nRows: Int
  def nCols: Int
  
  def add(m: M): M
  def subtract(m: M): M
  def multiply(m: M): M
  
  def multiply(hv: HailVector): HailVector
  
  def vectorPointwiseMultiplyEveryColumn(v: HailVector): M
  def vectorPointwiseMultiplyEveryRow(v: HailVector): M

  def scalarAdd(e: Double): M
  def scalarMultiply(e: Double): M

  def transpose: M

  def filterRows(pred: Int => Boolean): Option[M]
  def filterCols(pred: Int => Boolean): Option[M]
  
  def write(hc: HailContext, uri: String)
}

object HailLocalMatrix {
  def read(hc: HailContext, uri: String): HailLocalMatrix = {
    val hadoop = hc.sc.hadoopConfiguration
    var nRows = 0
    var nCols = 0
    var data: Array[Double] = null

    hadoop.readDataFile(uri) { is =>
      nRows = is.readInt()
      nCols = is.readInt()
      data = Array.ofDim[Double](nRows * nCols)
      
      var i = 0
      while (i < nRows * nCols) {
        data(i) = is.readDouble()
        i += 1
      }
      // check end of file
    }

    val m = new DenseMatrix[Double](nRows, nCols, data)

    HailLocalMatrix(m)
  }
}

case class HailLocalMatrix(m: DenseMatrix[Double]) extends HailMatrix[HailLocalMatrix] {
  def nRows: Int = m.rows
  def nCols: Int = m.cols
  
  def add(lm: HailLocalMatrix) = HailLocalMatrix(m + lm.m)
  def subtract(lm: HailLocalMatrix) = HailLocalMatrix(m - lm.m)
  def multiply(lm: HailLocalMatrix) = HailLocalMatrix(m - lm.m)
  
  def scalarAdd(e: Double) = HailLocalMatrix(m + e)
  def scalarMultiply(e: Double) = HailLocalMatrix(m * e)

  def vectorPointwiseMultiplyEveryColumn(hv: HailVector) = HailLocalMatrix(m(*, ::) :* hv.v)
  def vectorPointwiseMultiplyEveryRow(hv: HailVector) = HailLocalMatrix(m(::, *) :* hv.v)

  def multiply(hv: HailVector): HailVector = HailVector(m * hv.v)
  
  def transpose = HailLocalMatrix(m.t)
  
  def filterRows(pred: Int => Boolean): Option[HailLocalMatrix] = m.filterRows(pred).map(HailLocalMatrix(_))
  def filterCols(pred: Int => Boolean): Option[HailLocalMatrix] = m.filterCols(pred).map(HailLocalMatrix(_))
    
  def write(hc: HailContext, uri: String) {
    val hadoop = hc.sc.hadoopConfiguration
    hadoop.mkDir(uri)

    hadoop.writeDataFile(uri) { os =>
      os.writeInt(nRows)
      os.writeInt(nCols)
      
      val data = m.toArrayShallow
      assert(data.length == nRows * nCols)
      
      var i = 0
      while (i < data.length) {
        os.writeDouble(data(i))
        i += 1
      }
    }
  }
  
  def toHailBlockMatrix: HailBlockMatrix = ???
}

// will use Dan's implementation
case class HailBlockMatrix(m: BlockMatrix) extends HailMatrix[HailBlockMatrix] {
  // all trait ops
  
  // def toHailLocalMatrix = ???
}


case class HailVector(v: DenseVector[Double]) {
  def add(hv: HailVector) = HailVector(v + hv.v)
  def multiply(e: Double) = HailVector(e * v)
  
  def length: Int = v.length
}

case class Keys(keySignature: Type, keyValues: Array[Annotation]) {
  def nKeys: Int = keyValues.length
  
  // def typecheck ?
}

// not sure if I should re-use the trait and class pattern for SymmetricKeyedMatrix, KeyedMatrix, and Eigen

// add object with read

case class KeyedMatrix[M](hm: HailMatrix[M], rowKeys: Keys, colKeys: Keys) {
  // all matrix ops with check of keys, e.g.
  def multiply(km: KeyedMatrix[M]): KeyedMatrix[M] = {
    // colKeys == km.rowKeys                                  // FIXME equality on Array
    //KeyedMatrix[M](rowKeys, km.colKeys, hm.multiply(km.hm)) // FIXME does not work
  }

  // filter by row or col key
}

trait SymmetricKeyedMatrix[M] {
  // all matrix ops with check of keys
  
  // symmetric filter by key
  
  def filter: M = ???
  
  def toKeyedMatrix: KeyedMatrix[M] = ???
  
  def write(uri: String) = ???
}

// add object with read
case class SymmetricKeyedMatrix[HailLocalMatrix](lm: HailLocalMatrix, keys: Keys) extends SymmetricKeyedMatrix[HailLocalMatrix] {
  // all matrix ops
  
  // compute, set rowKeys to keys, colKeys to (0 to nCols - 1)
  def eigen(): Eigen[HailLocalMatrix] = ???

  def toSymmetricKeyedBlockMatrix: SymmetricKeyedMatrix[HailBlockMatrix] = ???
  }

// add object with read
case class SymmetricKeyedMatrix[HailBlockMatrix](bm: HailBlockMatrix, keys: Keys) extends SymmetricKeyedMatrix[HailBlockMatrix] {
  // all matrix ops
  
  def toSymmetricKeyedLocalMatrix: SymmetricKeyedMatrix[HailLocalMatrix] = ???
}

// Eigen object with read

case class Eigen[M](km: KeyedMatrix[M], evals: HailVector) {
  // matrix ops with checks
  
  def dropThreshold: Eigen[M] = ???
  
  def dropProportion: Eigen[M] = ???
  
  def takeTop: Eigen[M] = ???
  
  def write(uri: String) = ???
}