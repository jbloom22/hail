package is.hail.utils.richUtils

import breeze.linalg.DenseMatrix
import is.hail.HailContext
import is.hail.utils._

object RichDenseMatrixDouble {
  def horzcat(oms: Option[DenseMatrix[Double]]*): Option[DenseMatrix[Double]] = {
    val ms = oms.flatten
    if (ms.isEmpty)
      None
    else
      Some(DenseMatrix.horzcat(ms: _*))
  }
  
  def read(hc: HailContext, uri: String): DenseMatrix[Double] = {
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
      if (is.read() != -1) // check EOF
        fatal("Malformed matrix file")
    }

    new DenseMatrix[Double](nRows, nCols, data)
  }
}

// Not supporting generic T because its difficult to do with ArrayBuilder and not needed yet. See:
// http://stackoverflow.com/questions/16306408/boilerplate-free-scala-arraybuilder-specialization
class RichDenseMatrixDouble(val m: DenseMatrix[Double]) extends AnyVal {
  def filterRows(keepRow: Int => Boolean): DenseMatrix[Double] = {
    val ab = new ArrayBuilder[Double]()

    var nRows = 0
    for (row <- 0 until m.rows)
      if (keepRow(row)) {
        nRows += 1
        for (col <- 0 until m.cols)
          ab += m.unsafeValueAt(row, col)
      }


    new DenseMatrix[Double](rows = nRows, cols = m.cols, data = ab.result(),
        offset = 0, majorStride = m.cols, isTranspose = true)
  }

  def filterCols(keepCol: Int => Boolean): DenseMatrix[Double] = {
    val ab = new ArrayBuilder[Double]()

    var nCols = 0
    for (col <- 0 until m.cols)
      if (keepCol(col)) {
        nCols += 1
        for (row <- 0 until m.rows)
          ab += m.unsafeValueAt(row, col)
      }

    new DenseMatrix[Double](rows = m.rows, cols = nCols, data = ab.result())
  }

  def forceSymmetry() {
    require(m.rows == m.cols, "only square matrices can be made symmetric")

    var i = 0
    while (i < m.rows) {
      var j = i + 1
      while (j < m.rows) {
        m(i, j) = m(j, i)
        j += 1
      }
      i += 1
    }
  }
  
  def asArray: Array[Double] =
    if (m.offset == 0 && m.majorStride == m.rows && !m.isTranspose)
      m.data
    else
      m.toArray
  
  def write(hc: HailContext, uri: String) {
    val hadoop = hc.sc.hadoopConfiguration
    hadoop.mkDir(uri)

    hadoop.writeDataFile(uri) { os =>
      os.writeInt(m.rows)
      os.writeInt(m.cols)

      val data = m.asArray
      assert(data.length == m.rows * m.cols)

      var i = 0
      while (i < data.length) {
        os.writeDouble(data(i))
        i += 1
      }
    }
  }
}
