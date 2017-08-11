package is.hail.utils.richUtils

import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import breeze.linalg.{DenseVector => BDV, Vector => BV}


class RichBlockMatrix (bm: BlockMatrix) {
  def toIndexedRowMatrixPartitioner(partitioner: Partitioner): IndexedRowMatrix = {
    val cols = bm.numCols().toInt

    require(cols < Int.MaxValue, s"The number of columns should be less than Int.MaxValue ($cols).")

    val rows = bm.blocks.flatMap { case ((blockRowIdx, blockColIdx), mat) =>
      mat.rowIter.zipWithIndex.map {
        case (vector, rowIdx) =>
          blockRowIdx * bm.rowsPerBlock + rowIdx -> (blockColIdx, new BDV[Double](vector.toDense.values))
      }
    }.groupByKey(partitioner).map { case (rowIdx, vectors) =>

      val wholeVector = BDV.zeros[Double](cols)

      vectors.foreach { case (blockColIdx: Int, vec: BV[Double]) =>
        val offset = bm.colsPerBlock * blockColIdx
        wholeVector(offset until Math.min(cols, offset + bm.colsPerBlock)) := vec
      }
      new IndexedRow(rowIdx, new DenseVector(wholeVector.data))
    }
    new IndexedRowMatrix(rows, bm.numRows(), cols)
  }
}
