package is.hail.utils.richUtils

import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.mllib.linalg.distributed.{BlockMatrix, IndexedRow, IndexedRowMatrix}
import breeze.linalg.{DenseVector => BDV, Vector => BV}
import is.hail.sparkextras.{OrderedPartitioner, OrderedRDD}
import org.apache.spark.rdd.ShuffledRDD


class RichBlockMatrix (bm: BlockMatrix) {
  def toIndexedRowMatrixOrderedPartitioner(partitioner: OrderedPartitioner[Int, Int]): IndexedRowMatrix = {
    val cols = bm.numCols().toInt

    require(cols < Int.MaxValue, s"The number of columns should be less than Int.MaxValue ($cols).")

    val rowsPerBlock = bm.rowsPerBlock
    val colsPerBlock = bm.colsPerBlock

    import partitioner.kOk
    val rows = OrderedRDD(new ShuffledRDD[Int, (Int, BDV[Double]), (Int, BDV[Double])](bm.blocks.flatMap { case ((blockRowIdx, blockColIdx), mat) =>
      mat.rowIter.zipWithIndex.map {
        case (vector, rowIdx) =>
          blockRowIdx * rowsPerBlock + rowIdx -> (blockColIdx, new BDV[Double](vector.toDense.values))
      }
    }, partitioner).setKeyOrdering(Ordering[Int]), partitioner).groupByKey().map { case (rowIdx, vectors) =>

      val wholeVector = BDV.zeros[Double](cols)

      vectors.foreach { case (blockColIdx: Int, vec: BV[Double]) =>
        val offset = colsPerBlock * blockColIdx
        wholeVector(offset until Math.min(cols, offset + colsPerBlock)) := vec
      }
      new IndexedRow(rowIdx, new DenseVector(wholeVector.data))
    }
    println("INNER1 " + partitioner.getPartition(875))
    println("INNER2 " + partitioner.getPartition(876))
    new IndexedRowMatrix(rows, bm.numRows(), cols)
  }
}
