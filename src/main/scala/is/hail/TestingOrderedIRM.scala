package is.hail

import is.hail.sparkextras.{OrderedKey, OrderedPartitioner, OrderedRDD}
import is.hail.stats.ToNormalizedIndexedRowMatrix
import is.hail.utils._
import org.apache.spark.Partitioner
import org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix

import scala.reflect.ClassTag

object TestingOrderedIRM {

  implicit val intOrderedKey = new OrderedKey[Int, Int] {
    def project(key: Int): Int = key

    def kOrd: Ordering[Int] = Ordering.Int

    def pkOrd: Ordering[Int] = Ordering.Int

    def kct: ClassTag[Int] = implicitly[ClassTag[Int]]

    override def pkct: ClassTag[Int] = implicitly[ClassTag[Int]]
  }

  def main(args: Array[String]): Unit = {
    val hc = HailContext()
    val sc = hc.sc

    val vds = hc.baldingNicholsModel(3, 1000, 1000)

    val irm = ToNormalizedIndexedRowMatrix(vds)
    val row1 = irm.rows.take(1)(0)

    println(s"Row 1 = $row1")

    val bm = irm.toBlockMatrixDense()

    val partitionSizes = vds.rdd.mapPartitions( it => Iterator(it.length), preservesPartitioning = true).collect()

    val boundsPlus2 = partitionSizes.scanLeft(-1)(_ + _)
    val bounds = boundsPlus2.slice(1, boundsPlus2.length - 1)

    val orderedPartitioner = new OrderedPartitioner[Int, Int](bounds, partitionSizes.length)
    val irm2 = bm.toIndexedRowMatrixOrderedPartitioner(orderedPartitioner)

    val variants = vds.variants.collect()

    val rowsRDD = irm2.rows.map(ir => ((ir.index.toInt), ir)).map{case(idx, ir) => (variants(idx), ir)}

    val variantsRDD = OrderedRDD(rowsRDD, vds.rdd.orderedPartitioner)

      //val variantsRDD = rowsRDD.map()
    rowsRDD.mapPartitionsWithIndex { case (i, it) => it.map(x => i -> x._1)}
        .collect().foreach(println)

    bounds.foreach(println(_))
    println(orderedPartitioner.getPartition(125))
    println(orderedPartitioner.getPartition(126))
    println(orderedPartitioner.getPartition(127))
    vds.rdd.orderedLeftJoin(variantsRDD).count()
  }

}
