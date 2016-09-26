package org.broadinstitute.hail.methods

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{Variant, VariantDataset}

object ToStandardizedIndexedRowMatrix {
  def apply(vds: VariantDataset): (Array[Variant], IndexedRowMatrix) = {
    val variants = vds.variants.collect()
    val nVariants = variants.length
    val nSamples = vds.nSamples
    val variantIdxBroadcast = vds.sparkContext.broadcast(variants.index)

    val standardized = vds
      .rdd
      .map { case (v, (va, gs)) =>
        val (count, sum) = gs.foldLeft((0, 0)) { case ((c, s), g) =>
          g.nNonRefAlleles match {
            case Some(n) => (c + 1, s + n)
            case None => (c, s)
          }
        }

        val p =
          if (count == 0) 0.0
          else sum.toDouble / (2 * count)
        val mean = 2 * p
        val sdRecip =
          if (sum == 0 || sum == 2 * count) 0.0
          else 1.0 / math.sqrt(2 * p * (1 - p) * nVariants)
        def standardize(c: Int): Double =
          (c - mean) * sdRecip

        IndexedRow(variantIdxBroadcast.value(v),
          Vectors.dense(gs.iterator.map(_.nNonRefAlleles.map(standardize).getOrElse(0.0)).toArray))
      }

    (variants, new IndexedRowMatrix(standardized.cache(), nVariants, nSamples))
  }

}

object ToIndexedRowMatrix {
  def apply(vds: VariantDataset): (Array[Variant], IndexedRowMatrix) = {
    val variants = vds.variants.collect()
    val nVariants = variants.length
    val nSamples = vds.nSamples
    val variantIdxBroadcast = vds.sparkContext.broadcast(variants.index)

    val matrix = vds
      .rdd
      .map { case (v, (va, gs)) =>
        val (count, sum) = gs.foldLeft((0, 0)) { case ((c, s), g) =>
          g.nNonRefAlleles match {
            case Some(n) => (c + 1, s + n)
            case None => (c, s)
          }
        }

        // FIXME: should filter constant and all missing here? Right now, all missing goes to all zeros.
        val p =
          if (count == 0) 0.0
          else sum.toDouble / (2 * count)
        val mean = 2 * p

        IndexedRow(variantIdxBroadcast.value(v),
          Vectors.dense(gs.iterator.map(_.nNonRefAlleles.map(_.toDouble).getOrElse(mean)).toArray))
      }

    (variants, new IndexedRowMatrix(matrix.cache(), nVariants, nSamples))
  }

}