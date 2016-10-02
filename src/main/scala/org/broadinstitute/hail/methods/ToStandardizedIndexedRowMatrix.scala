package org.broadinstitute.hail.methods

import breeze.linalg._
import org.apache.spark.mllib.linalg.{Vectors, SparseVector => SSparseVector}
import org.apache.spark.mllib.linalg.distributed.{IndexedRow, IndexedRowMatrix}
import org.apache.spark.rdd.RDD
import org.broadinstitute.hail.utils._
import org.broadinstitute.hail.variant.{Genotype, GenotypeStream, Variant, VariantDataset}

import scala.collection.mutable

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

class SparseGtBuilder extends Serializable {
  private val missingRowIndices = new mutable.ArrayBuilder.ofInt()
  private val rowsX = new mutable.ArrayBuilder.ofInt()
  private val valsX = new mutable.ArrayBuilder.ofDouble()
  private var sparseLength = 0 // length of rowsX and valsX (ArrayBuilder has no length), used to track missingRowIndices
  private var sumX = 0
  private var nHet = 0

  def merge(row: Int, g: Genotype): SparseGtBuilder = {
    g.gt match {
      case Some(0) =>
      case Some(1) =>
        rowsX += row
        valsX += 1d
        sparseLength += 1
        sumX += 1
        nHet += 1
      case Some(2) =>
        rowsX += row
        valsX += 2d
        sparseLength += 1
        sumX += 2
      case None =>
        missingRowIndices += sparseLength
        rowsX += row
        valsX += 0d // placeholder for meanX
        sparseLength += 1
      case _ => throw new IllegalArgumentException("Genotype value " + g.gt.get + " must be 0, 1, or 2.")
    }

    this
  }

  // variant is atomic => combOp merge not called
  def merge(that: SparseGtBuilder): SparseGtBuilder = {
    missingRowIndices ++= that.missingRowIndices.result().map(_ + sparseLength)
    rowsX ++= that.rowsX.result()
    valsX ++= that.valsX.result()
    sparseLength += that.sparseLength
    sumX += that.sumX
    nHet += that.nHet

    this
  }

  def toGtSSparseVector(nSamples: Int): Option[SparseVector[Double]] = {
    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = nSamples - nMissing

    // all HomRef | all Het | all HomVar || all missing
    if (sumX == 0 || nHet == nPresent || sumX == 2 * nPresent || nPresent == 0)
      None
    else {
      val rowsXArray = rowsX.result()
      val valsXArray = valsX.result()
      val meanX = sumX.toDouble / nPresent

      missingRowIndicesArray.foreach(valsXArray(_) = meanX)

      // Variant is atomic => combOp merge not called => rowsXArray is sorted (as expected by SparseVector constructor)
      assert(rowsXArray.isIncreasing)

      Some(new SSparseVector(nSamples, rowsXArray, valsXArray))
    }
  }

  def toGtSparseVector(nSamples: Int): Option[SparseVector[Double]] = {
    val missingRowIndicesArray = missingRowIndices.result()
    val nMissing = missingRowIndicesArray.size
    val nPresent = nSamples - nMissing

    // all HomRef | all Het | all HomVar || all missing
    if (sumX == 0 || nHet == nPresent || sumX == 2 * nPresent || nPresent == 0)
      None
    else {
      val rowsXArray = rowsX.result()
      val valsXArray = valsX.result()
      val meanX = sumX.toDouble / nPresent

      missingRowIndicesArray.foreach(valsXArray(_) = meanX)

      // Variant is atomic => combOp merge not called => rowsXArray is sorted (as expected by SparseVector constructor)
      assert(rowsXArray.isIncreasing)

      Some(new SparseVector(rowsXArray, valsXArray, nSamples))
    }
  }
}

object ToSparseIndexedRowMatrix {
  def apply(vds: VariantDataset): (Array[Variant], IndexedRowMatrix) = {
    val variants = vds.variants.collect()
    val nVariants = variants.length
    val nSamples = vds.nSamples
    val variantIdxBroadcast = vds.sparkContext.broadcast(variants.index)
    val sampleIndexBc = vds.sparkContext.broadcast(vds.sampleIds.zipWithIndex.toMap)

    val mat = vds.aggregateByVariantWithKeys[SparseGtBuilder](new SparseGtBuilder())(
      (sb, v, s, g) => sb.merge(sampleIndexBc.value(s), g),
      (sb1, sb2) => sb1.merge(sb2))
      .flatMap{ case (v, sb) => sb.toGtSSparseVector(nVariants).map(IndexedRow(variantIdxBroadcast.value(v), _)) }

    (variants, new IndexedRowMatrix(mat.cache(), nVariants, nSamples))
  }
}

// These methods filter out constant vectors, but their names don't indicate it
object ToSparseRDD {
  def apply(vds: VariantDataset): RDD[(Variant, Vector[Double])] = {
    val nSamples = vds.nSamples
    val sampleIndexBc = vds.sparkContext.broadcast(vds.sampleIds.zipWithIndex.toMap)

    vds.aggregateByVariantWithKeys[SparseGtBuilder](new SparseGtBuilder())(
      (sb, v, s, g) => sb.merge(sampleIndexBc.value(s), g),
      (sb1, sb2) => sb1.merge(sb2))
      .flatMap { case (v, sb) => sb.toGtSSparseVector(nSamples).map((v, _)) }
  }
}

// FIXME: reconcile with logreg copy
object ToGTColumn {
  def apply(gs: Iterable[Genotype]): Option[DenseMatrix[Double]] = {
    val (nCalled, gtSum, allHet) = gs.flatMap(_.nNonRefAlleles).foldLeft((0, 0, true))((acc, gt) => (acc._1 + 1, acc._2 + gt, acc._3 && (gt == 1) ))

    // allHomRef || allHet || allHomVar || allNoCall
    if (gtSum == 0 || allHet || gtSum == 2 * nCalled || nCalled == 0 )
      None
    else {
      val gtMean = gtSum.toDouble / nCalled
      val gtArray = gs.map(_.gt.map(_.toDouble).getOrElse(gtMean)).toArray
      Some(new DenseMatrix(gtArray.length, 1, gtArray))
    }
  }
}