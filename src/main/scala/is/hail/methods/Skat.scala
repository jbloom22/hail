package is.hail.methods

import is.hail.stats.SkatPerGene
import breeze.linalg.qr.QR
import breeze.linalg.{DenseMatrix, DenseVector, SparseVector, VectorBuilder, qr}
import is.hail.expr.{TArray, TDouble, TSet, TStruct}
import is.hail.keytable.KeyTable
import is.hail.stats.RegressionUtils
import is.hail.utils.{ArrayBuilder, fatal, plural}
import is.hail.variant._
import scala.math.sqrt
import org.apache.spark.sql.Row
import is.hail.annotations.Annotation

object Skat {

  def seqOp(t1: (Double, ArrayBuilder[SparseVector[Double]], ArrayBuilder[DenseVector[Double]]),
    t2: (Double, SparseVector[Double], DenseVector[Double])):
  (Double, ArrayBuilder[SparseVector[Double]], ArrayBuilder[DenseVector[Double]]) = {
    (t1, t2) match {
      case ((sum, genotypesAB, qgAB), (skatStat, g, qg)) =>
        (skatStat + sum, {
          genotypesAB += g;
          genotypesAB
        }, {
          qgAB += qg;
          qgAB
        })
      case _ => fatal("SeqOp function passed in invalid parameters")
    }
  }

  def combOp(t1: (Double, ArrayBuilder[SparseVector[Double]], ArrayBuilder[DenseVector[Double]]),
    t2: (Double, ArrayBuilder[SparseVector[Double]], ArrayBuilder[DenseVector[Double]])):
  (Double, ArrayBuilder[SparseVector[Double]], ArrayBuilder[DenseVector[Double]]) = {
    (t1, t2) match {
      case ((sum1, genotypesAB1, qgAB1), (sum2, genotypesAB2, qgAB2)) =>
        (sum1 + sum2, {
          for (i <- 1 to genotypesAB2.size) {
            genotypesAB1 += genotypesAB2.apply(i)
          }
          genotypesAB1
        }, {
          for (i <- 1 to qgAB2.size) {
            qgAB1 += qgAB2.apply(i)
          }
          qgAB1
        }
        )
      case _ => fatal("CombOp function passed in invalid parameters")
    }
  }


  def apply(vds: VariantDataset,
    keyName: String,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: String,
    yExpr: String,
    covExpr: Array[String]): KeyTable = {

    //get variables
    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)
    println(completeSamples)
    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    // fit null model

    //expand Covariates to include intercept
    val QR(q, r) = qr.reduced.impl_reduced_DM_Double(cov)
    val beta = r \ (q.t * y)

    val residual = y - cov * beta
    val sigmaSq = (residual dot residual) / d

    // prepare variables to broadcast

    // aggregateByKey
    val filteredVds = vds.filterSamplesList(completeSamples.toSet)

    val (keysType, keysQuerier) = filteredVds.queryVA(variantKeys)
    //TODO: make this work for numerical data rather than just doubles
    val weightQuerier = filteredVds.queryVA(weightExpr) match {
      case (TDouble, q) => q.asInstanceOf[Annotation => Double]
      case (t, _) => fatal("Weights must be Doubles")
    }
    //check that weightType is numeric

    def square = (x: Double) => x * x

    val sc = filteredVds.sparkContext
    val QBc = sc.broadcast()
    val residualBc = sc.broadcast(residual)

    val (keyType, keyedRdd) =
      if (singleKey) {
        (keysType, filteredVds.rdd.flatMap { case (v, (va, gs)) =>
          val w = Option(weightQuerier(va))
          //TODO: add in option to compute HardCalls or dosages
          val keys = if (w.isEmpty) None else Option(keysQuerier(va))
          val x = RegressionUtils.hardCalls(gs, n) * sqrt(w.get)
          val wSjSqd = if (keys.isEmpty) 0.0 else {
            square(residualBc.value dot x)
          }
          keys.map((_, (wSjSqd, x, q.t * x)))
        })
      } else {
        val keyType = keysType match {
          case TArray(e) => e
          case TSet(e) => e
          case _ => fatal(s"With single_key=False, variant keys must be of type Set[T] or Array[T], got $keysType")
        }
        (keyType, filteredVds.rdd.flatMap { case (v, (va, gs)) =>
          val w = Option(weightQuerier(va))
          val keys = if (w.isEmpty) Iterable.empty
          else {
            Option(keysQuerier(va).asInstanceOf[Iterable[_]]).getOrElse(Iterable.empty)
          }
          if (keys.isEmpty)
            Iterable.empty
          else {
            val x = RegressionUtils.hardCalls(gs, n) * sqrt(w.get)
            val wSjSqd = if (keys.isEmpty) 0.0 else {
              square(residualBc.value dot x)
            }
            keys.map((_, (wSjSqd, x, q.t * x)))
          }
        })
      }

    def resultOP(p: (Double, ArrayBuilder[SparseVector[Double]],
      ArrayBuilder[DenseVector[Double]])): (Double, Double) = {
      p match {
        case (unscaledSkatStat, genotypeAB, qgAB) => {
          val m = genotypeAB.size
          val gw = DenseMatrix.zeros[Double](n, m)

          //copy in non-zeros
          for (i <- 0 until m) {
            val nnz = genotypeAB.apply(i).used
            for (j <- 0 until nnz) {
              val index = genotypeAB.apply(i).index(j)
              gw(i, index) = genotypeAB.apply(i).data(index)
            }
          }

          //make Qg non-zeros array initialized with all 0's
          val k = qgAB.apply(0).length
          val qtgw = DenseMatrix.zeros[Double](m, k)

          //copy in non-zeros
          for (i <- 0 until m) {
            for (j <- 0 until k) {
              qtgw(i, j) = qgAB.apply(i)(j)
            }
          }

          val SPG = new SkatPerGene(gw, qtgw, unscaledSkatStat / (2 * sigmaSq))
          SPG.computeSkatStats()
        }

      }


    }

    val zeroVal = (0.0, new ArrayBuilder[SparseVector[Double]](), new ArrayBuilder[DenseVector[Double]]())
    val aggregatedKT = keyedRdd.aggregateByKey(zeroVal)(seqOp, combOp)
    val skatRDD = aggregatedKT.map { case (key, value) =>
        val (skatStat, pValue) = resultOP(value)
        Row(key, skatStat, pValue)
      }
    val (skatSignature, _) = TStruct(keyName -> keyType).merge(TStruct(("skatStat",TDouble),("pValue",TDouble)))

   new KeyTable(filteredVds.hc,skatRDD,skatSignature,key = Array(keyName))
  }
}
