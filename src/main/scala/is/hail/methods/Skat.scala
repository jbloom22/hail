package is.hail.methods

import is.hail.utils.{ArrayBuilder, _}
import is.hail.variant._
import is.hail.expr.{TArray, TDouble, TSet, TStruct}
import is.hail.keytable.KeyTable
import is.hail.stats.RegressionUtils
import is.hail.annotations.Annotation
import is.hail.stats.SkatModel
import breeze.linalg.{SparseVector, _}
import org.apache.spark.sql.Row

case class SkatStat(q: Double, pval: Double)

case class SkatTuple(q: Double, xw: SparseVector[Double], qtxw: DenseVector[Double])

object SkatAgg {
  val zeroVal = SkatAgg(0.0, new ArrayBuilder[SparseVector[Double]](), new ArrayBuilder[DenseVector[Double]]())

  def seqOp(sa: SkatAgg, st: SkatTuple): SkatAgg = SkatAgg(sa.q + st.q, sa.xws + st.xw, sa.qtxws + st.qtxw)

  def combOp(sa: SkatAgg, sa2: SkatAgg): SkatAgg =
    SkatAgg(sa.q + sa2.q, sa.xws ++ sa2.xws.result(), sa.qtxws ++ sa2.qtxws.result())
  
  def resultOp(sa: SkatAgg): SkatStat = ???
}

//    def resultOP(p: (Double, ArrayBuilder[SparseVector[Double]],
//      ArrayBuilder[DenseVector[Double]])): (Double, Double) = {
//      p match {
//        case (unscaledSkatStat, genotypeAB, qgAB) => {
//          val m = genotypeAB.size
//          val gw = DenseMatrix.zeros[Double](n, m)
//
//          //copy in non-zeros
//          for (i <- 0 until m) {
//            val nnz = genotypeAB.apply(i).used
//            for (j <- 0 until nnz) {
//              val index = genotypeAB.apply(i).index(j)
//              gw(i, index) = genotypeAB.apply(i).data(index)
//            }
//          }
//
//          //make Qg non-zeros array initialized with all 0's
//          val k = qgAB.apply(0).length
//          val qtgw = DenseMatrix.zeros[Double](m, k)
//
//          //copy in non-zeros
//          for (i <- 0 until m) {
//            for (j <- 0 until k) {
//              qtgw(i, j) = qgAB.apply(i)(j)
//            }
//          }
//
//          val SPG = new SkatPerGene(gw, qtgw, unscaledSkatStat / (2 * sigmaSq))
//          SPG.computeSkatStats()
//        }



case class SkatAgg(q: Double, xws: ArrayBuilder[SparseVector[Double]], qtxws: ArrayBuilder[DenseVector[Double]])

object Skat {
  def apply(vds: VariantDataset,
    keyName: String,
    variantKeys: String,
    singleKey: Boolean,
    weightExpr: String,
    yExpr: String,
    covExpr: Array[String]): KeyTable = {

    //get variables
    val (y, cov, completeSamples) = RegressionUtils.getPhenoCovCompleteSamples(vds, yExpr, covExpr)

    val n = y.size
    val k = cov.cols
    val d = n - k - 1

    if (d < 1)
      fatal(s"$n samples and $k ${ plural(k, "covariate") } including intercept implies $d degrees of freedom.")

    // fit null model
    val qr.QR(q, r) = qr.reduced.impl_reduced_DM_Double(cov)
    val beta = r \ (q.t * y)
    val res = y - cov * beta
    val sigmaSq = (res dot res) / d

    val filteredVds = vds.filterSamplesList(completeSamples.toSet)

    val (keysType, keysQuerier) = filteredVds.queryVA(variantKeys)
    val (weightType, weightQuerier) = filteredVds.queryVA(weightExpr)

    // FIXME extend to other types
    weightType match {
      case TDouble => weightQuerier.asInstanceOf[Annotation => Double]
      case _ => fatal("Weights must be Doubles")
    }
    
    val sc = filteredVds.sparkContext
    val resBc = sc.broadcast(res)

    val (keyType, keyedRdd) =
      if (singleKey) {
        (keysType, filteredVds.rdd.flatMap { case (v, (va, gs)) =>
          (Option(keysQuerier(va)), Option(weightQuerier(va))) match {
            case (Some(key), Some(w)) =>
              val wx = math.sqrt(w.asInstanceOf[Double]) * RegressionUtils.hardCalls(gs, n)
              val wSj = resBc.value dot wx 
              Some((key, SkatTuple(wSj * wSj, wx, q.t * wx)))
            case _ => None
          }
        })
      } else {
        val keyType = keysType match {
          case TArray(e) => e
          case TSet(e) => e
          case _ => fatal(s"With single_key=False, variant keys must be of type Set[T] or Array[T], got $keysType")
        }
        (keyType, filteredVds.rdd.flatMap { case (v, (va, gs)) =>
          val keys = Option(keysQuerier(va).asInstanceOf[Iterable[_]]).getOrElse(Iterable.empty)
          val optWeight = Option(weightQuerier(va))
          if (keys.isEmpty || optWeight.isEmpty)
            Iterable.empty
          else {
            val w = optWeight.get.asInstanceOf[Double]
            val wx = math.sqrt(w) * RegressionUtils.hardCalls(gs, n)
            val wSj = resBc.value dot wx
            keys.map((_, SkatTuple(wSj * wSj, wx, q.t * wx)))
          }
        })
      }
      
    val aggregatedKT = keyedRdd.aggregateByKey(SkatAgg.zeroVal)(SkatAgg.seqOp, SkatAgg.combOp)

    val skatRDD = aggregatedKT.map { case (key, value) =>
        val (qstat, pval) = SkatAgg.resultOp(value)
        Row(key, qstat, pval)
      }
    val (skatSignature, _) = TStruct(keyName -> keyType).merge(TStruct(("Qstat",TDouble),("pval",TDouble)))

   new KeyTable(filteredVds.hc, skatRDD, skatSignature, key = Array(keyName))
  }
}
