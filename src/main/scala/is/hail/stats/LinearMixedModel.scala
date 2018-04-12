package is.hail.stats

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import is.hail.HailContext
import is.hail.annotations.{Region, RegionValue, RegionValueBuilder}
import is.hail.expr.types.{TArray, TFloat64, TStruct}
import is.hail.linalg.RowMatrix
import is.hail.table.Table
import is.hail.utils._

case class LMMData(delta: Double, residualSq: Double, y: BDV[Double], C: BDM[Double],
  Py: BDV[Double], PC: BDM[Double], Z: BDV[Double], ydy: Double, Cdy: BDV[Double], CdC: BDM[Double])

object LinearMixedModel {
  def apply(hc: HailContext,
    delta: Double, residualSq: Double, y: Array[Double], C: BDM[Double],
    Py: Array[Double], PC: BDM[Double], Z: Array[Double], ydy: Double, Cdy: Array[Double], CdC: BDM[Double]) =
    
    new LinearMixedModel(hc, LMMData(delta, residualSq, BDV(y), C, BDV(Py), PC, BDV(Z), ydy, BDV(Cdy), CdC))
}

class LinearMixedModel(hc: HailContext, lmmData: LMMData) {
  val rowType = TStruct(
      "beta" -> TFloat64(),
      "sigma_sq" -> TFloat64(),
      "chi_sq" -> TFloat64(),
      "p_value" -> TFloat64())

  def fit(pathXt: String, pathPXt: String, partitionSize: Int): Table = {
    val Xt = RowMatrix.readBlockMatrix(hc, pathXt, partitionSize)
    val PXt = RowMatrix.readBlockMatrix(hc, pathPXt, partitionSize)
    
    if (Xt.nRows != PXt.nRows)
      fatal("Numbers disagree")
    
    val sc = hc.sc
    val lmmDataBc = sc.broadcast(lmmData)
    val rowTypeBc = sc.broadcast(rowType)

    val rvd = Xt.rows.zipPartitions(PXt.rows, preservesPartitioning = true) { case (itx, itPx) =>
      val LMMData(delta, nullResidualSq, y, c, py, pc, z, ydy, cdy, cdc) = lmmDataBc.value
      val C = c
      val Py = py
      val PC = pc
      val Z = z
      val Cdy = cdy.copy
      val CdC = cdc.copy

      val n = C.rows
      val nCovs = C.cols
      val dof = n - nCovs - 1
      val r0 = 0 to 0
      val r1 = 1 to nCovs

      val region = Region()
      val rv = RegionValue(region)
      val rvb = new RegionValueBuilder(region)
      val localRowType = rowTypeBc.value

      new Iterator[RegionValue] {
        override def hasNext: Boolean = {
          val hn = itx.hasNext
          assert(hn == itPx.hasNext)
          hn
        }

        def next(): RegionValue = {
          val (i, x0) = itx.next()
          val (i2, px0) = itPx.next()
          assert(i == i2)

          val x = BDV(x0)
          val Px = BDV(px0)
          val ZPx = Z *:* Px

          Cdy(0) = (y dot x) / delta + (Py dot ZPx)

          CdC(0, 0) = (x dot x) / delta + (Px dot ZPx)
          CdC(r0, r1) := (C.t * x) / delta + PC.t * ZPx
          CdC(r1, r0) := CdC(0, r1).t

          try {
            val b = CdC \ Cdy
            val residualSq = ydy - (Cdy dot b)
            val sigmaSq = residualSq / dof
            val chiSq = n * math.log(nullResidualSq / residualSq)
            val pValue = chiSquaredTail(chiSq, 1)
            
            rvb.start(fullRowType)
            rvb.startStruct()
            
            
          } catch {
            case e: breeze.linalg.MatrixSingularException => // rvb null
          }
        }
      }
    }
    
    new Table()
  }
}
