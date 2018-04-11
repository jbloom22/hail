package is.hail.methods

import is.hail.HailContext
import is.hail.linalg.RowMatrix
import is.hail.table.Table
import is.hail.utils._
import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}

case class LinearMixedModel(
  hc: HailContext,
  y: Array[Double],
  C: BDM[Double],
  Py: Array[Double],
  PC: BDM[Double],
  Z: Array[Double],
  delta: Double,
  residualSq: Double,
  ydy: Double,
  Cdy: Array[Double],
  CdC: BDM[Double]) {
  
  def fit(pathXt: String, pathPXt: String, partitionSize: Int): Table = {
    val Xt = RowMatrix.readBlockMatrix(hc, pathXt, partitionSize)
    val PXt = RowMatrix.readBlockMatrix(hc, pathPXt, partitionSize)
    
    if (Xt.nRows != PXt.nRows)
      fatal("Numbers disagree")
    
    
    
    // preservesPartitioning?
    val rdd = Xt.rows.zipPartitions(PXt.rows, preservesPartitioning = true) { case (itx, itPx) =>
        
        
    
    }
  }
}
