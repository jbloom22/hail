package is.hail.stats

import breeze.linalg.{DenseMatrix, DenseVector}
import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.distributedmatrix.BlockMatrixIsDistributedMatrix
import is.hail.expr.Type
import is.hail.utils._
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.linalg.distributed.BlockMatrix
import org.json4s.jackson

case class Eigen(rowSignature: Type, rowIds: Array[Annotation], evects: DenseMatrix[Double], evals: DenseVector[Double]) {
  require(evects.rows == rowIds.length)
  require(evects.cols == evals.length)
    
  def nEvects: Int = evals.length
  
  def filterRows(signature: Type, pred: (Annotation => Boolean)): Eigen = {
    require(signature == rowSignature)
        
    val (newRowIds, newRows) = rowIds.zipWithIndex.filter{ case (id, row) => pred(id) }.unzip
    val newEvects = evects.filterRows(newRows.toSet).getOrElse(fatal("No rows left")) // FIXME: improve message
    
    Eigen(rowSignature, newRowIds, newEvects, evals)
  }
  
  def takeRight(k: Int): Eigen = {
    if (k < 1)
      fatal(s"k must be a positive integer, got $k")
    else if (k >= nEvects)
      this
    else
      Eigen(rowSignature, rowIds,
        evects(::, (nEvects - k) until nEvects).copy, evals((nEvects - k) until nEvects).copy)
  }
  
  def evectsSpark(): linalg.DenseMatrix = new linalg.DenseMatrix(evects.rows, evects.cols, evects.data, evects.isTranspose)
  
  def evalsArray(): Array[Double] = evals.toArray
  
  def distribute(sc: SparkContext): EigenDistributed = {
    val U = BlockMatrixIsDistributedMatrix.from(sc, evects.asSpark(), 1024, 1024)
    EigenDistributed(rowSignature, rowIds, U, evals)
  }

//  private val metadataRelativePath = "/metadata.json"
//  private val evectsRelativePath = "/evects"
//  private val evalsRelativePath = "/evals"
//
//  def write(hc: HailContext, uri: String) {
//    val hadoop = hc.sc.hadoopConfiguration
//    hadoop.mkDir(uri)
//    
//    hadoop.writeDataFile(uri + matrixRelativePath)
//    
//        m.blocks.map { case ((i, j), m) =>
//      (new PairWriter(i, j), new MatrixWriter(m.numRows, m.numCols, m.toArray)) }
//      .saveAsSequenceFile(uri+matrixRelativePath)
//    
//    hadoop.writeDataFile(uri + metadataRelativePath) { os =>
//      jackson.Serialization.write(
//        EigenData(rowIds.length, evals.length, rowIds.asInstanceOf[Array[String]], evects.toArrayShallow, evals.data)
//          os)
//    }
//  }
}


case class EigenData(nSamples: Int, nEigs: Int, sampleIds: Array[String], evectsData: Array[Double], evalsData: Array[Double])

case class EigenDistributed(rowSignature: Type, rowIds: Array[Annotation], evects: BlockMatrix, evals: DenseVector[Double]) {
  require(evects.numRows() == rowIds.length)
  require(evects.numCols() == evals.length)
  
  def nEvects: Int = evals.length
  
  def evalsArray(): Array[Double] = evals.toArray
}