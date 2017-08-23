package is.hail.stats

import breeze.linalg.{DenseMatrix, DenseVector}
import is.hail.HailContext
import is.hail.annotations.Annotation
import is.hail.distributedmatrix.BlockMatrixIsDistributedMatrix
import is.hail.expr.{TString, Type}
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
  
  import Eigen._
  
  def write(hc: HailContext, uri: String) {
    val hadoop = hc.sc.hadoopConfiguration
    hadoop.mkDir(uri)

    hadoop.writeDataFile(uri + evectsRelativePath) { os =>
      val evectsData = evects.toArrayShallow
      var i = 0
      while (i < evectsData.length) {
        os.writeDouble(evectsData(i))
        i += 1
      }
    }

    hadoop.writeDataFile(uri + evalsRelativePath) { os =>
      val evalsData = evals.toArray
      var i = 0
      while (i < evalsData.length) {
        os.writeDouble(evalsData(i))
        i += 1
      }
    }
    
    hadoop.writeDataFile(uri + metadataRelativePath) { os =>
      jackson.Serialization.write(
        EigenMetadata(rowIds.length, evals.length, rowIds.map(_.toString())),
          os)
    }
  }
}

object Eigen {
  private val metadataRelativePath = "/metadata.json"
  private val evectsRelativePath = "/evects"
  private val evalsRelativePath = "/evals"
  
  def read(hc: HailContext, uri: String): Eigen = {
    val hadoop = hc.hadoopConf

    val EigenMetadata(nSamples, nEigs, sampleIds) =
      hadoop.readTextFile(uri + metadataRelativePath) { isr =>
        jackson.Serialization.read[EigenMetadata](isr)
      }

    assert(nSamples == sampleIds.length)
    
    val nEntries = nSamples * nEigs
    val evectsData = Array.ofDim[Double](nEntries)
    val evalsData = Array.ofDim[Double](nEigs)
    
    hadoop.readDataFile(uri + evectsRelativePath) { is =>
      var i = 0
      while (i < nEntries) {
        evectsData(i) = is.readDouble()
        i += 1
      }
    }

    hadoop.readDataFile(uri + evalsRelativePath) { is =>
      var i = 0
      while (i < nEigs) {
        evalsData(i) = is.readDouble()
        i += 1
      }
    }
    
    new Eigen(TString, sampleIds.asInstanceOf[Array[Annotation]], new DenseMatrix[Double](nSamples, nEigs, evectsData), DenseVector(evalsData))
  }
}

case class EigenMetadata(nSamples: Int, nEigs: Int, sampleIds: Array[String])

case class EigenDistributed(rowSignature: Type, rowIds: Array[Annotation], evects: BlockMatrix, evals: DenseVector[Double]) {
  require(evects.numRows() == rowIds.length)
  require(evects.numCols() == evals.length)
  
  def nEvects: Int = evals.length
  
  def evalsArray(): Array[Double] = evals.toArray
}