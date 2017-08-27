package is.hail.stats

import breeze.linalg._
import breeze.stats.mean
import is.hail.{SparkSuite, TestUtils, stats}
import is.hail.annotations.Annotation
import is.hail.expr.TString
import org.apache.commons.math3.random.JDKRandomGenerator
import org.testng.annotations.Test

class EigenSuite extends SparkSuite {
  def assertEqual(e1: Eigen, e2: Eigen) {
    assert(e1.rowSignature == e2.rowSignature)
    assert(e1.rowIds sameElements e2.rowIds)
    assert(e1.evects == e2.evects)
    assert(e1.evals == e2.evals)
  }
  
  def assertEigenEqualityUpToSign(e1: Eigen, e2: Eigen, r: Range, tolerance: Double = 1e-6) {
    assert(e1.rowSignature == e2.rowSignature)
    assert(e1.rowIds sameElements e2.rowIds)
    assert(math.abs(max(e1.evals - e2.evals)) < 1e-6)
    r.foreach(j => TestUtils.assertVectorEqualityUpToSignDouble(e1.evects(::, j), e2.evects(::, j), tolerance))
  }
  
  // comparison over non-zero eigenvalues
  @Test def compareDirectKinshipAndLDMatrix() {
    def testMatrix(G: DenseMatrix[Int], H: DenseMatrix[Double]) {
      for (i <- 0 until H.cols) {
        H(::, i) -= mean(H(::, i))
        H(::, i) *= math.sqrt(H.rows) / norm(H(::, i))
      }
      val K = (1.0 / H.cols) * (H * H.t)
      val L = (1.0 / H.rows) * (H.t * H)

      val eigen = eigSymD(K)
      val rank = (H.rows min H.cols) - 1
      
      val vds = stats.vdsFromGtMatrix(hc)(G)
      val eigenK = vds.rrm().eigen()
      val ldMatrix = vds.ldMatrix()
      val eigenL = ldMatrix.eigen().toEigenDistributedRRM(vds, ldMatrix.nSamplesUsed).localize()
      
      val r = -rank to -1
      
      TestUtils.assertVectorEqualityUpToSignDouble(eigen.eigenvalues(r), eigenK.evals(r))
      r.foreach(j => TestUtils.assertVectorEqualityUpToSignDouble(eigen.eigenvectors(::, j), eigenK.evects(::, j)))
      assertEigenEqualityUpToSign(eigenK, eigenL, r)
    }
    
    val G = DenseMatrix((0, 1),
                        (2, 1),
                        (0, 2))

    val G1 = DenseMatrix((0,  1,  0,  2),
                         (2, -1,  0,  2),
                         (1,  2,  0, -1))

    val H1 = DenseMatrix((0.0, 1.0),
                         (2.0, 1.5),
                         (1.0, 2.0))

    val G2 = DenseMatrix((0, 1, 2),
                         (2, 1, 0),
                         (0, 2, 1))
    
    val G3 = DenseMatrix((0, 1, 2, 1, 0),
                         (2, 1, 0, 2, 1),
                         (0, 2, 0, 0, 0))

    
    testMatrix(G, convert(G, Double))
    testMatrix(G1, H1)
    testMatrix(G2, convert(G2, Double))
    testMatrix(G3, convert(G3, Double))
  }

  @Test def readWriteIdentity() {
    val fname = tmpDir.createTempFile("test", extension = ".eig")
    
    val seed = 0

    val rand = new JDKRandomGenerator()
    rand.setSeed(seed)

    val samplesIds: Array[Annotation] = Array("A", "B", "C")
    val n = 3
    val m = 10
    val W = DenseMatrix.fill[Double](n, m)(rand.nextGaussian())

    val svdW = svd(W)
    val eigen = Eigen(TString, samplesIds, svdW.leftVectors, svdW.singularValues)
 
    eigen.write(hc, fname)
    assertEqual(eigen, Eigen.read(hc, fname))
  }
}