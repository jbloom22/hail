package is.hail.driver

import breeze.linalg.{DenseMatrix, DenseVector}

object NativesCommand {
  def apply(n: Int): String = {
    val A = DenseMatrix.rand[Double](n, n)
    val v = DenseVector.rand[Double](n)
    val u = A \ v

    println(u)

    "Success!"
  }
}
