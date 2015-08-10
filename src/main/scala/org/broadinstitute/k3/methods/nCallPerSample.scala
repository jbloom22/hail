package org.broadinstitute.k3.methods

import org.broadinstitute.k3.variant._

import scala.collection.Map

object nCallPerSample {
  def apply(vds: VariantDataset): Map[Int, Int] = {
    vds
      .mapValues(g => if (g.called) 1 else 0)
      .foldBySample(0)(_ + _)
  }
}
