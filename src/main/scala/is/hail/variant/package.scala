package is.hail

import java.io.Serializable

import is.hail.utils.IntIterator

import scala.language.implicitConversions

package object variant {
  type VariantDataset = VariantSampleMatrix[Genotype]

  class RichIterableGenotype(val ig: Iterable[Genotype]) extends AnyVal {
    def toGenotypeStream(v: Variant, isDosage: Boolean): GenotypeStream =
      ig match {
        case gs: GenotypeStream => gs
        case _ =>
          val b: GenotypeStreamBuilder = new GenotypeStreamBuilder(v.nAlleles, isDosage = isDosage)
          b ++= ig
          b.result()
      }

    def hardCallIterator: IntIterator = ig match {
      case gs: GenotypeStream => gs.gsHardCallIterator
      case _ =>
        new IntIterator {
          val it: Iterator[Genotype] = ig.iterator
          override def hasNext: Boolean = it.hasNext
          override def nextInt(): Int = it.next().unboxedGT
        }
    }

    def lazyFilterWithGenotypes[T2](i2: Iterable[T2], p: (Genotype, T2) => Boolean): Iterable[Genotype] =
      new Iterable[Genotype] with Serializable {
        def iterator: Iterator[Genotype] = new Iterator[Genotype] {
          val it: Iterator[Genotype] = ig match {
            case gs: GenotypeStream => gs.genericIterator
            case _ => ig.iterator
          }
          val it2: Iterator[T2] = i2.iterator

          var pending: Boolean = false
          var pendingNext: Genotype = _

          def hasNext: Boolean = {
            while (!pending && it.hasNext && it2.hasNext) {
              val n = it.next()
              val n2 = it2.next()
              if (p(n, n2)) {
                pending = true
                pendingNext = n
              }
            }
            pending
          }

          def next(): Genotype = {
            assert(pending)
            pending = false
            pendingNext
          }
        }
      }
  }

  implicit def toRichIterableGenotype(ig: Iterable[Genotype]): RichIterableGenotype = new RichIterableGenotype(ig)

  implicit def toVDSFunctions(vds: VariantDataset): VariantDatasetFunctions = new VariantDatasetFunctions(vds)
}
