package is.hail.annotations

import java.io.{DataInputStream, DataOutputStream}

import is.hail.expr._
import is.hail.utils.{SerializableHadoopConfiguration, _}
import net.jpountz.lz4.{LZ4BlockInputStream, LZ4Factory}
import org.apache.commons.lang3.StringUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
import org.apache.spark.{Partition, SparkContext, TaskContext}

class Decoder(in: DataInputStream) {

  def readByte(): Byte = {
    val i = in.read()
    assert(i != -1)
    i.toByte
  }

  def readBoolean(): Boolean = readByte() != 0

  def readInt(): Int = {
    var b: Byte = readByte()
    var x: Int = b & 0x7f
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = readByte()
      x |= ((b & 0x7f) << shift)
      shift += 7
    }

    x
  }

  def readLong(): Long = {
    var b: Byte = readByte()
    var x: Long = b & 0x7fL
    var shift: Int = 7
    while ((b & 0x80) != 0) {
      b = readByte()
      x |= ((b & 0x7fL) << shift)
      shift += 7
    }

    x
  }

  def readFloat(): Float = in.readFloat()

  def readDouble(): Double = in.readDouble()

  def readBytes(b: Array[Byte], off: Int, n: Int) {
    var totalRead = 0
    while (totalRead < n) {
      val read = in.read(b, off + totalRead, n - totalRead)
      assert(read > 0)
      totalRead += read
    }
  }

  def readBinary(region: MemoryBuffer, off: Int) {
    val length = readInt()
    region.align(4)
    val boff = region.allocate(4 + length)
    region.storeInt(off, boff)
    region.storeInt(boff, length)
    readBytes(region.mem, boff + 4, length)
  }

  def readArray(t: TArray, region: MemoryBuffer, offset: Int) {
    val length = readInt()

    val contentSize = t.contentsByteSize(length)
    region.align(t.contentsAlignment)
    val aoff = region.allocate(contentSize)

    region.storeInt(offset, aoff)

    val nMissingBytes = (length + 7) / 8
    region.storeInt(aoff, length)
    readBytes(region.mem, aoff + 4, nMissingBytes)

    val elemsOff = aoff + t.elementsOffset(length)
    val elemSize = UnsafeUtils.arrayElementSize(t.elementType)

    var i = 0
    while (i < length) {
      if (!region.loadBit(aoff + 4, i)) {
        val off = elemsOff + i * elemSize
        t.elementType match {
          case t2: TStruct => readStruct(t2, region, off)
          case t2: TArray => readArray(t2, region, off)
          case TBoolean => region.storeByte(off, readBoolean().toByte)
          case TInt32 => region.storeInt(off, readInt())
          case TInt64 => region.storeLong(off, readLong())
          case TFloat32 => region.storeFloat(off, readFloat())
          case TFloat64 => region.storeDouble(off, readDouble())
          case TBinary => readBinary(region, off)
        }
      }
      i += 1
    }
  }

  def readStruct(t: TStruct, region: MemoryBuffer, offset: Int) {
    val nMissingBytes = (t.size + 7) / 8
    readBytes(region.mem, offset, nMissingBytes)

    var i = 0
    while (i < t.size) {
      if (!region.loadBit(offset, i)) {
        val f = t.fields(i)
        val off = offset + t.byteOffsets(i)
        f.typ match {
          case t2: TStruct => readStruct(t2, region, off)
          case t2: TArray => readArray(t2, region, off)
          case TBoolean => region.storeByte(off, readBoolean().toByte)
          case TInt32 => region.storeInt(off, readInt())
          case TInt64 => region.storeLong(off, readLong())
          case TFloat32 => region.storeFloat(off, readFloat())
          case TFloat64 => region.storeDouble(off, readDouble())
          case TBinary => readBinary(region, off)
        }
      }
      i += 1
    }
  }

  def readRegionValue(t: TStruct, region: MemoryBuffer): Int = {
    region.align(t.alignment)
    val start = region.allocate(t.byteSize)

    readStruct(t, region, start)

    start
  }
}

class Encoder(out: DataOutputStream) {

  def writeByte(b: Byte) {
    out.write(b)
  }

  def writeBoolean(b: Boolean) {
    writeByte(b.toByte)
  }

  def writeInt(i: Int) {
    var j = i
    do {
      var b = j & 0x7f
      j >>>= 7
      if (j != 0)
        b |= 0x80
      writeByte(b.toByte)
    } while (j != 0)
  }

  def writeLong(l: Long) {
    var j = l
    do {
      var b = j & 0x7f
      j >>>= 7
      if (j != 0)
        b |= 0x80
      writeByte(b.toByte)
    } while (j != 0)
  }

  def writeFloat(f: Float) {
    out.writeFloat(f)
  }

  def writeDouble(d: Double) {
    out.writeDouble(d)
  }

  def writeBytes(b: Array[Byte], off: Int, n: Int) {
    out.write(b, off, n)
  }

  def writeBinary(region: MemoryBuffer, offset: Int) {
    val boff = region.loadInt(offset)
    val length = region.loadInt(boff)
    writeInt(length)
    writeBytes(region.mem, boff + 4, length)
  }

  def writeArray(t: TArray, region: MemoryBuffer, offset: Int) {
    val aoff = region.loadInt(offset)
    val length = region.loadInt(aoff)

    val nMissingBytes = (length + 7) / 8
    writeInt(length)
    writeBytes(region.mem, aoff + 4, nMissingBytes)

    val elemsOff = aoff + t.elementsOffset(length)
    val elemSize = UnsafeUtils.arrayElementSize(t.elementType)
    var i = 0
    while (i < length) {
      if (!region.loadBit(aoff + 4, i)) {
        val off = elemsOff + i * elemSize
        t.elementType match {
          case t2: TStruct => writeStruct(t2, region, off)
          case t2: TArray => writeArray(t2, region, off)
          case TBoolean => writeBoolean(region.loadByte(off) != 0)
          case TInt32 => writeInt(region.loadInt(off))
          case TInt64 => writeLong(region.loadLong(off))
          case TFloat32 => writeFloat(region.loadFloat(off))
          case TFloat64 => writeDouble(region.loadDouble(off))
          case TBinary => writeBinary(region, off)
        }
      }

      i += 1
    }
  }

  def writeStruct(t: TStruct, region: MemoryBuffer, offset: Int) {
    val nMissingBytes = (t.size + 7) / 8
    writeBytes(region.mem, offset, nMissingBytes)

    var i = 0
    while (i < t.size) {
      if (!region.loadBit(offset, i)) {
        val off = offset + t.byteOffsets(i)
        t.fields(i).typ match {
          case t2: TStruct => writeStruct(t2, region, off)
          case t2: TArray => writeArray(t2, region, off)
          case TBoolean => writeBoolean(region.loadByte(off) != 0)
          case TInt32 => writeInt(region.loadInt(off))
          case TInt64 => writeLong(region.loadLong(off))
          case TFloat32 => writeFloat(region.loadFloat(off))
          case TFloat64 => writeDouble(region.loadDouble(off))
          case TBinary => writeBinary(region, off)
        }
      }

      i += 1
    }
  }

  def writeRegionValue(t: TStruct, region: MemoryBuffer, offset: Int) {
    writeStruct(t, region, offset)
  }
}

class RichRDDUnsafeRow(val rdd: RDD[UnsafeRow]) extends AnyVal {
  def writeRows(path: String, t: TStruct) {
    val sc = rdd.sparkContext
    val hadoopConf = sc.hadoopConfiguration

    hadoopConf.mkDir(path + "/rowstore")

    val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(hadoopConf))

    val nPartitions = rdd.partitions.length
    val d = digitsNeeded(nPartitions)

    val rowCount = rdd.mapPartitionsWithIndex { case (i, it) =>
      val buffer = new Array[Byte](8 * 1024)
      var rowCount = 0L

      val is = i.toString
      assert(is.length <= d)
      val pis = StringUtils.leftPad(is, d, "0")

      sHadoopConfBc.value.value.writeLZ4DataFile(path + "/rowstore/part-" + pis,
        64 * 1024,
        LZ4Factory.fastestInstance().highCompressor()) { out =>
        val en = new Encoder(out)

        it.foreach { r =>

          val rowSize = r.region.offset
          out.writeInt(rowSize)

          en.writeRegionValue(t.fundamentalType, r.region, r.offset)

          rowCount += 1
        }

        out.writeInt(-1)
      }

      Iterator(rowCount)
    }
      .fold(0L)(_ + _)

    info(s"wrote $rowCount records")
  }
}

case class ReadRowsRDDPartition(index: Int) extends Partition

class ReadRowsRDD(sc: SparkContext,
  path: String, t: TStruct, nPartitions: Int) extends RDD[Row](sc, Nil) {
  val ttBc = BroadcastTypeTree(sc, t)

  override def getPartitions: Array[Partition] =
    Array.tabulate(nPartitions)(i => ReadRowsRDDPartition(i))

  private val sHadoopConfBc = sc.broadcast(new SerializableHadoopConfiguration(sc.hadoopConfiguration))

  override def compute(split: Partition, context: TaskContext): Iterator[UnsafeRow] = {
    val d = digitsNeeded(nPartitions)
    val localPath = path

    new Iterator[UnsafeRow] {
      private val in = {
        val is = split.index.toString
        assert(is.length <= d)
        val pis = StringUtils.leftPad(is, d, "0")
        new DataInputStream(
          new LZ4BlockInputStream(sHadoopConfBc.value.value.unsafeReader(localPath + "/rowstore/part-" + pis),
            LZ4Factory.fastestInstance().fastDecompressor()))
      }

      private var rowSize = in.readInt()

      private val buffer = new Array[Byte](8 * 1024)
      private val region = MemoryBuffer(rowSize.max(8 * 1024))

      private val dec = new Decoder(in)

      private val t = ttBc.value.typ.asInstanceOf[TStruct]

      def hasNext: Boolean = rowSize != -1

      def next(): UnsafeRow = {
        if (!hasNext)
          throw new NoSuchElementException("next on empty iterator")

        region.clear()
        region.ensure(rowSize)

        dec.readRegionValue(t.fundamentalType, region)

        rowSize = in.readInt()
        if (rowSize == -1)
          in.close()

        new UnsafeRow(ttBc, region.copy(), 0)
      }
    }
  }
}
