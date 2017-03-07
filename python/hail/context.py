from __future__ import print_function  # Python 2 and 3 print compatibility

from pyspark.sql import SQLContext

from hail.dataset import VariantDataset
from hail.java import *
from hail.keytable import KeyTable
from hail.type import Type
from hail.utils import TextTableConfig
from hail.stats import UniformDist, BetaDist, TruncatedBetaDist


class HailContext(object):
    """The main entrypoint for Hail functionality.

    :param sc: spark context, will be auto-generated if None
    :type sc: :class:`.pyspark.SparkContext`

    :param appName: Spark application identifier

    :param master: Spark cluster master

    :param local: local resources to use

    :param log: log path

    :param quiet: suppress log messages

    :param append: write to end of log file instead of overwriting

    :param parquet_compression: level of on-disk annotation compression

    :param min_block_size: minimum file split size in MB

    :param branching_factor: branching factor for tree aggregation

    :param tmp_dir: temporary directory for file merging

    :ivar sc: Spark context
    :vartype sc: :class:`.pyspark.SparkContext`
    """

    def __init__(self, sc=None, appName="Hail", master=None, local='local[*]',
                 log='hail.log', quiet=False, append=False, parquet_compression='snappy',
                 min_block_size=1, branching_factor=50, tmp_dir='/tmp'):

        from pyspark import SparkContext
        SparkContext._ensure_initialized()

        self._gateway = SparkContext._gateway
        self._jvm = SparkContext._jvm

        Env._jvm = self._jvm
        Env._gateway = self._gateway
        Env._hc = self

        # hail package
        self._hail = getattr(self._jvm, 'is').hail

        jsc = sc._jsc if sc  else None

        self._jhc = scala_object(self._hail, 'HailContext').apply(
            jsc, appName, joption(master), local, log, quiet, append,
            parquet_compression, min_block_size, branching_factor, tmp_dir)

        self._jsc = self._jhc.sc()
        self.sc = sc if sc else SparkContext(gateway=self._gateway, jsc=self._jvm.JavaSparkContext(self._jsc))
        self._jsql_context = self._jhc.sqlContext()
        self._sql_context = SQLContext(self.sc, self._jsql_context)

    @handle_py4j
    def grep(self, regex, path, max_count=100):
        """Grep big files, like, really fast.

        **Examples**

        Print all lines containing the string ``hello`` in *file.txt*:

        >>> hc.grep('hello','data/file.txt')

        Print all lines containing digits in *file1.txt* and *file2.txt*:

        >>> hc.grep('\d', ['data/file1.txt','data/file2.txt'])

        **Background**

        :py:meth:`~hail.HailContext.grep` mimics the basic functionality of Unix ``grep`` in parallel, printing results to screen. This command is provided as a convenience to those in the statistical genetics community who often search enormous text files like VCFs. Find background on regular expressions at `RegExr <http://regexr.com/>`_.

        :param str regex: The regular expression to match.

        :param path: The files to search.
        :type path: str or list of str

        :param int max_count: The maximum number of matches to return.
        """

        self._jhc.grep(regex, jindexed_seq_args(path), max_count)

    @handle_py4j
    def import_annotations_table(self, path, variant_expr, code=None, npartitions=None, config=TextTableConfig()):
        """Import variants and variant annotations from a delimited text file
        (text table) as a sites-only VariantDataset.

        :param path: The files to import.
        :type path: str or list of str

        :param str variant_expr: Expression to construct a variant
            from a row of the text table.  Must have type Variant.

        :param code: Expression to build the variant annotations.
        :type code: str or None

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig`

        :rtype: :class:`.VariantDataset`
        """

        jvds = self._jhc.importAnnotationsTables(jindexed_seq_args(path), variant_expr,
                                                 joption(code), joption(npartitions),
                                                 config._to_java())
        return VariantDataset(self, jvds)

    @handle_py4j
    def import_bgen(self, path, tolerance=0.2, sample_file=None, npartitions=None):
        """Import .bgen files as VariantDataset

        **Examples**

        Importing a BGEN file as a VDS (assuming it has already been indexed).

        >>> vds = hc.import_bgen("data/example3.bgen", sample_file="data/example3.sample")

        **Notes**

        Hail supports importing data in the BGEN file format. For more information on the BGEN file format,
        see `here <http://www.well.ox.ac.uk/~gav/bgen_format/bgen_format_v1.1.html>`_. Note that only v1.1 BGEN files
        are supported at this time.

        Before importing, ensure that:
        - Files reside in Hadoop file system.
        - The sample file has the same number of samples as the BGEN file.
        - No duplicate sample IDs are present.

        To load multiple files at the same time, use :ref:`Hadoop Glob Patterns <sec-hadoop-glob>`.

        .. _dosagefilters:

        **Dosage representation**:

        Since dosages are understood as genotype probabilities, :py:meth:`~hail.HailContext.import_bgen` automatically sets to missing those genotypes for which the sum of the dosages is a distance greater than the ``tolerance`` parameter from 1.0.  The default tolerance is 0.2, so a genotype with sum .79 or 1.21 is filtered out, whereas a genotype with sum .8 or 1.2 remains.

        :py:meth:`~hail.HailContext.import_gen` normalizes all dosages to sum to 1.0. Therefore, an input dosage of (0.98, 0.0, 0.0) will be stored as (1.0, 0.0, 0.0) in Hail.

        Even when the dosages sum to 1.0, Hail may store slightly different values than the original GEN file (maximum observed difference is 3E-4).

        **Annotations**

        :py:meth:`~hail.HailContext.import_bgen` adds the following variant annotations:

         - **va.varid** (*String*) -- 2nd column of .gen file if chromosome present, otherwise 1st column.

         - **va.rsid** (*String*) -- 3rd column of .gen file if chromosome present, otherwise 2nd column.

        :param path: .bgen files to import.
        :type path: str or list of str

        :param float tolerance: If the sum of the dosages for a
            genotype differ from 1.0 by more than the tolerance, set
            the genotype to missing.

        :param sample_file: The sample file.
        :type sample_file: str or None

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :rtype: :class:`.VariantDataset`
        :return: A dataset imported from the bgen file.
        """

        jvds = self._jhc.importBgens(jindexed_seq_args(path), joption(sample_file),
                                     tolerance, joption(npartitions))
        return VariantDataset(self, jvds)

    @handle_py4j
    def import_gen(self, path, sample_file=None, tolerance=0.2, npartitions=None, chromosome=None):
        """Import .gen files as VariantDataset.

        **Examples**

        Read a .gen file and a .sample file and write to a .vds file:

        >>> (hc.import_gen('data/example.gen', sample_file='data/example.sample')
        ...    .write('output/gen_example1.vds'))

        Load multiple files at the same time with :ref:`Hadoop glob patterns <sec-hadoop-glob>`:

        >>> (hc.import_gen('data/example.chr*.gen', sample_file='data/example.sample')
        ...    .write('output/gen_example2.vds'))

        **Notes**

        For more information on the .gen file format, see `here <http://www.stats.ox.ac.uk/%7Emarchini/software/gwas/file_format.html#mozTocId40300>`_.

        To ensure that the .gen file(s) and .sample file are correctly prepared for import:

        - If there are only 5 columns before the start of the dosage data (chromosome field is missing), you must specify the chromosome using the ``chromosome`` parameter

        - No duplicate sample IDs are allowed

        The first column in the .sample file is used as the sample ID ``s.id``.

        Also, see section in :py:meth:`~hail.HailContext.import_bgen` linked :ref:`here <dosagefilters>` for information about Hail's dosage representation.

        **Annotations**

        :py:meth:`~hail.HailContext.import_gen` adds the following variant annotations:

         - **va.varid** (*String*) -- 2nd column of .gen file if chromosome present, otherwise 1st column.

         - **va.rsid** (*String*) -- 3rd column of .gen file if chromosome present, otherwise 2nd column.

        :param path: .gen files to import.
        :type path: str or list of str

        :param str sample_file: The sample file.

        :param float tolerance: If the sum of the dosages for a genotype differ from 1.0 by more than the tolerance, set the genotype to missing.

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param chromosome: Chromosome if not listed in the .gen file.
        :type chromosome: str or None

        :rtype: :class:`.VariantDataset`
        :return: A dataset imported from a .gen and .sample file.
        """

        jvds = self._jhc.importGens(jindexed_seq_args(path), sample_file, joption(chromosome), joption(npartitions), tolerance)
        return VariantDataset(self, jvds)

    @handle_py4j
    def import_keytable(self, path, key_names=[], npartitions=None, config=TextTableConfig()):
        """Import delimited text file (text table) as KeyTable.

        :param path: files to import.
        :type path: str or list of str

        :param key_names: The name(s) of fields to be considered keys
        :type key_names: str or list of str

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param config: Configuration options for importing text files
        :type config: :class:`.TextTableConfig`

        :rtype: :class:`.KeyTable`
        """

        if not config:
            config = TextTableConfig()

        jkt = self._jhc.importKeyTable(jindexed_seq_args(path), jindexed_seq_args(key_names),
                                       joption(npartitions), config._to_java())
        return KeyTable(self, jkt)

    @handle_py4j
    def import_plink(self, bed, bim, fam, npartitions=None, delimiter='\\\\s+', missing='NA', quantpheno=False):
        """Import PLINK binary file (BED, BIM, FAM) as VariantDataset

        **Examples**

        Import data from a PLINK binary file:

        >>> vds = hc.import_plink(bed="data/test.bed",
        ...                       bim="data/test.bim",
        ...                       fam="data/test.fam")

        **Implementation Details**

        Only binary SNP-major mode files can be read into Hail. To convert your file from individual-major mode to SNP-major mode, use PLINK to read in your fileset and use the ``--make-bed`` option.

        The centiMorgan position is not currently used in Hail (Column 3 in BIM file).

        The ID (``s.id``) used by Hail is the individual ID (column 2 in FAM file).

        .. warning::

            No duplicate individual IDs are allowed.

        Chromosome names (Column 1) are automatically converted in the following cases:
        
          - 23 => "X"
          - 24 => "Y"
          - 25 => "X"
          - 26 => "MT"

        **Annotations**

        :py:meth:`~hail.HailContext.import_plink` adds the following annotations:

         - **va.rsid** (*String*) -- Column 2 in the BIM file.
         - **sa.famID** (*String*) -- Column 1 in the FAM file. Set to missing if ID equals "0".
         - **sa.patID** (*String*) -- Column 3 in the FAM file. Set to missing if ID equals "0".
         - **sa.matID** (*String*) -- Column 4 in the FAM file. Set to missing if ID equals "0".
         - **sa.isFemale** (*String*) -- Column 5 in the FAM file. Set to missing if value equals "-9", "0", or "N/A".
           Set to true if value equals "2". Set to false if value equals "1".
         - **sa.isCase** (*String*) -- Column 6 in the FAM file. Only present if ``quantpheno`` equals False.
           Set to missing if value equals "-9", "0", "N/A", or the value specified by ``missing``.
           Set to true if value equals "2". Set to false if value equals "1".
         - **sa.qPheno** (*String*) -- Column 6 in the FAM file. Only present if ``quantpheno`` equals True.
           Set to missing if value equals ``missing``.

        :param str bed: PLINK BED file.

        :param str bim: PLINK BIM file.

        :param str fam: PLINK FAM file.

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param str missing: The string used to denote missing values **only** for the phenotype field. This is in addition to "-9", "0", and "N/A" for case-control phenotypes.

        :param str delimiter: FAM file field delimiter regex.

        :param bool quantpheno: If True, FAM phenotype is interpreted as quantitative.

        :return: A dataset imported from a PLINK binary file.

        :rtype: :class:`.VariantDataset`
        """

        jvds = self._jhc.importPlink(bed, bim, fam, joption(npartitions), delimiter, missing, quantpheno)

        return VariantDataset(self, jvds)

    @handle_py4j
    def read(self, path, sites_only=False, samples_only=False):
        """Read .vds files as VariantDataset

        When loading multiple .vds files, they must have the same
        sample IDs, split status and variant metadata.

        :param path: .vds files to read.
        :type path: str or list of str

        :param bool sites_only: If True, create sites-only
          dataset.  Don't load sample ids, sample annotations
          or gneotypes.

        :param bool samples_only: If True, create samples-only
          dataset (no variants or genotypes).

        :return: A dataset read from disk
        :rtype: :class:`.VariantDataset`
        """

        jvds = self._jhc.readAll(jindexed_seq_args(path), sites_only, samples_only)
        return VariantDataset(self, jvds)

    @handle_py4j
    def write_partitioning(self, path):
        """Write partitioning.json.gz file for legacy VDS file.

        :param str path: path to VDS file.
        """

        self._jhc.writePartitioning(path)

    @handle_py4j
    def import_vcf(self, path, force=False, force_bgz=False, header_file=None, npartitions=None,
                   sites_only=False, store_gq=False, pp_as_pl=False, skip_bad_ad=False):
        """Import VCF file(s) as :py:class:`.VariantDataset`

        **Examples**

        >>> vds = hc.import_vcf('data/example2.vcf.bgz')

        **Notes**

        Hail is designed to be maximally compatible with files in the `VCF v4.2 spec <https://samtools.github.io/hts-specs/VCFv4.2.pdf>`_.

        :py:meth:`~hail.HailContext.import_vcf` takes a list of VCF files to load. All files must have the same header and the same set of samples in the same order
        (e.g., a dataset split by chromosome). Files can be specified as :ref:`Hadoop glob patterns <sec-hadoop-glob>`.

        Ensure that the VCF file is correctly prepared for import: VCFs should either be uncompressed (*.vcf*) or block compressed
        (*.vcf.bgz*).  If you have a large compressed VCF that ends in *.vcf.gz*, it is likely that the file is actually block-compressed,
        and you should rename the file to ".vcf.bgz" accordingly. If you actually have a standard gzipped file, it is possible to import
        it to Hail using the ``force`` optional parameter. However, this is not recommended -- all parsing will have to take place on one node because
        gzip decompression is not parallelizable. In this case, import could take significantly longer.

        Hail makes certain assumptions about the genotype fields, see :class:`Representation <hail.representation.Genotype>`. On import, Hail filters
        (sets to no-call) any genotype that violates these assumptions. Hail interprets the format fields: GT, AD, OD, DP, GQ, PL; all others are
        silently dropped.

        :py:meth:`~hail.HailContext.import_vcf` does not perform deduplication - if the provided VCF(s) contain multiple records with the same chrom, pos, ref, alt, all
        these records will be imported and will not be collapsed into a single variant.

        Since Hail's genotype representation does not yet support ploidy other than 2,
        this method imports haploid genotypes as diploid. Hail fills in missing indices
        in PL / PP arrays with 1000 to support the standard VCF / VDS "genotype schema.

        Below are two example haploid genotypes and diploid equivalents that Hail sees.

        .. code-block:: text

            Haploid:     1:0,6:7:70:70,0
            Imported as: 1/1:0,6:7:70:70,1000,0

            Haploid:     2:0,0,9:9:24:24,40,0
            Imported as: 2/2:0,0,9:9:24:24,1000,40,1000:1000:0


        **Annotations**

        - **va.pass** (*Boolean*) -- true if the variant contains `PASS` in the filter field (false if `.` or other)
        - **va.filters** (*Set[String]*) -- set containing the list of filters applied to a variant. Accessible using `va.filters.contains("VQSRTranche99.5...")`, for example
        - **va.rsid** (*String*) -- rsid of the variant, if it has one ("." otherwise)
        - **va.qual** (*Double*) -- the number in the qual field
        - **va.info** (*T*) -- matches (with proper capitalization) any defined info field. Data types match the type specified in the vcf header, and if the `Number` is "A", "R", or "G", the result will be stored in an array (accessed with array\[index\]).

        :param path: VCF file(s) to read.
        :type path: str or list of str

        :param bool force: If True, load .gz files serially.

        :param bool force_bgz: If True, load .gz files as blocked gzip files (BGZF)

        :param header_file: File to load VCF header from.  If not specified, the first file in path is used.
        :type header_file: str or None

        :param npartitions: Number of partitions.
        :type npartitions: int or None

        :param bool sites_only: If True, create sites-only
            VariantDataset.  Don't load sample ids, sample annotations
            or genotypes.

        :param bool store_gq: If True, store GQ FORMAT field instead of computing from PL.

        :param bool pp_as_pl: If True, store PP FORMAT field as PL.  EXPERIMENTAL.

        :param bool skip_bad_ad: If True, set AD FORMAT field with
            wrong number of elements to missing, rather than setting
            the entire genotype to missing.

        :return: A dataset imported from the VCF file
        :rtype: :py:class:`.VariantDataset`
        """

        jvds = self._jhc.importVCFs(jindexed_seq_args(path), force, force_bgz, joption(header_file),
                                    joption(npartitions), sites_only, store_gq,
                                    pp_as_pl, skip_bad_ad)

        return VariantDataset(self, jvds)

    @handle_py4j
    def index_bgen(self, path):
        """Index .bgen files.  import_bgen cannot run without these indices.

        :param path: .bgen files to index.
        :type path: str or list of str

        """

        self._jhc.indexBgen(jindexed_seq_args(path))

    @handle_py4j
    def balding_nichols_model(self, populations, samples, variants, npartitions=None,
                              pop_dist=None, fst=None, af_dist=UniformDist(0.1, 0.9),
                              seed=0):
        """Generate a VariantDataset using the Balding-Nichols model.

        **Examples**

        To generate a VDS with 3 populations, 100 samples in total, and 1000 variants:

        >>> vds = hc.balding_nichols_model(3, 100, 1000)

        To generate a VDS with 4 populations, 2000 samples, 5000 variants, 10 partitions, population distribution [0.1, 0.2, 0.3, 0.4], :math:`F_st` values [.02, .06, .04, .12], ancestral allele frequencies drawn from a truncated beta distribution with a = .01 and b = .05 over the interval [0.05, 1], and random seed 1:

        >>> from hail.stats import TruncatedBetaDist
        >>> vds = hc.balding_nichols_model(4, 40, 150, 10,
        ...                                pop_dist=[0.1, 0.2, 0.3, 0.4],
        ...                                fst=[.02, .06, .04, .12],
        ...                                af_dist=TruncatedBetaDist(a=0.01, b=2.0, minVal=0.05, maxVal=1.0),
        ...                                seed=1)

        **Notes**

        Hail is able to randomly generate a VDS using the Balding-Nichols model.

        - :math:`K` populations are labeled by integers 0, 1, ..., K - 1
        - :math:`N` samples are named by strings 0, 1, ..., N - 1
        - :math:`M` variants are defined as ``1:1:A:C``, ``1:2:A:C``, ..., ``1:M:A:C``
        - The default ancestral frequency distribution :math:`P_0` is uniform on [0.1, 0.9]. Options are UniformDist(minVal, maxVal), BetaDist(a, b), and TruncatedBetaDist(a, b, minVal, maxVal). All three classes are located in hail.stats.
        - The population distribution :math:`\pi` defaults to uniform
        - The :math:`F_{st}` values default to 0.1
        - The number of partitions defaults to one partition per million genotypes (i.e., samples * variants / 10^6) or 8, whichever is larger

        The Balding-Nichols model models genotypes of individuals from a structured population comprising :math:`K` homogeneous subpopulations
        that have each diverged from a single ancestral population (a `star phylogeny`). We take :math:`N` samples and :math:`M` bi-allelic variants in perfect
        linkage equilibrium. The relative sizes of the subpopulations are given by a probability vector :math:`\pi`; the ancestral allele frequencies are
        drawn independently from a frequency spectrum :math:`P_0`; the subpopulations have diverged with possibly different :math:`F_{ST}` parameters :math:`F_k`
        (here and below, lowercase indices run over a range bounded by the corresponding uppercase parameter, e.g. :math:`k = 1, \ldots, K`).
        For each variant, the subpopulation allele frequencies are drawn a `beta distribution <https://en.wikipedia.org/wiki/Beta_distribution>`_, a useful continuous approximation of
        the effect of genetic drift. We denote the individual subpopulation memberships by :math:`k_n`, the ancestral allele frequences by :math:`p_{0, m}`,
        the subpopulation allele frequencies by :math:`p_{k, m}`, and the genotypes by :math:`g_{n, m}`. The generative model in then given by:

        .. math::
            k_n \,&\sim\, \pi

            p_{0,m}\,&\sim\, P_0

            p_{k,m}\mid p_{0,m}\,&\sim\, \mathrm{Beta}(\mu = p_{0,m},\, \sigma^2 = F_k p_{0,m}(1 - p_{0,m}))

            g_{n,m}\mid k_n, p_{k, m} \,&\sim\, \mathrm{Binomial}(2, p_{k_n, m})

        We have parametrized the beta distribution by its mean and variance; the usual parameters are :math:`a = (1 - p)(1 - F)/F,\; b = p(1-F)/F` with :math:`F = F_k,\; p = p_{0,m}`.

        **Annotations**

        :py:meth:`~hail.HailContext.balding_nichols_model` adds the following global, sample, and variant annotations:

         - **global.nPops** (*Int*) -- Number of populations
         - **global.nSamples** (*Int*) -- Number of samples
         - **global.nVariants** (*Int*) -- Number of variants
         - **global.popDist** (*Array[Double]*) -- Normalized population distribution indexed by population
         - **global.Fst** (*Array[Double]*) -- F_st values indexed by population
         - **global.seed** (*Int*) -- Random seed
         - **global.ancestralAFDist** (*Struct*) -- Information about ancestral allele frequency distribution
         - **sa.pop** (*Int*) -- Population of sample
         - **va.ancestralAF** (*Double*) -- Ancestral allele frequency
         - **va.AF** (*Array[Double]*) -- Allele frequency indexed by population

        :param int populations: Number of populations.

        :param int samples: Number of samples.

        :param int variants: Number of variants.

        :param int npartitions: Number of partitions.

        :param pop_dist: Unnormalized population distribution
        :type pop_dist: array of float or None

        :param fst: F_st values
        :type fst: array of float or None

        :param af_dist: Ancestral allele frequency distribution
        :type af_dist: :class:`.UniformDist` or :class:`.BetaDist` or :class:`.TruncatedBetaDist`

        :param int seed: Random seed.

        :rtype: :class:`.VariantDataset`
        :return: A VariantDataset generated by the Balding-Nichols model.
        """

        if pop_dist is None:
            jvm_pop_dist_opt = joption(pop_dist)
        else:
            jvm_pop_dist_opt = joption(jarray(self._jvm.double, pop_dist))

        if fst is None:
            jvm_fst_opt = joption(fst)
        else:
            jvm_fst_opt = joption(jarray(self._jvm.double, fst))

        jvds = self._jhc.baldingNicholsModel(populations, samples, variants,
                                             joption(npartitions),
                                             jvm_pop_dist_opt,
                                             jvm_fst_opt,
                                             af_dist._jrep(),
                                             seed)
        return VariantDataset(self, jvds)

    @handle_py4j
    def dataframe_to_keytable(self, df, keys=[]):
        """Convert Spark SQL DataFrame to KeyTable.

        Spark SQL data types are converted to Hail types in the obvious way as follows:

        .. code-block:: text

          BooleanType => Boolean
          IntegerType => Int
          LongType => Long
          FloatType => Float
          DoubleType => Double
          StringType => String
          BinaryType => Binary
          ArrayType => Array
          StructType => Struct

        Unlisted Spark SQL data types are currently unsupported.

        :param keys: List of key column names.
        :type keys: list of string

        :return: The DataFrame as a KeyTable.
        :rtype: :class:`.KeyTable`
        """

        jkeys = jarray(self._jvm.java.lang.String, keys)
        return KeyTable(self, self._hail.keytable.KeyTable.fromDF(self._jhc, df._jdf, jkeys))

    def natives(self, n):
        """Test natives.

        :param int n: Dimension.

        """
        return self.hail.driver.NativesCommand.apply(n)

    @handle_py4j
    def eval_expr_typed(self, expr):
        """Evaluate an expression and return the result as well as its type.

        :param str expr: Expression to evaluate.

        :rtype: (annotation, :class:`.Type`)

        """

        x = self._jhc.eval(expr)
        t = Type._from_java(x._2())
        v = t._convert_to_py(x._1())
        return (v, t)

    @handle_py4j
    def eval_expr(self, expr):
        """Evaluate an expression.

        :param str expr: Expression to evaluate.

        :rtype: annotation
        """

        r, t = self.eval_expr_typed(expr)
        return r

    def stop(self):
        """ Shut down the Hail Context """
        self.sc.stop()
        self.sc = None
