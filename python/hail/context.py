from pyspark import SparkContext
from pyspark.sql import SQLContext

from hail.genetics.reference_genome import ReferenceGenome
from hail.typecheck import nullable, typecheck, typecheck_method, enumeration
from hail.utils import wrap_to_list, get_env_or_default
from hail.utils.java import Env, joption, FatalError, connect_logger, install_exception_handler, uninstall_exception_handler

import sys


class HailContext(object):
    @typecheck_method(sc=nullable(SparkContext),
                      app_name=str,
                      master=nullable(str),
                      local=str,
                      log=str,
                      quiet=bool,
                      append=bool,
                      min_block_size=int,
                      branching_factor=int,
                      tmp_dir=nullable(str),
                      default_reference=str)
    def __init__(self, sc=None, app_name="Hail", master=None, local='local[*]',
                 log='hail.log', quiet=False, append=False,
                 min_block_size=1, branching_factor=50, tmp_dir=None,
                 default_reference="GRCh37"):

        if Env._hc:
            raise FatalError('Hail Context has already been created, restart session '
                             'or stop Hail context to change configuration.')

        SparkContext._ensure_initialized()

        self._gateway = SparkContext._gateway
        self._jvm = SparkContext._jvm

        # hail package
        self._hail = getattr(self._jvm, 'is').hail

        Env._jvm = self._jvm
        Env._gateway = self._gateway

        jsc = sc._jsc.sc() if sc else None

        tmp_dir = get_env_or_default(tmp_dir, 'TMPDIR', '/tmp')

        # we always pass 'quiet' to the JVM because stderr output needs
        # to be routed through Python separately.
        self._jhc = self._hail.HailContext.apply(
            jsc, app_name, joption(master), local, log, True, append,
            min_block_size, branching_factor, tmp_dir)

        self._jsc = self._jhc.sc()
        self.sc = sc if sc else SparkContext(gateway=self._gateway, jsc=self._jvm.JavaSparkContext(self._jsc))
        self._jsql_context = self._jhc.sqlContext()
        self._sql_context = SQLContext(self.sc, jsqlContext=self._jsql_context)
        self._counter = 1

        super(HailContext, self).__init__()

        # do this at the end in case something errors, so we don't raise the above error without a real HC
        Env._hc = self

        self._default_ref = None
        Env.hail().variant.ReferenceGenome.setDefaultReference(self._jhc, default_reference)

        if not quiet:
            sys.stderr.write('Running on Apache Spark version {}\n'.format(self.sc.version))
            if self._jsc.uiWebUrl().isDefined():
                sys.stderr.write('SparkUI available at {}\n'.format(self._jsc.uiWebUrl().get()))

            connect_logger('localhost', 12888)

            sys.stderr.write(
                'Welcome to\n'
                '     __  __     <>__\n'
                '    / /_/ /__  __/ /\n'
                '   / __  / _ `/ / /\n'
                '  /_/ /_/\_,_/_/_/   version {}\n'.format(self.version))

            if self.version.startswith('devel'):
                sys.stderr.write('WARNING: This is an unstable development build.\n')

        install_exception_handler()


    @property
    def version(self):
        return self._jhc.version()

    @property
    def default_reference(self):
        if not self._default_ref:
            self._default_ref = ReferenceGenome._from_java(Env.hail().variant.ReferenceGenome.defaultReference())
        return self._default_ref

    def stop(self):
        self.sc.stop()
        self.sc = None
        Env._jvm = None
        Env._gateway = None
        Env._hc = None
        uninstall_exception_handler()
        Env._dummy_table = None

@typecheck(sc=nullable(SparkContext),
           app_name=str,
           master=nullable(str),
           local=str,
           log=str,
           quiet=bool,
           append=bool,
           min_block_size=int,
           branching_factor=int,
           tmp_dir=str,
           default_reference=enumeration('GRCh37', 'GRCh38'))
def init(sc=None, app_name='Hail', master=None, local='local[*]',
             log='hail.log', quiet=False, append=False,
             min_block_size=1, branching_factor=50, tmp_dir='/tmp',
             default_reference='GRCh37'):
    """Initialize Hail and Spark.

    Parameters
    ----------
    sc : pyspark.SparkContext, optional
        Spark context. By default, a Spark context will be created.
    app_name : :obj:`str`
        Spark application name.
    master : :obj:`str`
        Spark master.
    local : :obj:`str`
       Local-mode master, used if `master` is not defined here or in the
       Spark configuration.
    log : :obj:`str`
        Local path for Hail log file. Does not currently support distributed
        file systems like Google Storage, S3, or HDFS.
    quiet : :obj:`bool`
        Print fewer log messages.
    append : :obj:`bool`
        Append to the end of the log file.
    min_block_size : :obj:`int`
        Minimum file block size in MB.
    branching_factor : :obj:`int`
        Branching factor for tree aggregation.
    tmp_dir : :obj:`str`
        Temporary directory for Hail files. Must be a network-visible
        file path.
    default_reference : :obj:`str`
        Default reference genome. Either ``'GRCh37'`` or ``'GRCh38'``.
    """
    HailContext(sc, app_name, master, local, log, quiet, append,
                min_block_size, branching_factor, tmp_dir, default_reference)

def stop():
    """Stop the currently running Hail session."""
    if Env._hc:
        Env.hc().stop()

def spark_context():
    """Returns the active Spark context.

    Returns
    -------
    :class:`pyspark.SparkContext`
    """
    return Env.hc().sc

def default_reference():
    """Returns the default reference genome ``'GRCh37'``.

    Returns
    -------
    :class:`.ReferenceGenome`
    """
    return Env.hc().default_reference

def get_reference(name):
    """Returns the reference genome corresponding to `name`.

    Notes
    -----

    Hail's built-in references are ``'GRCh37'`` and ``GRCh38'``. The contig names
    and lengths come from the GATK resource bundle:
    `human_g1k_v37.dict <ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/b37/human_g1k_v37.dict>`__
    and `Homo_sapiens_assembly38.dict <ftp://gsapubftp-anonymous@ftp.broadinstitute.org/bundle/hg38/Homo_sapiens_assembly38.dict>`__.

    If ``name='default'``, the value of :func:`.default_reference` is returned.

    Parameters
    ----------
    name : :obj:`str`
        Name of a previously loaded reference genome or one of Hail's built-in
        references: ``'GRCh37'``, ``'GRCh38'``, and ``'default'``.

    Returns
    -------
    :class:`.ReferenceGenome`
    """
    from hail import ReferenceGenome

    if name == 'default':
        return default_reference()
    else:
        return ReferenceGenome._references.get(
            name,
            ReferenceGenome._from_java(Env.hail().variant.ReferenceGenome.getReference(name))
        )

@property
def tmp_dir():
    """Returns the path of the temporary directory."""

    return Env.hc()._jhc.tmpDir()
