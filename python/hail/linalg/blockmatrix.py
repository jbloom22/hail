from hail.utils import new_temp_file, storage_level
from hail.utils.java import Env, jarray, joption, FatalError
from hail.typecheck import *
from hail.matrixtable import MatrixTable
from hail.table import Table
from hail.expr.expressions import expr_numeric
import numpy as np

block_matrix_type = lazy()


class BlockMatrix(object):
    """Hail's block-distributed matrix of :py:data:`.tfloat64` elements.

    .. include:: ../_templates/experimental.rst

    Notes
    -----
    Use ``+``, ``-``, ``*``, and ``/`` for element-wise addition, subtraction,
    multiplication, and division. Each operand may be a block matrix or a scalar
    or type :obj:`int` or :obj:`float`. Block matrix operands must have the same
    shape.

    Use ``**`` for element-wise exponentiation of a block matrix using a power
    of type :obj:`int` or :obj:`float`.

    Use ``@`` for matrix multiplication of block matrices.

    Blocks are square with side length a common block size.
    Blocks in the final block row or block column may be truncated.
    """

    def __init__(self, jbm):
        self._jbm = jbm

    @classmethod
    @typecheck_method(path=str)
    def read(cls, path):
        """Reads a block matrix.

        Parameters
        ----------
        path: :obj:`str`
            Path to input file.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        hc = Env.hc()
        return cls(Env.hail().linalg.BlockMatrix.read(
            hc._jhc, path))

    @classmethod
    @typecheck_method(path=str,
                      n_rows=int,
                      n_cols=int,
                      block_size=nullable(int))
    def fromfile(cls, path, n_rows, n_cols, block_size=None):
        """Creates a block matrix from a binary file.

        Examples
        --------

        >>> import numpy as np
        >>> a = np.random.rand(10, 20)
        >>> a.tofile('output/binary')

        To create a block matrix of the same dimensions:

        >>> from hail.linalg import BlockMatrix
        >>> bm = BlockMatrix.fromfile('output/binary', 10, 20)

        Notes
        -----
        This method, analogous to `numpy.fromfile
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.fromfile.html>`__,
        reads a binary file of float64 values in row-major order, as produced by
        `numpy.fromfile
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.fromfile.html>`__.

        While :meth:`BlockMatrix.fromfile` can also read binary files produced
        by :meth:`BlockMatrix.tofile`, it more efficient and scalable to use
        :meth:`BlockMatrix.write` and :meth:`BlockMatrix.read` to save and
        restore block matrices.

        The number of entries must be less than :math:`2^{31}`.

        See also :meth:`BlockMatrix.from_numpy`.

        Parameters
        ----------
        path: :obj:`str`, optional
            Path for output.
        n_rows: :obj:`int`
            Number of rows.
        n_cols: :obj:`int`
            Number of columns.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`default_block_size`.
        """
        if not block_size:
            block_size = BlockMatrix.default_block_size()

        n_entries = n_rows * n_cols
        if n_entries >= 2 << 31:
            raise FatalError('Number of entries must be less than 2^31, found {}'.format(n_entries))

        hc = Env.hc()
        bdm = Env.hail().utils.richUtils.RichDenseMatrixDouble.readDoubles(hc._jhc, path, n_rows, n_cols, True)

        return cls(Env.hail().linalg.BlockMatrix.fromBreezeMatrix(hc._jsc, bdm, block_size))

    @classmethod
    @typecheck_method(ndarray=np.ndarray,
                      block_size=nullable(int))
    def from_numpy(cls, ndarray, block_size=None):
        """Distributes a `NumPy ndarray
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`__
        as a block matrix.

        Examples
        --------

        >>> import numpy as np
        >>> a = np.random.rand(10, 20)
        >>> bm = BlockMatrix.from_numpy(a)

        Notes
        -----
        The ndarray must have two dimensions, each of non-zero size.

        The example above is implemented as:

        >>> a.tofile('output/binary')
        >>> bm = BlockMatrix.fromfile('output/binary', 10, 20)

        except that the file used to pass data from Python to Java is created
        automatically on the driver's local disk using Python's
        `tempfile.mkstemp
        <https://docs.python.org/3/library/tempfile.html#tempfile.mkstemp>`__
        function. This function places the file in a system temp directory but
        does not automatically delete the file at exit.

        The number of entries must be less than :math:`2^{31}`.

        Parameters
        ----------
        ndarray: :class:`numpy.ndarray`
            ndarray with two dimensions, each of non-zero size.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`default_block_size`.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if not block_size:
            block_size = BlockMatrix.default_block_size()

        if ndarray.ndim != 2:
            raise FatalError("from_numpy: ndarray must have two axes, found shape {}".format(ndarray.shape))
        n_rows, n_cols = ndarray.shape[0:2]
        if n_rows == 0 or n_cols == 0:
            raise FatalError("from_numpy: ndarray dimensions must be non-zero, found shape {}".format(ndarray.shape))
        if ndarray.dtype != np.float64:
            ndarray = ndarray.astype(np.float64)

        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            ndarray.tofile(f.name)

        return cls.fromfile(f.name, n_rows, n_cols, block_size)

    @classmethod
    @typecheck_method(entry_expr=expr_numeric,
                      path=nullable(str),
                      block_size=nullable(int))
    def from_entry_expr(cls, entry_expr, path=None, block_size=None):
        """Creates a block matrix from a numeric matrix table entry expression.

        Notes
        -----
        For shuffle resiliency, this functions writes the resulting block matrix
        to disk and then reads the result. By specifying a path, this block
        matrix can be read again using :meth:`BlockMatrix.read`. Otherwise, a
        file is created in the temp directory given by :func:`hail.tmp_dir` and
        then deleted when :func:`hail.stop` is called or the program terminates.

        Parameters
        ----------
        entry_expr: :class:`.NumericExpression`
            Numeric entry expression for matrix entries.
        path: :obj:`str`, optional
            Path used to write the resulting block matrix.
            If not specified, a temporary file is used.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`default_block_size`.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if not path:
            path = new_temp_file(suffix="bm")
        if not block_size:
            block_size = cls.default_block_size()

        source = entry_expr._indices.source
        if not isinstance(source, MatrixTable):
            raise ValueError("Expect an expression of 'MatrixTable', found {}".format(
                "expression of '{}'".format(source.__class__) if source is not None else 'scalar expression'))

        if entry_expr._indices != source._entry_indices:
            from hail.expr.expressions import ExpressionException
            raise ExpressionException("from_entry_expr: 'entry_expr' must be entry-indexed,"
                                      " found indices {}".format(list(entry_expr._indices.axes)))

        if entry_expr in source._fields_inverse:
            source._jvds.writeBlockMatrix(path, source._fields_inverse[entry_expr], block_size)
        else:
            uid = Env._get_uid()
            source.select_entries(**{uid: entry_expr})._jvds.writeBlockMatrix(path, uid, block_size)

        return cls.read(path)

    @classmethod
    @typecheck_method(n_rows=int,
                      n_cols=int,
                      block_size=nullable(int),
                      seed=int,
                      uniform=bool)
    def random(cls, n_rows, n_cols, block_size=None, seed=0, uniform=True):
        """Creates a block matrix with uniform or normal random entries.

        Parameters
        ----------
        n_rows: :obj:`int`
            Number of rows.
        n_cols: :obj:`int`
            Number of columns.
        block_size: :obj:`int`, optional
            Block size. Default given by :meth:`default_block_size`.
        seed: :obj:`int`
            Random seed.
        uniform: :obj:`bool`
            If ``True``, entries are drawn from the uniform distribution
            on [0,1]. If ``False``, entries are drawn from the standard
            normal distribution.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if not block_size:
            block_size = BlockMatrix.default_block_size()

        hc = Env.hc()
        return cls(Env.hail().linalg.BlockMatrix.random(
            hc._jhc, n_rows, n_cols, block_size, seed, uniform))

    @classmethod
    @typecheck_method(n_rows=int,
                      n_cols=int,
                      data=listof(float),
                      row_major=bool,
                      block_size=int)
    def _create_block_matrix(cls, n_rows, n_cols, data, row_major, block_size):
        """Private method for creating small test matrices."""

        bdm = Env.hail().utils.richUtils.RichDenseMatrixDouble.apply(n_rows,
                                                                     n_cols,
                                                                     jarray(Env.jvm().double, data),
                                                                     row_major)
        hc = Env.hc()
        return cls(Env.hail().linalg.BlockMatrix.fromBreezeMatrix(hc._jsc, bdm, block_size))

    @staticmethod
    def default_block_size():
        """Default block side length."""
        return Env.hail().linalg.BlockMatrix.defaultBlockSize()

    @property
    def n_rows(self):
        """Number of rows.

        Returns
        -------
        :obj:`int`
        """
        return self._jbm.nRows()

    @property
    def n_cols(self):
        """Number of columns.

        Returns
        -------
        :obj:`int`
        """
        return self._jbm.nCols()

    @property
    def block_size(self):
        """Block size.

        Returns
        -------
        :obj:`int`
        """
        return self._jbm.blockSize()

    @typecheck_method(path=str,
                      force_row_major=bool)
    def write(self, path, force_row_major=False):
        """Writes the block matrix.

        Parameters
        ----------
        path: :obj:`str`
            Path for output file.
        force_row_major: :obj:`bool`
            If ``True``, transform blocks in column-major format
            to row-major format before writing.
            If ``False``, write blocks in their current format.
        """
        self._jbm.write(path, force_row_major, joption(None))

    @typecheck_method(rows_to_keep=listof(int))
    def filter_rows(self, rows_to_keep):
        """Filters matrix rows.

        Parameters
        ----------
        rows_to_keep: :obj:`list` of :obj:`int`
            Indices of rows to keep. Must be non-empty and increasing.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.filterRows(jarray(Env.jvm().long, rows_to_keep)))

    @typecheck_method(cols_to_keep=listof(int))
    def filter_cols(self, cols_to_keep):
        """Filters matrix columns.

        Parameters
        ----------
        cols_to_keep: :obj:`list` of :obj:`int`
            Indices of columns to keep. Must be non-empty and increasing.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.filterCols(jarray(Env.jvm().long, cols_to_keep)))

    @typecheck_method(rows_to_keep=listof(int),
                      cols_to_keep=listof(int))
    def filter(self, rows_to_keep, cols_to_keep):
        """Filters matrix rows and columns.

        Notes
        -----
        This method has the same effect as :meth:`BlockMatrix.filter_cols`
        followed by :meth:`BlockMatrix.filter_rows` (or vice versa), but
        filters the block matrix in a single pass which may be more efficient.

        Parameters
        ----------
        rows_to_keep: :obj:`list` of :obj:`int`
            Indices of rows to keep. Must be non-empty and increasing.
        cols_to_keep: :obj:`list` of :obj:`int`
            Indices of columns to keep. Must be non-empty and increasing.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.filter(jarray(Env.jvm().long, rows_to_keep),
                                            jarray(Env.jvm().long, cols_to_keep)))

    @typecheck_method(path=str)
    def tofile(self, path):
        """Collects and writes data to a binary file.

        Examples
        --------
        
        >>> from hail.linalg import BlockMatrix
        >>> import numpy as np
        >>> bm = BlockMatrix.random(10, 20)
        >>> bm.tofile('output/binary')

        To create a :class:`numpy.ndarray` of the same dimensions:

        >>> a = np.fromfile('output/binary').reshape((10, 20))

        Notes
        -----
        The number of entries must be less than :math:`2^{31}`.

        This method, analogous to `numpy.tofile
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tofile.html>`__,
        produces a binary file of float64 values in row-major order, which can
        be read with `numpy.fromfile
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.fromfile.html>`__
        and then reshaped to the desired dimensions.

        While this file can also be read with :meth:`BlockMatrix.fromfile`, it
        more efficient and scalable to use :meth:`BlockMatrix.write` and
        :meth:`BlockMatrix.read` to save and restore block matrices.

        See also :meth:`.to_numpy`.

        Parameters
        ----------
        path: :obj:`str`, optional
            Path for binary output file.
        """
        n_entries = self.n_rows * self.n_cols
        if n_entries >= 2 << 31:
            raise FatalError('Number of entries must be less than 2^31, found {}'.format(n_entries))
        
        bdm = self._jbm.toBreezeMatrix()
        hc = Env.hc()
        row_major = Env.hail().utils.richUtils.RichDenseMatrixDouble.writeDoubles(hc._jhc, path, bdm, True)
        assert row_major

    def to_numpy(self):
        """Collects the block matrix into a `NumPy ndarray
        <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`__.

        Examples
        --------

        >>> from hail.linalg import BlockMatrix
        >>> bm = BlockMatrix.random(10, 20)
        >>> a = bm.to_numpy()

        Notes
        -----
        The number of entries must be less than :math:`2^{31}`.

        The resulting ndarray will have the same shape as the block matrix.

        The example above is implemented as:

        >>> bm.tofile('output/binary')
        >>> a = np.fromfile('output/binary').reshape((10, 20))

        except that the file used to pass data from Java to Python is created
        automatically on the driver's local disk using Python's
        `tempfile.mkstemp
        <https://docs.python.org/3/library/tempfile.html#tempfile.mkstemp>`__
        function. This function places the file in a system temp directory but
        does not automatically delete the file at exit.

        Returns
        -------
        :class:`numpy.ndarray`
        """
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.tofile(f.name)

        return np.fromfile(f.name).reshape((self.n_rows, self.n_cols))

    @property
    def T(self):
        """Matrix transpose.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.transpose())

    def cache(self):
        """Persist this block matrix in memory.

        Notes
        -----
        This method is an alias for :meth:`persist("MEMORY_ONLY") <hail.linalg.BlockMatrix.persist>`.

        Returns
        -------
        :class:`.BlockMatrix`
            Cached block matrix.
        """
        return self.persist('MEMORY_ONLY')

    @typecheck_method(storage_level=storage_level)
    def persist(self, storage_level='MEMORY_AND_DISK'):
        """Persists this block matrix in memory or on disk.

        Notes
        -----
        The :meth:`.BlockMatrix.persist` and :meth:`.BlockMatrix.cache`
        methods store the current block matrix on disk or in memory temporarily
        to avoid redundant computation and improve the performance of Hail
        pipelines. This method is not a substitution for
        :meth:`.BlockMatrix.write`, which stores a permanent file.

        Most users should use the "MEMORY_AND_DISK" storage level. See the `Spark
        documentation
        <http://spark.apache.org/docs/latest/programming-guide.html#rdd-persistence>`__
        for a more in-depth discussion of persisting data.

        Parameters
        ----------
        storage_level : str
            Storage level.  One of: NONE, DISK_ONLY,
            DISK_ONLY_2, MEMORY_ONLY, MEMORY_ONLY_2, MEMORY_ONLY_SER,
            MEMORY_ONLY_SER_2, MEMORY_AND_DISK, MEMORY_AND_DISK_2,
            MEMORY_AND_DISK_SER, MEMORY_AND_DISK_SER_2, OFF_HEAP

        Returns
        -------
        :class:`.BlockMatrix`
            Persisted block matrix.
        """
        return BlockMatrix(self._jbm.persist(storage_level))

    def unpersist(self):
        """Unpersists this block matrix from memory/disk.

        Notes
        -----
        This function will have no effect on a dataset that was not previously
        persisted.

        Returns
        -------
        :class:`.BlockMatrix`
            Unpersisted block matrix.
        """
        return BlockMatrix(self._jbm.unpersist())

    def __pos__(self):
        return self

    def __neg__(self):
        """Negation: -a.

        Returns
        -------
        :class:`.BlockMatrix`
        """

        op = getattr(self._jbm, "unary_$minus")
        return BlockMatrix(op())

    @typecheck_method(b=oneof(numeric, block_matrix_type))
    def __add__(self, b):
        """Addition: a + b.

        Parameters
        ----------
        b: :class:`BlockMatrix` or :obj:`int` or :obj:`float`

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if isinstance(b, int) or isinstance(b, float):
            return BlockMatrix(self._jbm.scalarAdd(float(b)))
        else:
            return BlockMatrix(self._jbm.add(b._jbm))

    @typecheck_method(b=numeric)
    def __radd__(self, b):
        return self + b

    @typecheck_method(b=oneof(numeric, block_matrix_type))
    def __sub__(self, b):
        """Subtraction: a - b.

        Parameters
        ----------
        b: :class:`BlockMatrix` or :obj:`int` or :obj:`float`

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if isinstance(b, int) or isinstance(b, float):
            return BlockMatrix(self._jbm.scalarSubtract(float(b)))
        else:
            return BlockMatrix(self._jbm.subtract(b._jbm))

    @typecheck_method(b=numeric)
    def __rsub__(self, b):
        return BlockMatrix(self._jbm.reverseScalarSubtract(float(b)))

    @typecheck_method(b=oneof(numeric, block_matrix_type))
    def __mul__(self, b):
        """Element-wise multiplication: a * b.

        Parameters
        ----------
        b: :class:`BlockMatrix` or :obj:`int` or :obj:`float`

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if isinstance(b, int) or isinstance(b, float):
            return BlockMatrix(self._jbm.scalarMultiply(float(b)))
        else:
            return BlockMatrix(self._jbm.pointwiseMultiply(b._jbm))

    @typecheck_method(b=numeric)
    def __rmul__(self, b):
        return self * b

    @typecheck_method(b=oneof(numeric, block_matrix_type))
    def __truediv__(self, b):
        """Element-wise division: a / b.

        Parameters
        ----------
        b: :class:`BlockMatrix` or :obj:`int` or :obj:`float`

        Returns
        -------
        :class:`.BlockMatrix`
        """
        if isinstance(b, int) or isinstance(b, float):
            return BlockMatrix(self._jbm.scalarDivide(float(b)))
        else:
            return BlockMatrix(self._jbm.pointwiseDivide(b._jbm))

    @typecheck_method(b=numeric)
    def __rtruediv__(self, b):
        return BlockMatrix(self._jbm.reverseScalarDivide(float(b)))

    @typecheck_method(b=block_matrix_type)
    def __matmul__(self, b):
        """Matrix multiplication: a @ b.

        Parameters
        ----------
        b: :class:`BlockMatrix`

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.multiply(b._jbm))

    @typecheck_method(x=numeric)
    def __pow__(self, x):
        """Element-wise exponentiation: a ** x.

        Parameters
        ----------
        x: :obj:`int` or :obj:`float`
            Exponent.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.pow(float(x)))

    def sqrt(self):
        """Element-wise square root.

        Returns
        -------
        :class:`.BlockMatrix`
        """
        return BlockMatrix(self._jbm.sqrt())

    def entries(self):
        """Returns a table with the coordinates and numeric value of each block matrix entry.

        Examples
        --------

        >>> from hail.linalg import BlockMatrix
        >>> import numpy as np
        >>> block_matrix = BlockMatrix.from_numpy(np.matrix([[5, 7], [2, 8]]), 2)
        >>> entries_table = block_matrix.entries()
        >>> entries_table.show()
        +-------+-------+-------------+
        |     i |     j |       entry |
        +-------+-------+-------------+
        | int64 | int64 |     float64 |
        +-------+-------+-------------+
        |     0 |     0 | 5.00000e+00 |
        |     0 |     1 | 7.00000e+00 |
        |     1 |     0 | 2.00000e+00 |
        |     1 |     1 | 8.00000e+00 |
        +-------+-------+-------------+

        Warning
        -------
        The resulting table may be filtered, aggregated, and queried, but should only be
        directly exported to disk if the block matrix is very small.

        Returns
        -------
        :class:`.Table`
            Table with a row for each entry.
        """
        hc = Env.hc()
        return Table(self._jbm.entriesTable(hc._jhc))


block_matrix_type.set(BlockMatrix)
