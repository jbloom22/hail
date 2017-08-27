from hail.typecheck import *
from hail.java import *

class Eigen:
    """
    Represents the eigenvectors and eigenvalues of a matrix.
    """
    def __init__(self, jeigen):
        self._jeigen = jeigen
        self._key_schema = None
    
    @property
    def key_schema(self):
        """Returns the signature of the key indexing the rows.

        :rtype: :class:`.Type`
        """

        if self._key_schema is None:
            self._key_schema = Type._from_java(self._jeigen.rowSignature())
        return self._key_schema
    
    def sample_list(self):
        """Gets the list of samples.

        :return: List of samples.
        :rtype: list of str
        """
        return [self.key_schema._convert_to_py(s) for s in self._jeigen.rowIds()]

    def evects(self):
        """Gets the matrix whose columns are eigenvectors, ordered by increasing eigenvalue.
                
        :return: Matrix of whose columns are eigenvectors.
        :rtype: `Matrix <https://spark.apache.org/docs/2.1.0/api/python/pyspark.mllib.html#pyspark.mllib.linalg.Matrix>`__
        """
        from pyspark.mllib.linalg import DenseMatrix

        j_evects = self._jeigen.evectsSpark()
        return DenseMatrix(j_evects.numRows(), j_evects.numCols(), list(j_evects.values()), j_evects.isTransposed())


    def evals(self):
        """Gets the eigenvalues.

        :return: List of eigenvalues in increasing order.
        :rtype: list of float
        """
        return list(self._jeigen.evalsArray())
    
    def num_evects(self):
        """Gets the number of eigenvectors and eigenvalues.
        
        :return: Number of eigenvectors and eigenvalues.
        :rtype: int
        """
        return self._jeigen.nEvects()
    
    @typecheck_method(k=integral)
    def take_top(self, k):
        """Take the top k eigenvectors and eigenvalues.
        If k is greater than or equal to the number present, then the calling eigendecomposition is returned.

        :param int k: Number of eigenvectors and eigenvalues to return.

        :return: The top k eigenvectors and eigenvalues.
        :rtype: :py:class:`.Eigen`
        """
        
        return Eigen(self._jeigen.takeTop(k))
    
    def distribute(self):
        """Convert to a distributed eigendecomposition.
        
        :return: Distributed eigendecomposition.
        :rtype: :py:class:`.EigenDistributed`
        """
        
        return EigenDistributed(self._jeigen.distribute(Env.hc()._jsc))
    
    @typecheck_method(vds=anytype,
                      num_samples_in_ld_matrix=integral)
    def to_eigen_distributed_rrm(self, vds, num_samples_in_ld_matrix):
        """
        Compute an eigendecomposition of the realized relationship matrix (RRM) of the variant dataset from an
        eigendecomposition of an LD matrix.
        
        *Notes*

        This method transforms an local eigendecomposition of the LD matrix to a distributed eigendecomposition
        of the corresponding realized relationship matrix.
        
        The variant dataset must include all variants represented in the eigendecomposition of the LD matrix.
        
        Filtering to just these variants, let :math:`G` be the genotype matrix with columns normalized to have
        mean 0 and variance 1. This matrix has :math:`n` rows and :math:`m` columns, the number of samples and variants,
        respectively. Let :math:`K` and :math:`L` be the corresponding RRM and LD matrix, respectively.
        Then by definition:

        .. math::
          
          \\begin{align*}
          K &= \\frac{1}{m} G G^T //
          L &= \\frac{1}{n} G^T G
          \\end{align*}
        
        The singular value decomposition of :math:`G` and the eigendecompositions of :math:`K` and :math:`L` are related
        by
        
        .. math::
        
          \\begin{align*}
          G &= U S^{\\frac{1}{2}} V.t //
          K &= \\frac{1}{m} U S U.t //
          L &= \\frac{1}{n} V S V.t
          \\end{align*}
          
        where the diagonal matrix :math:`S` is of the necessary dimension in each case, extended by zeroes.
          
        In particular, given :math:`V` and :math:`\\frac{1}{n} S` (whose columns and diagonal are the eigenvectors and
        eigenvalues, respectively, of the LD matrix) and :math:`n` (the number of samples used to form the LD
        matrix), we can solve for the top :math:`m` eigenvectors and eigenvalues of the RRM by:
        
        .. math::
        
          \\begin{align*}
          U_m &= G * V * (\\frac{m}{n} S_m)^{-\\frac{1}{2}} \\
          S_m &= \\frac{1}{m} S
          \\end{align*}
        
        These emcompass all non-zero eigenvalues of the RRM.
        
        :param vds: Variant dataset
        :type vds: :py:class:`.VariantDataset`
        
        :param int num_samples_in_ld_matrix: Number of samples used to form the LD matrix.
        
        :return: Distributed eigendecomposition of the realized relationship matrix.
        :rtype: :py:class:`.EigenDistributed`
        """
        
        return EigenDistributed(self._jeigen.toEigenDistributedRRM(vds._jvds, num_samples_in_ld_matrix))
    
    @typecheck_method(path=strlike)
    def write(self, path):
        """
        Writes the eigendecomposition to a directory enging in ``.eig``.

        **Examples**

        >>> vds.rrm().eigen().write('output/example.eig')

        :param str path: path to which to write the eigendecomposition
        """

        self._jeigen.write(Env.hc()._jhc, path)
        
    @staticmethod
    def read(path):
        """
        Reads the eigendecomposition from a directory ending in ``.eig``.

        **Examples**

        >>>  eigen = Eigen.read('data/example.eig')

        :param str path: path from which to read the LD matrix
        
        :return: Eigendecomposition
        :rtype: :py:class:`.Eigen`
        """

        jeigen = Env.hail().stats.Eigen.read(Env.hc()._jhc, path)
        return Eigen(jeigen)

class EigenDistributed:
    """
    Represents the eigenvectors and eigenvalues of a matrix. Eigenvectors are stored as columns of a distributed matrix.
    """
    def __init__(self, jeigen):
        self._jeigen = jeigen
        self._key_schema = None
    
    @property
    def key_schema(self):
        """Returns the signature of the key indexing the rows.

        :rtype: :class:`.Type`
        """

        if self._key_schema is None:
            self._key_schema = Type._from_java(self._jeigen.rowSignature())
        return self._key_schema
    
    def sample_list(self):
        """Gets the list of samples.

        :return: List of samples.
        :rtype: list of str
        """
        return [self.key_schema._convert_to_py(s) for s in self._jeigen.rowIds()]

    def evects(self):
        """Gets the block matrix whose columns are eigenvectors, ordered by increasing eigenvalue.
                
        :return: Matrix whose columns are eigenvectors.
        :rtype: `BlockMatrix <https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.linalg.distributed.BlockMatrix>`__
        """
        from pyspark.mllib.linalg.distributed import BlockMatrix

        return BlockMatrix(self._jeigen.evects())

    def evals(self):
        """Gets the eigenvalues.

        :return: List of eigenvalues in increasing order.
        :rtype: list of float
        """
        return list(self._jeigen.evalsArray())
    
    def num_evects(self):
        """Gets the number of eigenvectors and eigenvalues.
        
        :return: Number of eigenvectors and eigenvalues.
        :rtype: int
        """
        return self._jeigen.nEvects()
