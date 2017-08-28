from hail.typecheck import *
from hail.java import *
from hail.expr import Type

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
    
    def row_ids(self):
        """Gets the list of row IDs.

        :return: List of row IDs of type key_schema
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
        
       **Notes** 
       
        If k is greater than or equal to the number present, then the calling eigendecomposition is returned.

        :param int k: Number of eigenvectors and eigenvalues to return.

        :return: The top k eigenvectors and eigenvalues.
        :rtype: :py:class:`.Eigen`
        """
        
        return Eigen(self._jeigen.takeTop(k))
    
    @typecheck_method(proportion=numeric)
    def drop_small(self, proportion = 1e-6):
        """Drop the maximum number of eigenvectors without losing more than ``proportion`` of the total variance (sum of
        eigenvalues).
        
        **Notes**
        
        For example, if the eigenvalues are [1.0, 2.0, 97.0] then the proportions 0.01, 0.02, and 0.03 will
        drop 1, 1, and 2 eigenvectors, respectively.

        :param float proportion: Proportion in the interval [0,1)

        :return: Eigendecomposition
        :rtype: :py:class:`.Eigen`
        """
        
        return Eigen(self._jeigen.dropSmall(proportion))
    
    def distribute(self):
        """Convert to a distributed eigendecomposition.
        
        :return: Distributed eigendecomposition.
        :rtype: :py:class:`.EigenDistributed`
        """
        
        return EigenDistributed(self._jeigen.distribute(Env.hc()._jsc))
    
    @typecheck_method(vds=anytype,
                      num_samples_in_ld_matrix=integral)
    def to_eigen_distributed_rrm(self, vds, num_samples_in_ld_matrix):
        """Compute eigendecomposition of the RRM from eigendecomposition of LD matrix.
        
        **Examples**
        
        Suppose the variant dataset saved at *data/example_lmmreg.vds* has Boolean variant annotations
        ``va.useInKinship``. To compute a full-rank distributed eigendecomposition of the realized relationship
        matrix:

        >>> vds1 = hc.read("data/example_lmmreg.vds")
        >>> ld = vds1.filter_variants_expr('va.useInKinship').ld_matrix()
        >>> eig = ld.eigen().to_eigen_distributed_rrm(vds1, ld.num_samples_used())
                
        **Notes**

        When run on the rank :math:`k` eigendecomposition of the LD matrix computed from a subset of the variants
        in the variant dataset ``vds`` using ``num_samples_in_ld_matrix`` samples, this method returns the
        the rank :math:`k` distributed eigendecomposition of the realized relationship matrix (RRM) built
        from the same variants.

        The following route to the variable ``eig`` is mathematically equivalent to the example above:

        >>> vds1 = hc.read("data/example_lmmreg.vds")
        >>> km = vds1.filter_variants_expr('va.useInKinship').rrm()
        >>> eig = km.eigen().distribute()

        However these implementations differ in their scalability: the LD route in the example is variant-limited
        since ``ld.eigen()`` runs on the local LD matrix, whereas the more direct RRM route is sample-limited (since
        ``km.eigen()`` runs on the local kinship matrix). In particular, only this method handles the case of more than
        32k samples.
        
        **Details**
        
        This method uses distributed matrix multiplication by the normalized genotype matrix to convert
        variant-indexed right singular vectors to sample-indexed left singular vectors. In particular, the variant
        dataset must include all variants represented in the eigendecomposition of the LD matrix. Filtering
        to just these variants, let :math:`G` be the genotype matrix with columns normalized to have
        mean 0 and variance 1. :math:`G` has :math:`n` rows and :math:`m` columns corresponding to samples and variants,
        respectively. Let :math:`K` and :math:`L` be the corresponding RRM and LD matrix, respectively.
        Then by definition,

        .. math::
          
          \\begin{align*}
          K &= \\frac{1}{m} G G^T \\\\
          L &= \\frac{1}{n} G^T G
          \\end{align*}
        
        The singular value decomposition of :math:`G` and the eigendecompositions of :math:`K` and :math:`L` are related
        by
        
        .. math::
        
          \\begin{align*}
          G &= U S^{\\frac{1}{2}} V^T \\\\
          K &= U \\left(\\frac{1}{m} S \\right) U^T \\\\
          L &= V \\left(\\frac{1}{n} S \\right) V^T
          \\end{align*}
          
        where the diagonal matrix :math:`S` is of the necessary dimension in each case, extended by zeroes.
          
        In particular, given :math:`V_k` and :math:`S_k` (whose columns and diagonal are
        the top :math:`k` eigenvectors and eigenvalues, respectively, of the LD matrix), then the top
        :math:`k` eigenvectors and eigenvalues of the RRM are given by
        
        .. math::
        
          \\begin{align*}
          U_k &= G V \\left(\\frac{n}{m} S_k\\right)^{-\\frac{1}{2}} \\\\
          S^{\\prime}_k &= \\frac{n}{m} S_k
          \\end{align*}
        
        :param vds: Variant dataset
        :type vds: :py:class:`.VariantDataset`
        
        :param int num_samples_in_ld_matrix: Number of samples used to form the LD matrix.
        
        :return: Distributed eigendecomposition of the realized relationship matrix.
        :rtype: :py:class:`.EigenDistributed`
        """
        
        return EigenDistributed(self._jeigen.toEigenDistributedRRM(vds._jvds, num_samples_in_ld_matrix))
    
    @typecheck_method(path=strlike)
    def write(self, path):
        """Writes the eigendecomposition to a path.

        >>> vds.rrm().eigen().write('output/example.eig')

        :param str path: path to directory ending in ``.eig`` to which to write the eigendecomposition
        """

        self._jeigen.write(Env.hc()._jhc, path)
        
    @staticmethod
    def read(path):
        """Reads the eigendecomposition from a path.

        >>> eig = Eigen.read('data/example.eig')

        :param str path: path to directory ending in ``.eig`` from which to read the LD matrix
        
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
    
    def row_ids(self):
        """Gets the list of row IDs.

        :return: List of rows.
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
    
    @typecheck_method(path=strlike,
                      vds=anytype,
                      y=nullable(strlike),
                      covariates=listof(strlike),
                      use_dosages=bool)
    def project_and_write(self, path, vds, y=None, covariates=[], use_dosages=False):
        """Project complete samples of vds using eigenvectors and write to disk.
        
        >>> eig = vds.rrm().eigen().distribute()
        >>> eig.project_and_write('output/example.proj', vds)
        """
        return self._jeigen.projectAndWrite(path, vds._jvds, joption(y), jarray(Env.jvm().java.lang.String, covariates), use_dosages)
    
    @typecheck_method(path=strlike)
    def write(self, path):
        """Writes the eigendecomposition to a path.

        >>> vds.rrm().eigen().distribute().write('output/example.eigd')

        :param str path: path to directory ending in ``.eigd`` to which to write the eigendecomposition
        """

        self._jeigen.write(path)
        
    @staticmethod
    def read(path):
        """Reads the eigendecomposition from a path.

        >>> eig = EigenDistributed.read('data/example.eigd')

        :param str path: path to directory ending in ``.eigd`` from which to read the LD matrix
        
        :return: Eigendecomposition
        :rtype: :py:class:`.EigenDistributed`
        """

        jeigen = Env.hail().stats.EigenDistributed.read(Env.hc()._jhc, path)
        return EigenDistributed(jeigen)
