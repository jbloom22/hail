from pyspark.sql import DataFrame
import hail as hl
from hail.expr.expression import *
from hail.utils import wrap_to_list, storage_level
from hail.utils.java import jiterable_to_list
from hail.utils.misc import get_nice_field_error, get_nice_attr_error, check_collisions

table_type = lazy()


class Ascending(object):
    def __init__(self, col):
        self.col = col

    def _j_obj(self):
        return scala_package_object(Env.hail().table).asc(self.col)


class Descending(object):
    def __init__(self, col):
        self.col = col

    def _j_obj(self):
        return scala_package_object(Env.hail().table).desc(self.col)


@typecheck(col=oneof(Expression, strlike))
def asc(col):
    """Sort by `col` ascending."""

    return Ascending(col)


@typecheck(col=oneof(Expression, strlike))
def desc(col):
    """Sort by `col` descending."""

    return Descending(col)


class TableTemplate(HistoryMixin):
    def __init__(self, jt):
        self._jt = jt

        self._globals = None
        self._global_schema = None
        self._schema = None
        self._num_columns = None
        self._key = None
        self._column_names = None
        self._fields = {}
        super(TableTemplate, self).__init__()

    def _set_field(self, key, value):
        self._fields[key] = value
        if key in dir(self):
            warn("Name collision: field '{}' already in object dict. "
                 "This field must be referenced with indexing syntax".format(key))
        else:
            self.__dict__[key] = value

    def _get_field(self, item):
        if item in self._fields:
            return self._fields[item]
        else:
            raise LookupError(get_nice_field_error(self, item))

    def __getitem__(self, item):
        return self._get_field(item)

    def __delattr__(self, item):
        if not item[0] == '_':
            raise NotImplementedError('Table objects are not mutable')

    def __setattr__(self, key, value):
        if not key[0] == '_':
            raise NotImplementedError('Table objects are not mutable')
        self.__dict__[key] = value

    def __getattr__(self, item):
        if item in self.__dict__:
            return self.__dict__[item]
        else:
            raise AttributeError(get_nice_attr_error(self, item))

    def __repr__(self):
        return self._jt.toString()

    @handle_py4j
    def get_globals(self):
        if self._globals is None:
            self._globals = self.global_schema._convert_to_py(self._jt.globals())
        return self._globals

    @property
    @handle_py4j
    def schema(self):
        if self._schema is None:
            self._schema = Type._from_java(self._jt.signature())
            assert (isinstance(self._schema, TStruct))
        return self._schema

    @property
    @handle_py4j
    def global_schema(self):
        if self._global_schema is None:
            self._global_schema = Type._from_java(self._jt.globalSignature())
            assert (isinstance(self._global_schema, TStruct))
        return self._global_schema

    @property
    @handle_py4j
    def key(self):
        if self._key is None:
            self._key = jiterable_to_list(self._jt.key())
        return self._key


class GroupedTable(TableTemplate):
    """Table that has been grouped.

    There are only two operations on a grouped table, :meth:`.GroupedTable.partition_hint`
    and :meth:`.GroupedTable.aggregate`.

    .. testsetup ::

        table1 = hl.import_table('data/kt_example1.tsv', impute=True, key='ID')

    """

    def __init__(self, parent, groups):
        super(GroupedTable, self).__init__(parent._jt)
        self._groups = groups
        self._parent = parent
        self._npartitions = None

        for fd in parent._fields:
            self._set_field(fd, parent._fields[fd])

    @handle_py4j
    @typecheck_method(n=integral)
    def partition_hint(self, n):
        """Set the target number of partitions for aggregation.

        Examples
        --------

        Use `partition_hint` in a :meth:`.Table.group_by` / :meth:`.GroupedTable.aggregate`
        pipeline:

        >>> table_result = (table1.group_by(table1.ID)
        ...                       .partition_hint(5)
        ...                       .aggregate(meanX = agg.mean(table1.X), sumZ = agg.sum(table1.Z)))

        Notes
        -----
        Until Hail's query optimizer is intelligent enough to sample records at all
        stages of a pipeline, it can be necessary in some places to provide some
        explicit hints.

        The default number of partitions for :meth:`.GroupedTable.aggregate` is the
        number of partitions in the upstream table. If the aggregation greatly
        reduces the size of the table, providing a hint for the target number of
        partitions can accelerate downstream operations.

        Parameters
        ----------
        n : int
            Number of partitions.

        Returns
        -------
        :class:`.GroupedTable`
            Same grouped table with a partition hint.
        """
        self._npartitions = n
        return self

    @handle_py4j
    def aggregate(self, **named_exprs):
        """Aggregate by group, used after :meth:`.Table.group_by`.

        Examples
        --------
        Compute the mean value of `X` and the sum of `Z` per unique `ID`:

        >>> table_result = (table1.group_by(table1.ID)
        ...                       .aggregate(meanX = agg.mean(table1.X), sumZ = agg.sum(table1.Z)))

        Group by a height bin and compute sex ratio per bin:

        >>> table_result = (table1.group_by(height_bin = (table1.HT / 20).to_int32())
        ...                       .aggregate(fraction_female = agg.fraction(table1.SEX == 'F')))

        Parameters
        ----------
        named_exprs : varargs of :class:`.Expression`
            Aggregation expressions.

        Returns
        -------
        :class:`.Table`
            Aggregated table.
        """
        agg_base = self._parent.columns[0]  # FIXME hack

        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}

        strs = []
        base, cleanup = self._parent._process_joins(*([v for _, v in self._groups] + named_exprs.values()))
        for k, v in named_exprs.items():
            analyze('GroupedTable.aggregate', v, self._parent._global_indices, {self._parent._row_axis})
            replace_aggregables(v._ast, agg_base)
            strs.append('{} = {}'.format(escape_id(k), v._ast.to_hql()))

        group_strs = ',\n'.join('{} = {}'.format(escape_id(k), v._ast.to_hql()) for k, v in self._groups)
        return cleanup(
            Table(base._jt.aggregate(group_strs, ",\n".join(strs), joption(self._npartitions))))


class Table(TableTemplate):
    """Hail's distributed implementation of a dataframe or SQL table.
    
    Use :func:`.import_table` to import a table from a text file and :func:`.read_table` to read a
    table that was written with :meth:`.Table.write`.

    In the examples below, we import two tables from text files (``table1`` and ``table2``).

    >>> table1 = hl.import_table('data/kt_example1.tsv', impute=True, key='ID')
    >>> table1.show()

    .. code-block:: text

        +-------+-------+--------+-------+-------+-------+-------+-------+
        |    ID |    HT | SEX    |     X |     Z |    C1 |    C2 |    C3 |
        +-------+-------+--------+-------+-------+-------+-------+-------+
        | Int32 | Int32 | String | Int32 | Int32 | Int32 | Int32 | Int32 |
        +-------+-------+--------+-------+-------+-------+-------+-------+
        |     1 |    65 | M      |     5 |     4 |     2 |    50 |     5 |
        |     2 |    72 | M      |     6 |     3 |     2 |    61 |     1 |
        |     3 |    70 | F      |     7 |     3 |    10 |    81 |    -5 |
        |     4 |    60 | F      |     8 |     2 |    11 |    90 |   -10 |
        +-------+-------+--------+-------+-------+-------+-------+-------+

    >>> table2 = hl.import_table('data/kt_example2.tsv', impute=True, key='ID')
    >>> table2.show()

    .. code-block:: text

        +-------+-------+--------+
        |    ID |     A | B      |
        +-------+-------+--------+
        | Int32 | Int32 | String |
        +-------+-------+--------+
        |     1 |    65 | cat    |
        |     2 |    72 | dog    |
        |     3 |    70 | mouse  |
        |     4 |    60 | rabbit |
        +-------+-------+--------+

    Define new annotations:

    >>> height_mean_m = 68
    >>> height_sd_m = 3
    >>> height_mean_f = 65
    >>> height_sd_f = 2.5
    >>>
    >>> def get_z(height, sex):
    ...    return hl.cond(sex == 'M',
    ...                  (height - height_mean_m) / height_sd_m,
    ...                  (height - height_mean_f) / height_sd_f)
    >>>
    >>> table1 = table1.annotate(height_z = get_z(table1.HT, table1.SEX))
    >>> table1 = table1.annotate_globals(global_field_1 = [1, 2, 3])

    Filter rows of the table:

    >>> table2 = table2.filter(table2.B != 'rabbit')

    Compute global aggregation statistics:

    >>> t1_stats = table1.aggregate(Struct(mean_c1 = agg.mean(table1.C1),
    ...                                    mean_c2 = agg.mean(table1.C2),
    ...                                    stats_c3 = agg.stats(table1.C3)))
    >>> print(t1_stats)

    Group columns and aggregate to produce a new table:

    >>> table3 = (table1.group_by(table1.SEX)
    ...                 .aggregate(mean_height_data = agg.mean(table1.HT)))
    >>> table3.show()

    Join tables together inside an annotation expression:

    >>> table2 = table2.key_by('ID')
    >>> table1 = table1.annotate(B = table2[table1.ID].B)
    >>> table1.show()
    """

    def __init__(self, jt):
        super(Table, self).__init__(jt)
        self._global_indices = Indices(axes=set(), source=self)
        self._row_axis = 'row'
        self._row_indices = Indices(axes={self._row_axis}, source=self)

        for fd in self.global_schema.fields:
            self._set_field(fd.name, construct_reference(fd.name, fd.typ, self._global_indices))

        for fd in self.schema.fields:
            self._set_field(fd.name, construct_reference(fd.name, fd.typ, self._row_indices))

    @typecheck_method(item=oneof(strlike, Expression, slice, tupleof(Expression)))
    def __getitem__(self, item):
        if isinstance(item, str) or isinstance(item, unicode):
            return self._get_field(item)
        elif isinstance(item, slice):
            s = item
            if not (s.start is None and s.stop is None and s.step is None):
                raise ExpressionException(
                    "Expect unbounded slice syntax ':' to indicate global table join, found unexpected attributes {}".format(
                        ', '.join(x for x in ['start' if s.start is not None else None,
                                              'stop' if s.stop is not None else None,
                                              'step' if s.step is not None else None] if x is not None)
                    )
                )

            return self.view_join_globals()
        else:
            exprs = item if isinstance(item, tuple) else (item,)
            return self.view_join_rows(*exprs)

    @property
    @handle_py4j
    def schema(self):
        if self._schema is None:
            self._schema = Type._from_java(self._jt.signature())
            assert (isinstance(self._schema, TStruct))
        return self._schema

    @property
    @handle_py4j
    def columns(self):
        if self._column_names is None:
            self._column_names = list(self._jt.fieldNames())
        return self._column_names

    @property
    @handle_py4j
    def num_columns(self):
        if self._num_columns is None:
            self._num_columns = self._jt.nColumns()
        return self._num_columns

    @handle_py4j
    def num_partitions(self):
        """Returns the number of partitions in the table.

        Returns
        -------
        :obj:`int`
        """
        return self._jt.nPartitions()

    @handle_py4j
    def count(self):
        return self._jt.count()

    @handle_py4j
    def _force_count(self):
        return self._jt.forceCount()

    @classmethod
    @handle_py4j
    @record_classmethod
    @typecheck_method(rows=oneof(listof(Struct), listof(dictof(strlike, anytype))),
                      schema=TStruct,
                      key=oneof(strlike, listof(strlike)),
                      num_partitions=nullable(integral))
    def parallelize(cls, rows, schema, key=[], num_partitions=None):
        return Table(
            Env.hail().table.Table.parallelize(
                Env.hc()._jhc, [schema._convert_to_j(r) for r in rows],
                schema._jtype, wrap_to_list(key), joption(num_partitions)))

    @handle_py4j
    @typecheck_method(keys=oneof(strlike, Expression))
    def key_by(self, *keys):
        """Change which columns are keys.

        Examples
        --------
        Assume `table1` is a :class:`.Table` with three columns: `C1`, `C2`
        and `C3`.

        Change key columns:

        >>> table_result = table1.key_by('C2', 'C3')

        >>> table_result = table1.key_by(table1.C1)

        Set to no keys:

        >>> table_result = table1.key_by()

        Parameters
        ----------
        keys : varargs of type :obj:`str`
            Field(s) to key by.

        Returns
        -------
        :class:`.Table`
            Table with new set of keys.
        """
        str_keys = []
        fields_rev = {expr: name for name, expr in self._fields.items()}
        for k in keys:
            if isinstance(k, Expression):
                if k not in fields_rev:
                    raise ExpressionException("'key_by' permits only top-level fields of the table")
                elif k._indices != self._row_indices:
                    raise ExpressionException("key_by' expects row fields, found index {}"
                                              .format(list(k._indices.axes)))
                str_keys.append(fields_rev[k])
            else:
                if k not in self._fields:
                    raise LookupError(get_nice_field_error(self, k))
                if not self._fields[k]._indices == self._row_indices:
                    raise ValueError("'{}' is not a row field".format(k))
                str_keys.append(k)

        return Table(self._jt.keyBy(str_keys))

    @handle_py4j
    def annotate_globals(self, **named_exprs):
        """Add new global fields.

        Examples
        --------

        Add a new global field:

        >>> table_result = table1.annotate(pops = ['EUR', 'AFR', 'EAS', 'SAS'])

        Parameters
        ----------
        named_exprs : varargs of :class:`.Expression`
            Annotation expressions.

        Returns
        -------
        :class:`.Table`
            Table with new global field(s).
        """

        exprs = []
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        base, cleanup = self._process_joins(*named_exprs.values())
        for k, v in named_exprs.items():
            analyze('Table.annotate_globals', v, self._global_indices)
            check_collisions(self._fields, k, self._global_indices)
            exprs.append('{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))

        m = Table(base._jt.annotateGlobalExpr(",\n".join(exprs)))
        return cleanup(m)

    @handle_py4j
    def select_globals(self, *exprs, **named_exprs):
        """Select existing global fields or create new fields by name, dropping the rest.

        Examples
        --------
        Select one existing field and compute a new one:

        >>> table_result = table1.select_globals(table1.global_field_1,
        ...                                      another_global=['AFR', 'EUR', 'EAS', 'AMR', 'SAS'])

        Notes
        -----
        This method creates new global fields. If a created field shares its name
        with a row-indexed field of the table, the method will fail.

        Note
        ----

        See :meth:`.Table.select` for more information about using ``select`` methods.

        Note
        ----
        This method does not support aggregation.

        Parameters
        ----------
        exprs : variable-length args of :obj:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.Table`
            Table with specified global fields.
        """

        exprs = [self[e] if not isinstance(e, Expression) else e for e in exprs]
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + named_exprs.values()))

        for e in exprs:
            all_exprs.append(e)
            analyze('Table.select_globals', e, self._global_indices)
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select_globals' expects keyword arguments for complex expressions")
            strs.append(e._ast.to_hql())
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('Table.select_globals', e, self._global_indices)
            check_collisions(self._fields, k, self._global_indices)
            strs.append('{} = {}'.format(escape_id(k), to_expr(e)._ast.to_hql()))

        return cleanup(Table(base._jt.selectGlobal(strs)))

    def transmute_globals(self, **named_exprs):
        raise NotImplementedError()

    def transmute(self, **named_exprs):
        """Add new fields and drop fields referenced.

        Examples
        --------

        Create a single field from an expression of `C1`, `C2`, and `C3`.

        .. testsetup::

            table4 = hl.import_table('data/kt_example4.tsv', impute=True,
                                  types={'B': hl.tstruct(['B0', 'B1'], [hl.tbool, hl.tstr]),
                                 'D': hl.tstruct(['cat', 'dog'], [hl.tint32, hl.tint32]),
                                 'E': hl.tstruct(['A', 'B'], [hl.tint32, hl.tint32])})

        .. doctest::

            >>> table4.show()
            +-------+---------+--------+---------+-------+-------+-------+-------+
            |     A | B.B0    | B.B1   | C       | D.cat | D.dog |   E.A |   E.B |
            +-------+---------+--------+---------+-------+-------+-------+-------+
            | Int32 | Boolean | String | Boolean | Int32 | Int32 | Int32 | Int32 |
            +-------+---------+--------+---------+-------+-------+-------+-------+
            |    32 | true    | hello  | false   |     5 |     7 |     5 |     7 |
            +-------+---------+--------+---------+-------+-------+-------+-------+\

            >>> table_result = table4.transmute(F=table4.A + 2 * table4.E.B)
            >>> table_result.show()
            +---------+--------+---------+-------+-------+-------+
            | B.B0    | B.B1   | C       | D.cat | D.dog |     F |
            +---------+--------+---------+-------+-------+-------+
            | Boolean | String | Boolean | Int32 | Int32 | Int32 |
            +---------+--------+---------+-------+-------+-------+
            | true    | hello  | false   |     5 |     7 |    46 |
            +---------+--------+---------+-------+-------+-------+

        Notes
        -----
        This method functions to create new row-indexed fields and consume
        fields found in the expressions in `named_exprs`.

        All row-indexed top-level fields found in an expression are dropped
        after the new fields are created.

        Warning
        -------
        References to fields inside a top-level struct will remove the entire
        struct, as field `E` was removed in the example above since `E.B` was
        referenced.

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            New field expressions.

        Returns
        -------
        :class:`.Table`
            Table with transmuted fields.
        """
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        exprs = []
        base, cleanup = self._process_joins(*named_exprs.values())
        fields_referenced = set()
        for k, v in named_exprs.items():
            analyze('Table.transmute', v, self._row_indices)
            check_collisions(self._fields, k, self._row_indices)
            exprs.append('{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))
            for name, inds in v._refs:
                if inds == self._row_indices:
                    fields_referenced.add(name)
        fields_referenced = fields_referenced - set(named_exprs.keys())

        return cleanup(Table(base._jt
                             .annotate(",\n".join(exprs))
                             .drop(list(fields_referenced))))

    @handle_py4j
    def annotate(self, **named_exprs):
        """Add new fields.

        Examples
        --------

        Add field `Y` by computing the square of `X`:

        >>> table_result = table1.annotate(Y = table1.X ** 2)

        Add multiple fields simultaneously:

        >>> table_result = table1.annotate(A = table1.X / 2,
        ...                                B = table1.X + 21)

        Parameters
        ----------
        named_exprs : keyword args of :class:`.Expression`
            Expressions for new fields.

        Returns
        -------
        :class:`.Table`
            Table with new fields.
        """
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        exprs = []
        base, cleanup = self._process_joins(*named_exprs.values())
        for k, v in named_exprs.items():
            analyze('Table.annotate', v, self._row_indices)
            check_collisions(self._fields, k, self._row_indices)
            exprs.append('{k} = {v}'.format(k=escape_id(k), v=v._ast.to_hql()))

        return cleanup(Table(base._jt.annotate(",\n".join(exprs))))

    @handle_py4j
    @typecheck_method(expr=anytype,
                      keep=bool)
    def filter(self, expr, keep=True):
        """Filter rows.

        Examples
        --------

        Keep rows where ``C1`` equals 5:

        >>> table_result = table1.filter(table1.C1 == 5)

        Remove rows where ``C1`` equals 10:

        >>> table_result = table1.filter(table1.C1 == 10, keep=False)

        Notes
        -----

        The expression `expr` will be evaluated for every row of the table. If `keep`
        is ``True``, then rows where `expr` evaluates to ``False`` will be removed (the
        filter keeps the rows where the predicate evaluates to ``True``). If `keep` is
        ``False``, then rows where `expr` evaluates to ``False`` will be removed (the
        filter removes the rows where the predicate evaluates to ``True``).

        Warning
        -------
        When `expr` evaluates to missing, the row will be removed regardless of `keep`.

        Note
        ----
        This method does not support aggregation.

        Parameters
        ----------
        expr : bool or :class:`.BooleanExpression`
            Filter expression.
        keep : bool
            Keep rows where `expr` is true.

        Returns
        -------
        :class:`.Table`
            Filtered table.
        """
        expr = to_expr(expr)
        analyze('Table.filter', expr, self._row_indices)
        base, cleanup = self._process_joins(expr)
        if not isinstance(expr._type, TBoolean):
            raise TypeError("method 'filter' expects an expression of type 'TBoolean', found {}"
                            .format(expr._type.__class__))

        return cleanup(Table(base._jt.filter(expr._ast.to_hql(), keep)))

    @handle_py4j
    @typecheck_method(exprs=oneof(Expression, strlike),
                      named_exprs=anytype)
    def select(self, *exprs, **named_exprs):
        """Select existing fields or create new fields by name, dropping the rest.

        Examples
        --------
        Select a few old columns and compute a new one:

        >>> table_result = table1.select(table1.ID, table1.C1, Y=table1.Z - table1.X)

        Notes
        -----
        This method creates new row-indexed fields. If a created field shares its name
        with a global field of the table, the method will fail.

        Note
        ----

        **Using select**

        Select and its sibling methods (:meth:`.Table.select_globals`,
        :meth:`.MatrixTable.select_globals`, :meth:`.MatrixTable.select_rows`,
        :meth:`.MatrixTable.select_cols`, and :meth:`.MatrixTable.select_entries`) accept
        both variable-length (``f(x, y, z)``) and keyword (``f(a=x, b=y, c=z)``)
        arguments.

        Variable-length arguments can be either strings or expressions that reference a
        (possibly nested) field of the table. Keyword arguments can be arbitrary
        expressions.

        **The following three usages are all equivalent**, producing a new table with
        columns `C1` and `C2` of `table1`.

        First, variable-length string arguments:

        >>> table_result = table1.select('C1', 'C2')

        Second, field reference variable-length arguments:

        >>> table_result = table1.select(table1.C1, table1.C2)

        Last, expression keyword arguments:

        >>> table_result = table1.select(C1 = table1.C1, C2 = table1.C2)

        Additionally, the variable-length argument syntax also permits nested field
        references. Given the following struct field `s`:

        >>> table3 = table1.annotate(s = hl.Struct(x=table1.X, z=table1.Z))

        The following two usages are equivalent, producing a table with one field, `x`.:

        >>> table3_result = table3.select(table3.s.x)

        >>> table3_result = table3.select(x = table3.s.x)

        The keyword argument syntax permits arbitrary expressions:

        >>> table_result = table1.select(foo=table1.X ** 2 + 1)

        These syntaxes can be mixed together, with the stipulation that all keyword arguments
        must come at the end due to Python language restrictions.

        >>> table_result = table1.select(table1.X, 'Z', bar = [table1.C1, table1.C2])

        Note
        ----
        This method does not support aggregation.

        Parameters
        ----------
        exprs : variable-length args of :obj:`str` or :class:`.Expression`
            Arguments that specify field names or nested field reference expressions.
        named_exprs : keyword args of :class:`.Expression`
            Field names and the expressions to compute them.

        Returns
        -------
        :class:`.Table`
            Table with specified fields.
        """
        exprs = [self[e] if not isinstance(e, Expression) else e for e in exprs]
        named_exprs = {k: to_expr(v) for k, v in named_exprs.items()}
        strs = []
        all_exprs = []
        base, cleanup = self._process_joins(*(exprs + named_exprs.values()))

        for e in exprs:
            all_exprs.append(e)
            analyze('Table.select', e, self._row_indices)
            if e._ast.search(lambda ast: not isinstance(ast, Reference) and not isinstance(ast, Select)):
                raise ExpressionException("method 'select' expects keyword arguments for complex expressions")
            strs.append(e._ast.to_hql())
        for k, e in named_exprs.items():
            all_exprs.append(e)
            analyze('Table.select', e, self._row_indices)
            check_collisions(self._fields, k, self._row_indices)
            strs.append('{} = {}'.format(escape_id(k), to_expr(e)._ast.to_hql()))

        return cleanup(Table(base._jt.select(strs, False)))

    @handle_py4j
    @typecheck_method(exprs=oneof(strlike, Expression))
    def drop(self, *exprs):
        """Drop fields from the table.

        Examples
        --------

        Drop fields `C1` and `C2` using strings:

        >>> table_result = table1.drop('C1', 'C2')

        Drop fields `C1` and `C2` using field references:

        >>> table_result = table1.drop(table1.C1, table1.C2)

        Drop a list of fields:

        >>> fields_to_drop = ['C1', 'C2']
        >>> table_result = table1.drop(*fields_to_drop)

        Notes
        -----

        This method can be used to drop global or row-indexed fields. The arguments
        can be either strings (``'field'``), or top-level field references
        (``table.field`` or ``table['field']``).

        Parameters
        ----------
        exprs : varargs of :obj:`str` or :class:`.Expression`
            Names of fields to drop or field reference expressions.

        Returns
        -------
        :class:`.Table`
            Table without specified fields.
        """
        all_field_exprs = {e: k for k, e in self._fields.items()}
        fields_to_drop = set()
        for e in exprs:
            if isinstance(e, Expression):
                if e in all_field_exprs:
                    fields_to_drop.add(all_field_exprs[e])
                else:
                    raise ExpressionException("method 'drop' expects string field names or top-level field expressions"
                                              " (e.g. table['foo'])")
            else:
                assert isinstance(e, str) or isinstance(e, unicode)
                if e not in self._fields:
                    raise IndexError("table has no field '{}'".format(e))
                fields_to_drop.add(e)

        table = self
        if any(self._fields[field]._indices == self._global_indices for field in fields_to_drop):
            # need to drop globals
            new_global_fields = [k.name for k in table.global_schema.fields if
                                 k.name not in fields_to_drop]
            table = table.select_globals(*new_global_fields)

        if any(self._fields[field]._indices == self._row_indices for field in fields_to_drop):
            # need to drop row fields
            new_row_fields = [k.name for k in table.schema.fields if
                              k.name not in fields_to_drop]
            table = table.select(*new_row_fields)

        return table

    @handle_py4j
    def export(self, output, types_file=None, header=True, parallel=None):
        """Export to a TSV file.

        Examples
        --------
        Export to a tab-separated file:

        >>> table1.export('output/table1.tsv.bgz')

        Note
        ----
        It is highly recommended to export large files with a ``.bgz`` extension,
        which will use a block gzipped compression codec. These files can be
        read natively with any Hail method, as well as with Python's ``gzip.open``
        and R's ``read.table``.

        Parameters
        ----------
        output : str
            URI at which to write exported file.
        types_file : str or None
            URI at which to write file containing column type information.
        header : bool
            Include a header in the file.
        parallel : str or None
            If None, a single file is produced, otherwise a
            folder of file shards is produced. If 'separate_header',
            the header file is output separately from the file shards. If
            'header_per_shard', each file shard has a header. If set to None
            the export will be slower.
        """

        self._jt.export(output, types_file, header, Env.hail().utils.ExportType.getExportType(parallel))

    def group_by(self, *exprs, **named_exprs):
        """Group by a new set of keys for use with :meth:`.GroupedTable.aggregate`.

        Examples
        --------
        Compute the mean value of `X` and the sum of `Z` per unique `ID`:

        >>> table_result = (table1.group_by(table1.ID)
        ...                       .aggregate(meanX = agg.mean(table1.X), sumZ = agg.sum(table1.Z)))

        Group by a height bin and compute sex ratio per bin:

        >>> table_result = (table1.group_by(height_bin = (table1.HT / 20).to_int32())
        ...                       .aggregate(fraction_female = agg.fraction(table1.SEX == 'F')))

        Notes
        -----
        This function is always followed by :meth:`.GroupedTable.aggregate`. Follow the
        link for documentation on the aggregation step.

        Note
        ----
        **Using group_by**

        **group_by** and its sibling methods (:meth:`.MatrixTable.group_rows_by` and
        :meth:`.MatrixTable.group_cols_by`) accept both variable-length (``f(x, y, z)``)
        and keyword (``f(a=x, b=y, c=z)``) arguments.

        Variable-length arguments can be either strings or expressions that reference a
        (possibly nested) field of the table. Keyword arguments can be arbitrary
        expressions.

        **The following three usages are all equivalent**, producing a
        :class:`.GroupedTable` grouped by columns `C1` and `C2` of `table1`.

        First, variable-length string arguments:

        >>> table_result = (table1.group_by('C1', 'C2')
        ...                       .aggregate(meanX = agg.mean(table1.X)))

        Second, field reference variable-length arguments:

        >>> table_result = (table1.group_by(table1.C1, table1.C2)
        ...                       .aggregate(meanX = agg.mean(table1.X)))

        Last, expression keyword arguments:

        >>> table_result = (table1.group_by(C1 = table1.C1, C2 = table1.C2)
        ...                       .aggregate(meanX = agg.mean(table1.X)))

        Additionally, the variable-length argument syntax also permits nested field
        references. Given the following struct field `s`:

        >>> table3 = table1.annotate(s = hl.Struct(x=table1.X, z=table1.Z))

        The following two usages are equivalent, grouping by one field, `x`:

        >>> table_result = (table3.group_by(table3.s.x)
        ...                       .aggregate(meanX = agg.mean(table3.X)))

        >>> table_result = (table3.group_by(x = table3.s.x)
        ...                       .aggregate(meanX = agg.mean(table3.X)))

        The keyword argument syntax permits arbitrary expressions:

        >>> table_result = (table1.group_by(foo=table1.X ** 2 + 1)
        ...                       .aggregate(meanZ = agg.mean(table1.Z)))

        These syntaxes can be mixed together, with the stipulation that all keyword arguments
        must come at the end due to Python language restrictions.

        >>> table_result = (table1.group_by(table1.C1, 'C2', height_bin = (table1.HT / 20).to_int32())
        ...                       .aggregate(meanX = agg.mean(table1.X)))

        Note
        ----
        This method does not support aggregation in key expressions.

        Arguments
        ---------
        exprs : varargs of type str or :class:`.Expression`
            Field names or field reference expressions.
        named_exprs : keyword args of type :class:`.Expression`
            Field names and expressions to compute them.

        Returns
        -------
        :class:`.GroupedTable`
            Grouped table; use :meth:`.GroupedTable.aggregate` to complete the aggregation.
        """
        groups = []
        for e in exprs:
            if isinstance(e, str) or isinstance(e, unicode):
                e = self[e]
            else:
                e = to_expr(e)
            analyze('Table.group_by', e, self._row_indices)
            ast = e._ast.expand()
            if any(not isinstance(a, Reference) and not isinstance(a, Select) for a in ast):
                raise ExpressionException("method 'group_by' expects keyword arguments for complex expressions")
            key = ast[0].name if isinstance(ast[0], Reference) else ast[0].selection
            groups.append((key, e))
        for k, e in named_exprs.items():
            e = to_expr(e)
            analyze('Table.group_by', e, self._row_indices)
            groups.append((k, e))

        return GroupedTable(self, groups)

    @handle_py4j
    def aggregate(self, expr):
        """Aggregate over rows into a local value.

        Examples
        --------
        Aggregate over rows:

        .. doctest::

            >>> table1.aggregate(Struct(fraction_male=agg.fraction(table1.SEX == 'M'),
            ...                         mean_x=agg.mean(table1.X)))
            Struct(fraction_male=0.5, mean_x=6.5)

        Note
        ----
        This method supports (and expects!) aggregation over rows.

        Parameters
        ----------
        expr : :class:`.Expression`
            Aggregation expression.

        Returns
        -------
        any
            Aggregated value dependent on `expr`.
        """
        agg_base = self.columns[0]  # FIXME hack

        expr = to_expr(expr)
        base, _ = self._process_joins(expr)
        analyze('Table.aggregate', expr, self._global_indices, {self._row_axis})
        replace_aggregables(expr._ast, agg_base)

        result_list = base._jt.query(jarray(Env.jvm().java.lang.String, [expr._ast.to_hql()]))
        ptypes = [Type._from_java(x._2()) for x in result_list]
        assert len(ptypes) == 1
        annotations = [ptypes[i]._convert_to_py(result_list[i]._1()) for i in range(len(ptypes))]
        return annotations[0]

    @handle_py4j
    @typecheck_method(output=strlike,
                      overwrite=bool,
                      _codec_spec=nullable(strlike))
    def write(self, output, overwrite=False, _codec_spec=None):
        """Write to disk.

        Examples
        --------

        >>> table1.write('output/table1.kt')

        Note
        ----
        The write path must end in ".kt".

        Warning
        -------
        Do not write to a path that is being read from in the same computation.

        Parameters
        ----------
        output : str
            Path at which to write.
        overwrite : bool
            If ``True``, overwrite an existing file at the destination.
        """

        self._jt.write(output, overwrite, _codec_spec)

    @handle_py4j
    @typecheck_method(n=integral, width=integral, truncate=nullable(integral), types=bool)
    def show(self, n=10, width=90, truncate=None, types=True):
        """Print the first few rows of the table to the console.

        Examples
        --------
        Show the first lines of the table:

        .. doctest::

            >>> table1.show()
            +-------+-------+--------+-------+-------+-------+-------+-------+
            |    ID |    HT | SEX    |     X |     Z |    C1 |    C2 |    C3 |
            +-------+-------+--------+-------+-------+-------+-------+-------+
            | Int32 | Int32 | String | Int32 | Int32 | Int32 | Int32 | Int32 |
            +-------+-------+--------+-------+-------+-------+-------+-------+
            |     1 |    65 | M      |     5 |     4 |     2 |    50 |     5 |
            |     2 |    72 | M      |     6 |     3 |     2 |    61 |     1 |
            |     3 |    70 | F      |     7 |     3 |    10 |    81 |    -5 |
            |     4 |    60 | F      |     8 |     2 |    11 |    90 |   -10 |
            +-------+-------+--------+-------+-------+-------+-------+-------+

        Parameters
        ----------
        n : :obj:`int`
            Maximum number of rows to show.
        width : :obj:`int`
            Horizontal width at which to break columns.
        truncate : :obj:`int`, optional
            Truncate each field to the given number of characters. If
            ``None``, truncate fields to the given `width`.
        types : :obj:`bool`
            Print an extra header line with the type of each field.
        """
        to_print = self._jt.showString(n, joption(truncate), types, width)
        print(to_print)

    @handle_py4j
    def view_join_rows(self, *exprs):
        if not len(exprs) > 0:
            raise ValueError('Require at least one expression to index a table')

        exprs = [to_expr(e) for e in exprs]
        if not len(exprs) == len(self.key):
            raise ExpressionException('Key mismatch: table has {} keys, found {} index expressions'.format(
                len(self.key), len(exprs)))

        indices, aggregations, joins, refs = unify_all(*exprs)

        from hail.matrixtable import MatrixTable
        uid = Env._get_uid()

        src = indices.source

        key_set = set(self.key)
        new_fields = [x for x in self.schema.fields if x.name not in key_set]
        new_schema = TStruct.from_fields(new_fields)

        if src is None or len(indices.axes) == 0:
            # FIXME: this should be OK: table[m.global_index_into_table]
            raise ExpressionException('found explicit join indexed by a scalar expression')
        elif isinstance(src, Table):
            for i, (k, e) in enumerate(zip(self.key, exprs)):
                if not self[k]._type == e._type:
                    raise ExpressionException(
                        "type mismatch at index {} of table key: "
                        "expected type '{}', found '{}'"
                            .format(i, k, e))

            for e in exprs:
                analyze('Table.view_join_rows', e, src._row_indices)

            right = self
            right_keys = [right[k] for k in right.key]
            select_struct = Struct(**{k: right[k] for k in right.columns})
            right = right.select(*right_keys, **{uid: select_struct})
            uids = [Env._get_uid() for i in range(len(exprs))]
            full_key_strs = ',\n'.join('{}={}'.format(uids[i], exprs[i]._ast.to_hql()) for i in range(len(exprs)))

            def joiner(left):
                left = Table(left._jt.annotate(full_key_strs)).key_by(*uids)
                left = Table(left._jt.join(right._jt, 'left'))
                return left

            all_uids = uids[:]
            all_uids.append(uid)
            return construct_expr(Reference(uid), new_schema, indices, aggregations,
                                  joins.push(Join(joiner, all_uids, uid)), refs)
        elif isinstance(src, MatrixTable):
            if len(exprs) == 1:
                key_type = self._fields[self.key[0]].dtype
                expr_type = exprs[0].dtype
                if not (key_type == expr_type or
                        key_type == tinterval(expr_type)):
                    raise ExpressionException(
                        "type mismatch at index 0 of table key: expected type {expected}, found '{et}'"
                            .format(expected="'{}'".format(key_type) if not isinstance(key_type, TInterval)
                        else "'{}' or '{}'".format(key_type, key_type.point_type), et=expr_type))
            else:
                for i, (k, e) in enumerate(zip(self.key, exprs)):
                    if not self[k]._type == e._type:
                        raise ExpressionException(
                            "type mismatch at index {} of table key: "
                            "expected type '{}', found '{}'"
                                .format(i, k, e))
            for e in exprs:
                analyze('Table.view_join_rows', e, src._entry_indices)

            right = self
            # match on indices to determine join type
            if indices == src._entry_indices:
                raise NotImplementedError('entry-based matrix joins')
            elif indices == src._row_indices:

                is_row_key = len(exprs) == len(src.row_key) and all(
                    exprs[i] is src._fields[src.row_key[i]] for i in range(len(exprs)))
                is_partition_key = len(exprs) == len(src.partition_key) and all(
                    exprs[i] is src._fields[src.partition_key[i]] for i in range(len(exprs)))

                if is_row_key or is_partition_key:
                    # no vds_key (way faster)
                    def joiner(left):
                        return MatrixTable(left._jvds.annotateVariantsTable(
                            right._jt, uid, False))

                    return construct_expr(Select(Reference('va'), uid), new_schema,
                                          indices, aggregations, joins.push(Join(joiner, [uid], uid)))
                else:
                    # use vds_key
                    uids = [Env._get_uid() for _ in range(len(exprs))]

                    def joiner(left):
                        from hail.expr import functions, aggregators as agg

                        rk_uids = [Env._get_uid() for _ in left.row_key]
                        k_uid = Env._get_uid()

                        # extract v, key exprs
                        left2 = left.select_rows(*left.row_key, **{uid: e for uid, e in zip(uids, exprs)})
                        lrt = (left2.rows_table()
                            .rename({name: u for name, u in zip(left2.row_key, rk_uids)})
                            .key_by(*uids))

                        vt = lrt.join(right)
                        # group uids
                        od = OrderedDict()
                        for u in uids:
                            od[u] = vt[u]
                        vt = vt.annotate(**{k_uid: Struct(**od)})
                        vt = vt.drop(*uids)
                        # group by v and index by the key exprs
                        vt = (vt.group_by(*rk_uids)
                            .aggregate(values=agg.collect(
                            Struct(**{c: vt[c] for c in vt.columns if c not in rk_uids}))))
                        vt = vt.annotate(values=hl.index(vt.values, k_uid))

                        jl = left._jvds.annotateVariantsTable(vt._jt, uid, False)
                        key_expr = '{uid} = va.{uid}.values.get({{ {es} }})'.format(uid=uid, es=','.join(
                            '{}: {}'.format(u, e._ast.to_hql()) for u, e in
                            zip(uids, exprs)))
                        jl = jl.annotateVariantsExpr(key_expr)
                        return MatrixTable(jl)

                    return construct_expr(Select(Reference('va'), uid),
                                          new_schema, indices, aggregations, joins.push(Join(joiner, [uid], uid)))

            elif indices == src._col_indices:
                if len(exprs) == len(src.col_key) and all([exprs[i] is src[src.col_key[i]] for i in range(len(exprs))]):
                    # no vds_key (faster)
                    joiner = lambda left: MatrixTable(left._jvds.annotateSamplesTable(
                        right._jt, None, uid, False))
                else:
                    # use vds_key
                    joiner = lambda left: MatrixTable(left._jvds.annotateSamplesTable(
                        right._jt, [e._ast.to_hql() for e in exprs], uid, False))
                return construct_expr(Select(Reference('sa'), uid), new_schema,
                                      indices, aggregations, joins.push(Join(joiner, [uid], uid)))
            else:
                raise NotImplementedError()
        else:
            raise TypeError("Cannot join with expressions derived from '{}'".format(src.__class__))

    @handle_py4j
    def view_join_globals(self):
        uid = Env._get_uid()

        def joiner(obj):
            from hail.matrixtable import MatrixTable
            if isinstance(obj, MatrixTable):
                return MatrixTable(Env.jutils().joinGlobals(obj._jvds, self._jt, uid))
            else:
                assert isinstance(obj, Table)
                return Table(Env.jutils().joinGlobals(obj._jt, self._jt, uid))

        return construct_expr(GlobalJoinReference(uid), self.global_schema,
                              joins=LinkedList(Join).push(Join(joiner, [uid], uid)))

    def _process_joins(self, *exprs):
        # ordered to support nested joins
        original_key = self.key

        all_uids = []
        left = self
        used_uids = set()

        for e in exprs:
            rewrite_global_refs(e._ast, self)
            for j in list(e._joins)[::-1]:
                if j.uid not in used_uids:
                    left = j.join_function(left)
                    all_uids.extend(j.temp_vars)
                    used_uids.add(j.uid)

        if left is not self:
            left = left.key_by(*original_key)

        def cleanup(table):
            remaining_uids = [uid for uid in all_uids if uid in table._fields]
            return table.drop(*remaining_uids)

        return left, cleanup

    @classmethod
    @handle_py4j
    @typecheck_method(n=integral,
                      num_partitions=nullable(integral))
    def range(cls, n, num_partitions=None):
        """Construct a table with `n` rows and one field `idx` that ranges from
        0 to ``n - 1``.

        Examples
        --------
        Construct a table with 100 rows:

        >>> range_table = hl.Table.range(100)

        Construct a table with one million rows and twenty partitions:

        >>> range_table = hl.Table.range(1000000, 20)

        Notes
        -----
        The resulting table has one column:

         - `idx` (**Int32**) - Unique row index from 0 to `n` - 1.

        Parameters
        ----------
        n : :obj:`int`
            Number of rows.
        num_partitions : :obj:`int`
            Number of partitions.

        Returns
        -------
        :class:`.Table`
            Table with one field, `index`.
        """
        return Table(Env.hail().table.Table.range(Env.hc()._jhc, n, 'idx', joption(num_partitions)))

    @handle_py4j
    def cache(self):
        """Persist this table in memory.

        Examples
        --------
        Persist the table in memory:

        >>> table = table.cache() # doctest: +SKIP

        Notes
        -----

        This method is an alias for :func:`persist("MEMORY_ONLY") <hail.Table.persist>`.

        Returns
        -------
        :class:`.Table`
            Cached table.
        """
        return self.persist('MEMORY_ONLY')

    @typecheck_method(storage_level=storage_level)
    def persist(self, storage_level='MEMORY_AND_DISK'):
        """Persist this table in memory or on disk.

        Examples
        --------
        Persist the key table to both memory and disk:

        >>> table = table.persist() # doctest: +SKIP

        Notes
        -----

        The :meth:`.Table.persist` and :meth:`.Table.cache` methods store the
        current table on disk or in memory temporarily to avoid redundant computation
        and improve the performance of Hail pipelines. This method is not a substitution
        for :meth:`.Table.write`, which stores a permanent file.

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
        :class:`.Table`
            Persisted table.
        """
        return Table(self._jt.persist(storage_level))

    @handle_py4j
    def unpersist(self):
        """
        Unpersists this table from memory/disk.

        Notes
        -----
        This function will have no effect on a table that was not previously
        persisted.

        Returns
        -------
        :class:`.Table`
            Unpersisted table.
        """
        self._jt.unpersist()

    @handle_py4j
    def collect(self):
        """Collect the rows of the table into a local list.

        Examples
        --------
        Collect a list of all `X` records:

        >>> all_xs = [row['X'] for row in table1.select(table1.X).collect()]

        Notes
        -----
        This method returns a list whose elements are of type :class:`.Struct`. Fields
        of these structs can be accessed similarly to fields on a table, using dot
        methods (``struct.foo``) or string indexing (``struct['foo']``).

        Warning
        -------
        Using this method can cause out of memory errors. Only collect small tables.

        Returns
        -------
        :obj:`list` of :class:`.Struct`
            List of rows.
        """
        return tarray(self.schema)._convert_to_py(self._jt.collect())

    def describe(self):
        """Print information about the fields in the table."""

        def format_type(typ):
            return typ.pretty(indent=4)

        if len(self.global_schema.fields) == 0:
            global_fields = '\n    None'
        else:
            global_fields = ''.join("\n    '{name}': {type} ".format(
                name=fd.name, type=format_type(fd.typ)) for fd in self.global_schema.fields)

        row_fields = ''.join("\n    '{name}': {type} ".format(
            name=fd.name, type=format_type(fd.typ)) for fd in self.schema.fields)

        row_key = ''.join("\n    '{name}': {type} ".format(name=f, type=format_type(self[f].dtype))
                          for f in self.key) if self.key else '\n    None'

        s = '----------------------------------------\n' \
            'Global fields:{g}\n' \
            '----------------------------------------\n' \
            'Row fields:{r}\n' \
            '----------------------------------------\n' \
            'Key:{rk}\n' \
            '----------------------------------------'.format(g=global_fields,
                                                              rk=row_key,
                                                              r=row_fields)
        print(s)

    @handle_py4j
    @typecheck_method(name=strlike)
    def index(self, name='idx'):
        """Add the integer index of each row as a new row field.

        Examples
        --------

        .. doctest::

            >>> table_result = table1.index()
            >>> table_result.show()
            +-------+-------+--------+-------+-------+-------+-------+-------+-------+
            |    ID |    HT | SEX    |     X |     Z |    C1 |    C2 |    C3 |   idx |
            +-------+-------+--------+-------+-------+-------+-------+-------+-------+
            | Int32 | Int32 | String | Int32 | Int32 | Int32 | Int32 | Int32 | Int64 |
            +-------+-------+--------+-------+-------+-------+-------+-------+-------+
            |     1 |    65 | M      |     5 |     4 |     2 |    50 |     5 |     0 |
            |     2 |    72 | M      |     6 |     3 |     2 |    61 |     1 |     1 |
            |     3 |    70 | F      |     7 |     3 |    10 |    81 |    -5 |     2 |
            |     4 |    60 | F      |     8 |     2 |    11 |    90 |   -10 |     3 |
            +-------+-------+--------+-------+-------+-------+-------+-------+-------+

        Notes
        -----

        This method returns a table with a new column whose name is given by
        the `name` parameter, with type ``Int64``. The value of this column is
        the integer index of each row, starting from 0. Methods that respect
        ordering (like :meth:`.Table.take` or :meth:`.Table.export`) will
        return rows in order.

        This method is also helpful for creating a unique integer index for
        rows of a table so that more complex types can be encoded as a simple
        number for performance reasons.

        Parameters
        ----------
        name : str
            Name of index column.

        Returns
        -------
        :class:`.Table`
            Table with a new index field.
        """

        return Table(self._jt.index(name))

    @handle_py4j
    @typecheck_method(tables=table_type)
    def union(self, *tables):
        """Union the rows of multiple tables.

        Examples
        --------

        Take the union of rows from two tables:

        .. testsetup::

            table = hl.import_table('data/kt_example1.tsv', impute=True, key='ID')
            other_table = table

        >>> union_table = table.union(other_table)

        Notes
        -----

        If a row appears in both tables identically, it is duplicated in the
        result. The left and right tables must have the same schema and key.

        Parameters
        ----------
        tables : varargs of :class:`.Table`
            Tables to union.

        Returns
        -------
        :class:`.Table`
            Table with all rows from each component table.
        """

        return Table(self._jt.union([table._jt for table in tables]))

    @handle_py4j
    @typecheck_method(n=integral)
    def take(self, n):
        """Collect the first `n` rows of the table into a local list.

        Examples
        --------
        Take the first three rows:

        .. doctest::

            >>> first3 = table1.take(3)
            >>> print(first3)
            [Struct(HT=65, SEX=M, X=5, C3=5, C2=50, C1=2, Z=4, ID=1),
             Struct(HT=72, SEX=M, X=6, C3=1, C2=61, C1=2, Z=3, ID=2),
             Struct(HT=70, SEX=F, X=7, C3=-5, C2=81, C1=10, Z=3, ID=3)]

        Notes
        -----

        This method does not need to look at all the data in the table, and
        allows for fast queries of the start of the table.

        This method is equivalent to :meth:`.Table.head` followed by
        :meth:`.Table.collect`.

        Parameters
        ----------
        n : int
            Number of rows to take.

        Returns
        -------
        :obj:`list` of :class:`.Struct`
            List of row structs.
        """

        return [self.schema._convert_to_py(r) for r in self._jt.take(n)]

    @handle_py4j
    @typecheck_method(n=integral)
    def head(self, n):
        """Subset table to first `n` rows.

        Examples
        --------
        Subset to the first three rows:

        .. doctest::

            >>> table_result = table1.head(3)
            >>> table_result.count()
            3

        Notes
        -----

        The number of partitions in the new table is equal to the number of
        partitions containing the first `n` rows.

        Parameters
        ----------
        n : int
            Number of rows to include.

        Returns
        -------
        :class:`.Table`
            Table including the first `n` rows.
        """

        return Table(self._jt.head(n))

    @handle_py4j
    @typecheck_method(p=numeric,
                      seed=integral)
    def sample(self, p, seed=0):
        """Downsample the table by keeping each row with probability ``p``.

        Examples
        --------

        Downsample the table to approximately 1% of its rows.

        >>> small_table1 = table1.sample(0.01)

        Parameters
        ----------
        p : :obj:`float`
            Probability of keeping each row.
        seed : :obj:`int`
            Random seed.

        Returns
        -------
        :class:`.Table`
            Table with approximately ``p * num_rows`` rows.
        """

        if not (0 <= p <= 1):
            raise ValueError("Requires 'p' in [0,1]. Found p={}".format(p))

        return Table(self._jt.sample(p, seed))

    @handle_py4j
    @typecheck_method(n=integral,
                      shuffle=bool)
    def repartition(self, n, shuffle=True):
        """Change the number of distributed partitions.

        Examples
        --------
        Repartition to 10 partitions:

        >>> table_result = table1.repartition(10)

        Warning
        -------
        When `shuffle` is ``False``, `repartition` can only decrease the number
        of partitions and simply combines adjacent partitions to achieve the
        desired number. It does not attempt to rebalance and so can produce a
        heavily unbalanced dataset. An unbalanced dataset can be inefficient to
        operate on because the work is not evenly distributed across partitions.

        Parameters
        ----------
        n : int
            Desired number of partitions.
        shuffle : bool
            If ``True``, shuffle data. Otherwise, naively coalesce.

        Returns
        -------
        :class:`.Table`
            Repartitioned table.
        """

        return Table(self._jt.repartition(n, shuffle))

    @handle_py4j
    @typecheck_method(right=table_type,
                      how=enumeration('inner', 'outer', 'left', 'right'))
    def join(self, right, how='inner'):
        """Join two tables together.

        Examples
        --------
        Join `table1` to `table2` to produce `table_joined`:

        >>> table_joined = table1.key_by('ID').join(table2.key_by('ID'))

        Notes
        -----
        Hail supports four types of joins specified by `how`:

        - **inner** -- Key must be present in both the left and right tables.
        - **outer** -- Key present in either the left or the right. For keys
          only in the left table, the right table's fields will be missing.
          For keys only in the right table, the left table's fields will be
          missing.
        - **left** -- Key present in the left table. For keys not found on
          the right, the right table's fields will be missing.
        - **right** -- Key present in the right table. For keys not found on
          the right, the right table's fields will be missing.

        Both tables must have the same number of keys and the corresponding
        types of each key must be the same (order matters), but the key names
        can be different. For example, if `table1` is keyed by fields ``['a',
        'b']``, both of type ``Int32``, and `table2` is keyed by fields ``['c',
        'd']``, both of type ``Int32``, then the two tables can be joined (their
        rows will be joined where ``table1.a == table2.c`` and ``table1.b ==
        table2.d``).

        The key field names and order from the left table are preserved, while
        the key fields from the right table are not present in the result.

        Parameters
        ----------
        right : :class:`.Table`
            Table with which to join.
        how : :obj:`str`
            Join type. One of "inner", "outer", "left", "right".

        Returns
        -------
        :class:`.Table`
            Joined table.
        """

        return Table(self._jt.join(right._jt, how))

    @handle_py4j
    @typecheck_method(expr=BooleanExpression)
    def all(self, expr):
        """Evaluate whether a boolean expression is true for all rows.

        Examples
        --------
        Test whether `C1` is greater than 5 in all rows of the table:

        >>> if table1.all(table1.C1 == 5):
        ...     print("All rows have C1 equal 5.")

        Parameters
        ----------
        expr : :class:`.BooleanExpression`
            Expression to test.

        Returns
        -------
        :obj:`bool`
        """
        expr = to_expr(expr)
        analyze('Table.all', expr, self._row_indices)
        base, cleanup = self._process_joins(expr)
        if not isinstance(expr._type, TBoolean):
            raise TypeError("method 'filter' expects an expression of type 'TBoolean', found {}"
                            .format(expr._type.__class__))

        return base._jt.forall(expr._ast.to_hql())

    @handle_py4j
    @typecheck_method(expr=BooleanExpression)
    def any(self, expr):
        """Evaluate whether a Boolean expression is true for at least one row.

        Examples
        --------

        Test whether `C1` is equal to 5 any row in any row of the table:

        >>> if table1.any(table1.C1 == 5):
        ...     print("At least one row has C1 equal 5.")

        Parameters
        ----------
        expr : :class:`.BooleanExpression`
            Boolean expression.

        Returns
        -------
        :obj:`bool`
            ``True`` if the predicate evaluated for ``True`` for any row, otherwise ``False``.
        """
        expr = to_expr(expr)
        analyze('Table.any', expr, self._row_indices)
        base, cleanup = self._process_joins(expr)
        if not isinstance(expr._type, TBoolean):
            raise TypeError("method 'filter' expects an expression of type 'TBoolean', found {}"
                            .format(expr._type.__class__))

        return base._jt.exists(expr._ast.to_hql())

    @handle_py4j
    @typecheck_method(mapping=dictof(strlike, strlike))
    def rename(self, mapping):
        """Rename fields of the table.

        Examples
        --------
        Rename `C1` to `col1` and `C2` to `col2`:

        >>> table_result = table1.rename({'C1' : 'col1', 'C2' : 'col2'})

        Parameters
        ----------
        mapping : :obj:`dict` of :obj:`str`, :obj:`str`
            Mapping from old column names to new column names.

        Notes
        -----
        Any field that does not appear as a key in `mapping` will not be
        renamed.

        Returns
        -------
        :class:`.Table`
            Table with renamed fields.
        """

        return Table(self._jt.rename(mapping))

    @handle_py4j
    def expand_types(self):
        """Expand complex types into structs and arrays.

        Examples
        --------

        >>> table_result = table1.expand_types()

        Notes
        -----
        Expands the following types: :class:`.TLocus`, :class:`.TInterval`,
        :class:`.TSet`, :class:`.TDict`.

        The only types that will remain after this method are:
        :class:`.TBoolean`, :class:`.TInt32`, :class:`.TInt64`,
        :class:`.TFloat64`, :class:`.TFloat32`, :class:`.TArray`,
        :class:`.TStruct`.

        Returns
        -------
        :class:`.Table`
            Expanded table.
        """

        return Table(self._jt.expandTypes())

    @handle_py4j
    def flatten(self):
        """Flatten nested structs.

        Examples
        --------
        Flatten table:

        >>> table_result = table1.flatten()

        Notes
        -----
        Consider a table with signature

        .. code-block:: text

            a: Struct {
                p: Int32,
                q: String
            },
            b: Int32,
            c: Struct {
                x: String,
                y: Array[Struct {
                    y: String,
                    z: String
                }]
            }

        and a single key column ``a``.  The result of flatten is

        .. code-block:: text

            a.p: Int32
            a.q: String
            b: Int32
            c.x: String
            c.y: Array[Struct {
                y: String,
                z: String
            }]

        with key fields ``a.p`` and ``a.q``.

        Note, structures inside collections like arrays or sets will not be
        flattened.

        Warning
        -------
        Flattening a table will produces fields that cannot be referenced using
        the ``table.<field>`` syntax, e.g. "a.b". Reference these fields using
        square bracket lookups: ``table['a.b']``.

        Returns
        -------
        :class:`.Table`
            Table with a flat schema (no struct fields).
        """

        return Table(self._jt.flatten())

    @handle_py4j
    @typecheck_method(exprs=oneof(strlike, Expression, Ascending, Descending))
    def order_by(self, *exprs):
        """Sort by the specified columns.

        Examples
        --------
        Four equivalent ways to order the table by field `HT`, ascending:

        >>> sorted_table = table1.order_by(table1.HT)

        >>> sorted_table = table1.order_by('HT')

        >>> sorted_table = table1.order_by(hl.asc(table1.HT))

        >>> sorted_table = table1.order_by(hl.asc('HT'))

        Notes
        -----
        Missing values are sorted after non-missing values. When multiple
        fields are passed, the table will be sorted first by the first
        argument, then the second, etc.

        Parameters
        ----------
        exprs : varargs of :class:`.Ascending` or :class:`.Descending` or :class:`.Expression` or :obj:`str`
            Fields to sort by.

        Returns
        -------
        :class:`.Table`
            Table sorted by the given fields.
        """
        sort_cols = []
        fields_rev = {v: k for k, v in self._fields.items()}
        for e in exprs:
            if isinstance(e, str) or isinstance(e, unicode):
                expr = self[e]
                if not expr._indices == self._row_indices:
                    raise ValueError("Sort fields must be row-indexed, found global field '{}'".format(e))
                sort_cols.append(asc(e)._j_obj())
            elif isinstance(e, Expression):
                if not e in fields_rev:
                    raise ValueError("Expect top-level field, found a complex expression")
                if not e._indices == self._row_indices:
                    raise ValueError("Sort fields must be row-indexed, found global field '{}'".format(e))
                sort_cols.append(asc(fields_rev[e])._j_obj())
            else:
                assert isinstance(e, Ascending) or isinstance(e, Descending)
                if isinstance(e.col, str) or isinstance(e.col, unicode):
                    expr = self[e.col]
                    if not expr._indices == self._row_indices:
                        raise ValueError("Sort fields must be row-indexed, found global field '{}'".format(e))
                    sort_cols.append(e._j_obj())
                else:
                    if not e.col in fields_rev:
                        raise ValueError("Expect top-level field, found a complex expression")
                    if not e.col._indices == self._row_indices:
                        raise ValueError("Sort fields must be row-indexed, found global field '{}'".format(e))
                    e.col = fields_rev[e.col]
                    sort_cols.append(e._j_obj())
        return Table(self._jt.orderBy(jarray(Env.hail().table.SortColumn, sort_cols)))

    @handle_py4j
    @typecheck_method(field=oneof(strlike, Expression))
    def explode(self, field):
        """Explode rows along a top-level field of the table.

        Each row is copied for each element of `field`.
        The explode operation unpacks the elements in a column of type
        ``Array`` or ``Set`` into its own row. If an empty ``Array`` or ``Set``
        is exploded, the entire row is removed from the table.

        Examples
        --------

        `people_table` is a :class:`.Table` with three columns: `Name`, `Age` and `Children`.

        .. testsetup::

            people_table = hl.import_table('data/explode_example.tsv', delimiter='\\s+',
                                     types={'Age': hl.tint32, 'Children': hl.tarray(hl.tstr)})

        .. doctest::

            >>> people_table.show()
            +----------+-------+--------------------------+
            | Name     |   Age | Children                 |
            +----------+-------+--------------------------+
            | String   | Int32 | Array[String]            |
            +----------+-------+--------------------------+
            | Alice    |    34 | ["Dave","Ernie","Frank"] |
            | Bob      |    51 | ["Gaby","Helen"]         |
            | Caroline |    10 | []                       |
            +----------+-------+--------------------------+

        :meth:`.Table.explode` can be used to produce a distinct row for each
        element in the `Children` field:

        .. doctest::

            >>> exploded = people_table.explode('Children')
            >>> exploded.show()
            +--------+-------+----------+
            | Name   |   Age | Children |
            +--------+-------+----------+
            | String | Int32 | String   |
            +--------+-------+----------+
            | Alice  |    34 | Dave     |
            | Alice  |    34 | Ernie    |
            | Alice  |    34 | Frank    |
            | Bob    |    51 | Gaby     |
            | Bob    |    51 | Helen    |
            +--------+-------+----------+

        Notes
        -----
        Empty arrays or sets produce no rows in the resulting table. In the
        example above, notice that the name "Caroline" is not found in the
        exploded table.

        Missing arrays or sets are treated as empty.

        Parameters
        ----------
        field : :obj:`str` or :class:`.Expression`
            Top-level field name or expression.

        Returns
        -------
        :class:`.Table`
            Table with exploded field.
        """

        if not isinstance(field, Expression):
            # field is a str
            field = self[field]

        fields_rev = {expr: k for k, expr in self._fields.items()}
        if not field in fields_rev:
            # nested or complex expression
            raise ValueError("method 'explode' expects a top-level field name or expression")
        if not field._indices == self._row_indices:
            # global field
            assert field._indices == self._global_indices
            raise ValueError("method 'explode' expects a field indexed by ['row'], found global field")

        return Table(self._jt.explode(fields_rev[field]))

    @typecheck_method(row_key=expr_any,
                      col_key=expr_any,
                      entry_exprs=expr_any)
    @handle_py4j
    def to_matrix_table(self, row_key, col_key, **entry_exprs):

        from hail.matrixtable import MatrixTable

        all_exprs = []
        all_exprs.append(row_key)
        analyze('to_matrix_table/row_key', row_key, self._row_indices)
        all_exprs.append(col_key)
        analyze('to_matrix_table/col_key', col_key, self._row_indices)

        exprs = []

        for k, e in entry_exprs.items():
            all_exprs.append(e)
            analyze('to_matrix_table/entry_exprs/{}'.format(k), e, self._row_indices)
            exprs.append('`{k}` = {v}'.format(k=k, v=e._ast.to_hql()))

        base, cleanup = self._process_joins(*all_exprs)

        return MatrixTable(base._jt.toMatrixTable(row_key._ast.to_hql(),
                                                  col_key._ast.to_hql(),
                                                  ",\n".join(exprs),
                                                  joption(None)))

    @property
    @handle_py4j
    def globals(self):
        """Returns a struct expression including all global fields.

        Returns
        -------
        :class:`.StructExpression`
            Struct of all global fields.
        """
        # FIXME: Impossible to correct a struct with the correct schema
        # FIXME: need 'row' and 'globals' symbol in the Table parser, like VSM
        raise NotImplementedError()
        # return to_expr(Struct(**{fd.name: self[fd.name] for fd in self.global_schema.fields}))

    @property
    @handle_py4j
    def row(self):
        """Returns a struct expression including all row-indexed fields.

        Returns
        -------
        :class:`.StructExpression`
            Struct of all row fields.
        """
        # FIXME: Impossible to correct a struct with the correct schema
        # FIXME: 'row' and 'globals' symbol in the Table parser, like VSM
        raise NotImplementedError()

    @handle_py4j
    @typecheck_method(expand=bool,
                      flatten=bool)
    def to_spark(self, expand=True, flatten=True):
        """Converts this key table to a Spark DataFrame.

        Parameters
        ----------
        expand : :obj:`bool`
            If True, expand_types before converting to Pandas DataFrame.

        flatten : :obj:`bool`
            If True, flatten before converting to Pandas DataFrame.
            If `expand` and `flatten` are True, flatten is run after
            expand so that expanded types are flattened.

        Returns
        -------
        :class:`.pyspark.sql.DataFrame`
            Spark DataFrame constructed from the table.
        """
        jt = self._jt
        if expand:
            jt = jt.expandTypes()
        if flatten:
            jt = jt.flatten()
        return DataFrame(jt.toDF(Env.hc()._jsql_context), Env.hc()._sql_context)

    @handle_py4j
    @typecheck_method(expand=bool,
                      flatten=bool)
    def to_pandas(self, expand=True, flatten=True):
        """Converts this table into a Pandas DataFrame.

        Parameters
        ----------
        expand : :obj:`bool`
            If True, expand_types before converting to Pandas DataFrame.

        flatten : :obj:`bool`
            If True, flatten before converting to Pandas DataFrame.
            If `expand` and `flatten` are True, flatten is run after
            expand so that expanded types are flattened.

        Returns
        -------
        :class:`.pandas.DataFrame`
            Pandas DataFrame constructed from the table.
        """
        return self.to_spark(expand, flatten).toPandas()

    @handle_py4j
    @typecheck_method(other=table_type, tolerance=nullable(numeric))
    def _same(self, other, tolerance=1e-6):
        return self._jt.same(other._jt, tolerance)


table_type.set(Table)
