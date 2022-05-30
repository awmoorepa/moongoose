from collections.abc import Iterator

import datset.amarrays as arr
import datset.ambasic as bas
import datset.dset as dat


class Noomname:
    def __init__(self, name_as_str: str):
        self.m_name = name_as_str
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_name, str)
        assert len(self.m_name) > 0
        assert not bas.string_contains(self.m_name, bas.character_space())

    def string(self) -> str:
        return self.m_name

    def colname(self) -> dat.Colname:
        return dat.colname_create(self.string())

    def equals(self, other) -> bool:
        assert isinstance(other, Noomname)
        return self.m_name == other.m_name


class Noomnames:
    def __init__(self, li: list[Noomname]):
        self.m_noomnames = li
        self.assert_ok()

    def assert_ok(self):
        for nn in self.m_noomnames:
            assert isinstance(nn, Noomname)
            nn.assert_ok()

        assert not self.strings().contains_duplicates()

    def add(self, nn: Noomname):
        self.m_noomnames.append(nn)

    def len(self) -> int:
        return len(self.m_noomnames)

    def append(self, nns):
        assert isinstance(nns, Noomnames)
        for nn in nns.range():
            self.add(nn)

    def strings(self) -> arr.Strings:
        result = arr.strings_empty()
        for nn in self.range():
            result.add(nn.string())
        return result

    def noomname(self, i: int) -> Noomname:
        assert 0 <= i < self.len()
        return self.m_noomnames[i]

    def equals(self, other) -> bool:
        assert isinstance(other, Noomnames)
        if self.len() != other.len():
            return False

        for a, b in zip(self.range(), other.range()):
            if not a.equals(b):
                return False
        return True

    def contains(self, target: Noomname) -> bool:
        for candidate in self.range():
            if candidate.equals(target):
                return True
        return False

    def range(self) -> Iterator[Noomname]:
        for nn in self.m_noomnames:
            yield nn

    def without_first(self):  # returns Noomnames
        assert self.len() > 0
        result = noomnames_empty()
        for i, nn in enumerate(self.range()):
            if i > 0:
                result.add(nn)
        assert isinstance(result, Noomnames)
        return result


def noomnames_empty() -> Noomnames:
    return Noomnames([])


def noomname_create(name_as_string: str) -> Noomname:
    assert len(name_as_string) > 0
    spa = bas.character_space()
    und = bas.character_create('_')
    return Noomname(bas.string_replace(name_as_string, spa, und))


def noomnames_from_colname_and_valnames(cn: dat.Colname, encoding_to_valname: arr.Namer) -> Noomnames:
    result = noomnames_empty()
    for vn_string in encoding_to_valname.range_keys():
        s = f'{cn.string()}_is_{vn_string}'
        result.add(noomname_create(s))
    return result


class Noomset:
    def __init__(self, nns: Noomnames, fm: arr.Fmat):
        self.m_noomnames = nns
        self.m_row_to_col_to_value = fm
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_noomnames, Noomnames)
        self.m_noomnames.assert_ok()
        assert isinstance(self.m_row_to_col_to_value, arr.Fmat)
        print(f'fmat =\n{self.m_row_to_col_to_value.pretty_string()}')
        print(f'n_noomnames = {self.num_noomnames()}, fm_n_cols = {self.m_row_to_col_to_value.num_cols()}')
        self.m_row_to_col_to_value.assert_ok()
        assert self.num_noomnames() + 1 == self.num_cols_in_x_matrix()

    def explain(self):
        print(self.pretty_string())

    def pretty_strings(self) -> arr.Strings:
        return self.datset().pretty_strings()

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def datset(self) -> dat.Datset:
        result = dat.datset_empty(self.num_rows())
        for nn, c in zip(self.range_noomnames(), self.range_columns()):
            result.add(dat.named_column_create(nn.colname(), dat.column_from_floats(c)))
        return result

    def num_cols(self) -> int:
        return self.m_noomnames.len()

    def column_as_datset_column(self, c: int):
        return dat.column_from_floats(self.column_as_floats(c))

    def column_as_floats(self, c: int) -> arr.Floats:
        return self.fmat().column(c)

    def fmat(self) -> arr.Fmat:
        return self.m_row_to_col_to_value

    def loosely_equals(self, other) -> bool:
        assert isinstance(other, Noomset)
        if not self.m_noomnames.equals(other.m_noomnames):
            return False
        return self.fmat().loosely_equals(other.fmat())

    def num_rows(self) -> int:
        return self.fmat().num_rows()

    def row(self, r: int) -> arr.Floats:
        return self.fmat().row(r)

    def range_noomnames(self) -> Iterator[Noomname]:
        return self.m_noomnames.range()

    def range_columns(self) -> Iterator[arr.Floats]:
        for c in self.m_row_to_col_to_value.range_columns():
            yield c

    def num_noomnames(self) -> int:
        return self.m_noomnames.len()

    def num_cols_in_x_matrix(self) -> int:
        return self.m_row_to_col_to_value.num_cols()


def categorical_noomname(cn: dat.Colname, vn: dat.Valname) -> Noomname:
    s = cn.string() + '_is_' + vn.string()
    return noomname_create(s)


def categorical_noomnames(cn: dat.Colname, encoding_to_valname: dat.Valnames) -> Noomnames:
    nns = noomnames_empty()
    for vn in encoding_to_valname.range():
        nns.add(categorical_noomname(cn, vn))
    return nns


class ValnameEncoder:
    def __init__(self, nm: arr.Namer):
        self.m_namer = nm
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_namer, arr.Namer)
        self.m_namer.assert_ok()

    def num_encodings(self) -> int:
        return self.m_namer.len()

    def encoding(self, vn: dat.Valname) -> tuple[int, bool]:
        return self.m_namer.key_from_name(vn.string())


def namer_from_encoding_to_valname(encoding_to_valname: dat.Valnames) -> arr.Namer:
    nm = arr.namer_empty()
    for vn in encoding_to_valname.range():
        s = vn.string()
        assert not nm.contains(s)
        nm.add(s)
    return nm


def valname_encoder_create(encoding_to_valname: dat.Valnames) -> ValnameEncoder:
    nm = namer_from_encoding_to_valname(encoding_to_valname)
    return ValnameEncoder(nm)


class Xformer:
    def assert_ok(self):
        bas.my_error(f'{self} base')

    def noomnames(self) -> Noomnames:
        bas.my_error(f'{self} base')
        return noomnames_empty()

    def transform_atom(self, a: dat.Atom) -> arr.Floats:
        bas.my_error(f'{self} base')
        return arr.floats_empty()


class CatTransformer(Xformer):
    def __init__(self, cn: dat.Colname, encoding_to_valname: dat.Valnames):
        self.m_noomnames = categorical_noomnames(cn, encoding_to_valname)
        self.m_valname_encoder = valname_encoder_create(encoding_to_valname)
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_noomnames, Noomnames)
        self.m_noomnames.assert_ok()
        assert isinstance(self.m_valname_encoder, ValnameEncoder)
        self.m_valname_encoder.assert_ok()
        assert self.m_noomnames.len() == self.m_valname_encoder.num_encodings()

    def noomnames(self) -> Noomnames:
        return self.m_noomnames

    def transform_valname(self, vn: dat.Valname):
        result = arr.floats_all_zero(self.num_encodings())
        encoding, ok = self.encoding_from_valname(vn)
        if ok:
            result.set(encoding, 1.0)
        return result

    def transform_atom(self, a: dat.Atom) -> arr.Floats:
        assert isinstance(a, dat.AtomCategorical)
        return self.transform_valname(a.valname())

    def num_encodings(self) -> int:
        return self.m_valname_encoder.num_encodings()

    def encoding_from_valname(self, vn: dat.Valname) -> tuple[int, bool]:
        return self.m_valname_encoder.encoding(vn)


def cat_transformer_create(cn: dat.Colname, encoding_to_valname: dat.Valnames) -> CatTransformer:
    return CatTransformer(cn, encoding_to_valname)


def transformer_from_cats(cn: dat.Colname, cts: dat.Cats) -> CatTransformer:
    value_to_frequency = cts.histogram()
    mcv = value_to_frequency.argmax()
    encoding_to_valname = dat.valnames_empty()
    for v, vn in enumerate(cts.range_valnames_by_value()):
        if v != mcv:
            encoding_to_valname.add(vn)

    return cat_transformer_create(cn, encoding_to_valname)


# class Transtype(enum.Enum):
#     constant = 0
#     bool = 1
#     float = 2
#     categorical = 3


def noomnames_singleton(nn: Noomname) -> Noomnames:
    nns = noomnames_empty()
    nns.add(nn)
    return nns


def noomname_default() -> Noomname:
    return noomname_create('default')


class FloatTransformer(Xformer):
    def __init__(self, nn: Noomname, fs: arr.Floats):
        self.m_noomname = nn
        self.m_interval = fs.extremes()
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_noomname, Noomname)
        self.m_noomname.assert_ok()
        assert isinstance(self.m_interval, bas.Interval)
        self.m_interval.assert_ok()

    def noomname(self) -> Noomname:
        return self.m_noomname

    def transform_float(self, f: float) -> arr.Floats:
        z = self.interval().fractional_from_absolute(f)
        return arr.floats_singleton(z)

    def interval(self) -> bas.Interval:
        return self.m_interval

    def noomnames(self) -> Noomnames:
        return noomnames_singleton(self.noomname())

    def transform_atom(self, a: dat.Atom) -> arr.Floats:
        assert isinstance(a, dat.AtomFloat)
        return self.transform_float(a.float())


def float_transformer_create(nn: Noomname, fs: arr.Floats) -> FloatTransformer:
    return FloatTransformer(nn, fs)


class BoolTransformer(Xformer):
    def __init__(self, nn: Noomname):
        self.m_noomname = nn
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_noomname, Noomname)
        self.m_noomname.assert_ok()

    def noomname(self) -> Noomname:
        return self.m_noomname

    def transform_bool(self, b: bool) -> arr.Floats:
        assert isinstance(self, BoolTransformer)
        z: float
        if b:
            z = 1.0
        else:
            z = 0.0
        return arr.floats_singleton(z)

    def noomnames(self) -> Noomnames:
        return noomnames_singleton(self.noomname())

    def transform_atom(self, a: dat.Atom) -> arr.Floats:
        assert isinstance(a, dat.AtomBool)
        return self.transform_bool(a.bool())


def bool_transformer_create(nn: Noomname) -> BoolTransformer:
    return BoolTransformer(nn)


def bool_transformer_from_colname(cn: dat.Colname) -> BoolTransformer:
    return bool_transformer_create(noomname_from_colname(cn))


# class Transformer:
#     def __init__(self, tt: Transtype, data):
#         self.m_transtype = tt
#         self.m_data = data
#         self.assert_ok()
#
#     def assert_ok(self):
#         assert isinstance(self.m_transtype, Transtype)
#         tt = self.m_transtype
#
#         if tt == Transtype.categorical:
#             assert isinstance(self.m_data, CatTransformer)
#             self.m_data.assert_ok()
#         elif tt == Transtype.float:
#             assert isinstance(self.m_data, FloatTransformer)
#             self.m_data.assert_ok()
#         elif tt == Transtype.bool:
#             assert isinstance(self.m_data, BoolTransformer)
#             self.m_data.assert_ok()
#         elif tt == Transtype.constant:
#             assert self.m_data is None
#         else:
#             bas.my_error("bad Transtype")
#
#     def noomnames(self) -> Noomnames:
#         tt = self.m_transtype
#         if tt == Transtype.categorical:
#             return self.cat_transformer().noomnames()
#         elif tt == Transtype.float:
#             return noomnames_singleton(self.float_transformer().noomname())
#         elif tt == Transtype.bool:
#             return noomnames_singleton(self.bool_transformer().noomname())
#         elif tt == Transtype.constant:
#             return noomnames_singleton(noomname_create("constant"))
#         else:
#             bas.my_error("bad Transtype")
#
#     def cat_transformer(self) -> CatTransformer:
#         assert self.transtype() == Transtype.categorical
#         return self.m_data
#
#     def float_transformer(self) -> FloatTransformer:
#         assert self.transtype() == Transtype.float
#         return self.m_data
#
#     def bool_transformer(self) -> BoolTransformer:
#         assert self.transtype() == Transtype.bool
#         return self.m_data
#
#     def transtype(self) -> Transtype:
#         return self.m_transtype
#
#     def transform_atom(self, a: dat.Atom) -> arr.Floats:
#         tt = self.transtype()
#         assert tt != Transtype.constant
#         if tt == Transtype.categorical:
#             return self.cat_transformer().transform_valname(a.valname())
#         elif tt == Transtype.float:
#             return self.float_transformer().transform_float(a.float())
#         elif tt == Transtype.bool:
#             return self.bool_transformer().transform_bool(a.bool())
#         else:
#             bas.my_error("bad coltype")


def noomname_from_colname(cn: dat.Colname) -> Noomname:
    return noomname_create(cn.string())


def transformer_from_floats(cn: dat.Colname, fs: arr.Floats) -> FloatTransformer:
    return float_transformer_create(noomname_from_colname(cn), fs)


def transformer_from_column(cn: dat.Colname, c: dat.Column) -> Xformer:
    if isinstance(c, dat.ColumnCats):
        return transformer_from_cats(cn, c.cats())
    elif isinstance(c, dat.ColumnFloats):
        return transformer_from_floats(cn, c.floats())
    elif isinstance(c, dat.ColumnBool):
        return bool_transformer_from_colname(cn)
    else:
        bas.my_error("bad Transtype")


def transformer_from_named_column(nc: dat.NamedColumn) -> Xformer:
    return transformer_from_column(nc.colname(), nc.column())


def noomset_from_fmat(nns: Noomnames, fm: arr.Fmat) -> Noomset:
    assert nns.len() == fm.num_cols() - 1
    return Noomset(nns, fm)


def noomset_from_row_indexed_fmat(nns: Noomnames, rif: arr.RowIndexedFmat) -> Noomset:
    return noomset_from_fmat(nns, arr.fmat_create(rif))


class Transformers:
    def __init__(self, tfs: list[Xformer]):
        self.m_transformers = tfs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_transformers, list)
        for tf in self.m_transformers:
            assert isinstance(tf, Xformer)
            tf.assert_ok()

    def add(self, tf: Xformer):
        self.m_transformers.append(tf)

    def noomnames(self) -> Noomnames:
        result = noomnames_empty()
        for tf in self.range():
            result.append(tf.noomnames())
        return result

    def transformer(self, i) -> Xformer:
        assert 0 <= i < self.len()
        return self.m_transformers[i]

    def len(self) -> int:
        return len(self.m_transformers)

    def transform_row(self, rw: dat.Row) -> arr.Floats:
        result = arr.floats_singleton(1.0)
        for t, a in zip(self.range(), rw.range()):
            fs = t.transform_atom(a)
            result.append(fs)
        return result

    def transform_datset(self, ds: dat.Datset) -> Noomset:
        nns = self.noomnames()
        n_noomnames = nns.len()
        n_cols_in_x_matrix = n_noomnames + 1  # There's a constant term at the left
        rif = arr.row_indexed_fmat_with_no_rows(n_cols_in_x_matrix)
        for row in ds.range_rows():
            z = self.transform_row(row)
            rif.add_row(z)
        return noomset_from_row_indexed_fmat(nns, rif)

    def range(self) -> Iterator[Xformer]:
        for tf in self.m_transformers:
            yield tf


def transformers_empty():
    return Transformers([])


def transformers_singleton(tf: Xformer) -> Transformers:
    result = transformers_empty()
    result.add(tf)
    return result


def transformers_from_datset(ds: dat.Datset) -> Transformers:
    tf = transformers_empty()
    for nc in ds.range_named_columns():
        tf.add(transformer_from_named_column(nc))
    return tf


def noomset_from_datset(ds: dat.Datset) -> Noomset:
    tf = transformers_from_datset(ds)
    return tf.transform_datset(ds)


def noomset_default() -> Noomset:
    nn = noomname_create("default")
    nns = noomnames_singleton(nn)
    return noomset_from_fmat(nns, arr.fmat_default())


def noomnames_from_strings(ss: arr.Strings) -> Noomnames:
    nns = noomnames_empty()
    for s in ss.range():
        nn = noomname_create(s)
        nns.add(nn)

    return nns


def noomset_from_smat(sm: arr.Smat) -> tuple[Noomset, bas.Errmess]:
    if sm.num_rows() < 1:
        return noomset_default(), bas.errmess_error("Need at least 1 row")

    nns = noomnames_from_strings(sm.row(0)).without_first()
    rest = sm.without_first_row()
    fm, err = arr.fmat_from_smat(rest)

    if err.is_error():
        return noomset_default(), err

    return noomset_from_fmat(nns, fm), bas.errmess_ok()


def noomset_from_multiline_string(s: str) -> tuple[Noomset, bas.Errmess]:
    sm, em = dat.smat_from_multiline_string(s)

    print(f'noomset_from first em = {em.string()}')

    if em.is_error():
        return noomset_default(), em

    print(f'sm(noomset_from_multiline) =\n{sm.pretty_string()}')
    ns, em2 = noomset_from_smat(sm)

    print(f'em2(noomset_from_multiline) = {em2.string()}')
    print(f'noomset_from_smat(noomset_from_multiline) = \n{ns.pretty_string()}')

    return ns, em2


def unit_test():
    ds_string = """color,is_bright\n
    red,true\n
    blue,false\n
    blue,false\n
    yellow,true"""

    ns_string = """constant,color_is_red,color_is_yellow,is_bright\n
    1,1,0,1\n
    1,0,0,0\n
    1,0,0,0\n
    1,0,1,1"""

    ds, ds_errmess = dat.datset_from_multiline_string(ds_string)

    print(f'ds_errmess = {ds_errmess.string()}')

    assert ds_errmess.is_ok()

    ns, ns_errmess = noomset_from_multiline_string(ns_string)

    print(f'ns_errmess = {ns_errmess.string()}')
    assert ns_errmess.is_ok()
    ns2 = noomset_from_datset(ds)

    assert ns2.loosely_equals(ns)
