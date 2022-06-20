from __future__ import annotations
from abc import abstractmethod
from typing import List, Tuple, Iterator

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
    def __init__(self, li: List[Noomname]):
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
    def __init__(self, nns: Noomnames, x: Termvecs):
        self.m_noomnames = nns
        self.m_row_to_col_to_value = x
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_noomnames, Noomnames)
        self.m_noomnames.assert_ok()
        assert isinstance(self.m_row_to_col_to_value, Termvecs)
        self.m_row_to_col_to_value.assert_ok()
        assert self.num_noomnames() + 1 == self.num_terms()

    def explain(self):
        print(self.pretty_string())

    def pretty_strings(self) -> arr.Strings:
        return self.datset().pretty_strings()

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def datset(self) -> dat.Datset:
        result = dat.datset_empty(self.num_rows())
        for term_num, nn in enumerate(self.range_noomnames()):
            c = self.column_as_floats(term_num)
            result.add(dat.named_column_create(nn.colname(), dat.column_from_floats(c)))
        return result

    def num_cols(self) -> int:
        return self.m_noomnames.len()

    def column_as_datset_column(self, c: int):
        return dat.column_from_floats(self.column_as_floats(c))

    def termvecs(self) -> Termvecs:
        return self.m_row_to_col_to_value

    def loosely_equals(self, other) -> bool:
        assert isinstance(other, Noomset)
        if not self.m_noomnames.equals(other.m_noomnames):
            return False
        return self.termvecs().loosely_equals(other.termvecs())

    def num_rows(self) -> int:
        return self.termvecs().num_rows()

    def termvec(self, r: int) -> Termvec:
        return self.termvecs().termvec(r)

    def range_noomnames(self) -> Iterator[Noomname]:
        return self.m_noomnames.range()

    def num_noomnames(self) -> int:
        return self.m_noomnames.len()

    def num_terms(self) -> int:
        return self.m_row_to_col_to_value.num_terms()

    def column_as_floats(self, term_num: int) -> arr.Floats:
        return self.termvecs().column_as_floats(term_num)


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

    def encoding(self, vn: dat.Valname) -> Tuple[int, bool]:
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


class Transformer:
    def assert_ok(self):
        bas.my_error(f'{self} base')

    def noomnames(self) -> Noomnames:
        bas.my_error(f'{self} base')
        return noomnames_empty()

    def transform_atom(self, a: dat.Atom) -> arr.Floats:
        bas.my_error(f'{self} base')
        return arr.floats_empty()

    def is_float(self) -> bool:
        return False

    def transform_float(self, x) -> arr.Floats:
        bas.my_error(f'{self} base')
        return arr.floats_empty()

    @abstractmethod
    def scaling_intervals(self):
        pass


class CatTransformer(Transformer):
    def scaling_intervals(self) -> bas.Intervals:
        return bas.intervals_all_unit(self.num_encodings())

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

    def encoding_from_valname(self, vn: dat.Valname) -> Tuple[int, bool]:
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


class FloatTransformer(Transformer):
    def scaling_intervals(self) -> bas.Intervals:
        return bas.intervals_singleton(self.interval())

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

    def is_float(self) -> bool:
        return True


def float_transformer_create(nn: Noomname, fs: arr.Floats) -> FloatTransformer:
    return FloatTransformer(nn, fs)


class BoolTransformer(Transformer):
    def scaling_intervals(self) -> bas.Intervals:
        return bas.intervals_all_unit(1)

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


def noomname_from_colname(cn: dat.Colname) -> Noomname:
    return noomname_create(cn.string())


def transformer_from_floats(cn: dat.Colname, fs: arr.Floats) -> FloatTransformer:
    return float_transformer_create(noomname_from_colname(cn), fs)


def transformer_from_column(cn: dat.Colname, c: dat.Column) -> Transformer:
    if isinstance(c, dat.ColumnCats):
        return transformer_from_cats(cn, c.cats())
    elif isinstance(c, dat.ColumnFloats):
        return transformer_from_floats(cn, c.floats())
    elif isinstance(c, dat.ColumnBool):
        return bool_transformer_from_colname(cn)
    else:
        bas.my_error("bad Transtype")


def transformer_from_named_column(nc: dat.NamedColumn) -> Transformer:
    return transformer_from_column(nc.colname(), nc.column())


def noomset_from_termvecs(nns: Noomnames, tvs: Termvecs) -> Noomset:
    assert nns.len() == tvs.num_terms() - 1
    return Noomset(nns, tvs)


class Numvec:
    def __init__(self, fs: arr.Floats):
        self.m_floats = fs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_floats, arr.Floats)
        self.m_floats.assert_ok()

    def floats(self) -> arr.Floats:
        return self.m_floats


def numvec_create(fs: arr.Floats) -> Numvec:
    return Numvec(fs)


def termvec_create(fs: arr.Floats) -> Termvec:
    return Termvec(fs)


def termvec_from_numvec(nv: Numvec):
    fs = arr.floats_singleton(1.0)
    fs.append(nv.floats())
    return termvec_create(fs)


def termvecs_empty(n_terms) -> Termvecs:
    return Termvecs(n_terms, [])


class Transformers:
    def __init__(self, tfs: List[Transformer]):
        self.m_transformers = tfs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_transformers, list)
        for tf in self.m_transformers:
            assert isinstance(tf, Transformer)
            tf.assert_ok()

    def add(self, tf: Transformer):
        self.m_transformers.append(tf)

    def noomnames(self) -> Noomnames:
        result = noomnames_empty()
        for tf in self.range():
            result.append(tf.noomnames())
        return result

    def transformer(self, i) -> Transformer:
        assert 0 <= i < self.len()
        return self.m_transformers[i]

    def len(self) -> int:
        return len(self.m_transformers)

    def numvec_from_row(self, rw: dat.Row) -> Numvec:
        result = arr.floats_empty()
        for t, a in zip(self.range(), rw.range()):
            fs = t.transform_atom(a)
            result.append(fs)
        return numvec_create(result)

    def transform_datset(self, ds: dat.Datset) -> Noomset:
        nns = self.noomnames()
        n_noomnames = nns.len()
        n_terms = n_noomnames + 1  # There's a constant term at the left
        tvs = termvecs_empty(n_terms)
        for row in ds.range_rows():
            z = self.termvec_from_row(row)
            tvs.add(z)
        return noomset_from_termvecs(nns, tvs)

    def range(self) -> Iterator[Transformer]:
        for tf in self.m_transformers:
            yield tf

    def termvec_from_row(self, row: dat.Row) -> Termvec:
        return termvec_from_numvec(self.numvec_from_row(row))


def transformers_empty():
    return Transformers([])


def transformers_singleton(tf: Transformer) -> Transformers:
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
    return noomset_from_termvecs(nns, termvecs_empty(1))


def noomnames_from_strings(ss: arr.Strings) -> Noomnames:
    nns = noomnames_empty()
    for s in ss.range():
        nn = noomname_create(s)
        nns.add(nn)

    return nns


def termvecs_from_fmat(fm: arr.Fmat) -> Termvecs:
    result = termvecs_empty(fm.num_cols())
    for r in fm.range_rows():
        result.add(termvec_create(r))
    return result


def noomset_from_smat(sm: arr.Smat) -> Tuple[Noomset, bas.Errmess]:
    if sm.num_rows() < 1:
        return noomset_default(), bas.errmess_error("Need at least 1 row")

    nns = noomnames_from_strings(sm.row(0)).without_first()
    rest = sm.without_first_row()
    fm, err = arr.fmat_from_smat(rest)

    if err.is_error():
        return noomset_default(), err

    return noomset_from_termvecs(nns, termvecs_from_fmat(fm)), bas.errmess_ok()


def noomset_from_multiline_string(s: str) -> Tuple[Noomset, bas.Errmess]:
    sm, em = dat.smat_from_multiline_string(s)

    if em.is_error():
        return noomset_default(), em

    ns, em2 = noomset_from_smat(sm)

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


class Termvec:
    def __init__(self, fs: arr.Floats):
        self.m_floats = fs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_floats, arr.Floats)
        self.m_floats.assert_ok()

    def floats(self) -> arr.Floats:
        return self.m_floats

    def num_terms(self) -> int:
        return self.m_floats.len()

    def float(self, term_num: int) -> float:
        return self.m_floats.float(term_num)

    def loosely_equals(self, other: Termvec) -> bool:
        return self.floats().loosely_equals(other.floats())

    def times(self, fs: arr.Floats) -> float:
        return self.floats().dot_product(fs)


class Termvecs:
    def __init__(self, n_terms: int, li: list):
        self.m_num_terms = n_terms
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for tv in self.m_list:
            assert isinstance(tv, Termvec)
            tv.assert_ok()
            assert tv.num_terms() == self.m_num_terms

    def num_rows(self) -> int:
        return len(self.m_list)

    def num_terms(self) -> int:
        return self.m_num_terms

    def termvec(self, k: int) -> Termvec:
        assert 0 <= k < self.num_rows()
        return self.m_list[k]

    def float(self, row: int, term_num: int) -> float:
        return self.termvec(row).float(term_num)

    def loosely_equals(self, other: Termvecs) -> bool:
        if self.num_rows() != other.num_rows():
            return False

        for s, t in zip(self.range(), other.range()):
            if not s.loosely_equals(t):
                return False

        return True

    def range(self) -> Iterator[Termvec]:
        for tv in self.m_list:
            yield tv

    def column_as_floats(self, term_num: int) -> arr.Floats:
        result = arr.floats_empty()
        for tv in self.range():
            result.add(tv.float(term_num))
        return result

    def add(self, tv: Termvec):
        assert self.num_terms() == tv.num_terms()
        self.m_list.append(tv)

    def times(self, fs: arr.Floats) -> arr.Floats:
        result = arr.floats_empty()
        for tv in self.range():
            result.add(tv.times(fs))
        return result
