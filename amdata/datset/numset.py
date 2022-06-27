from __future__ import annotations
from abc import abstractmethod, ABC
from typing import List, Tuple, Iterator

import datset.amarrays as arr
import datset.ambasic as bas
import datset.dset as dat


class NamedFloatRecords:
    def __init__(self, nns: arr.Strings, x: FloatRecords):
        self.m_col_to_name = nns
        self.m_row_to_col_to_value = x
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_col_to_name, arr.Strings)
        self.m_col_to_name.assert_ok()
        assert isinstance(self.m_row_to_col_to_value, FloatRecords)
        self.m_row_to_col_to_value.assert_ok()
        assert self.m_col_to_name.len() == self.m_row_to_col_to_value.num_cols()

    def explain(self):
        print(self.pretty_string())

    def pretty_strings(self) -> arr.Strings:
        return self.datset().pretty_strings()

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def datset(self) -> dat.Datset:
        result = dat.datset_empty(self.num_rows())
        for col, name in enumerate(self.names().range()):
            c = self.column_as_floats(col)
            result.add(dat.named_column_create(dat.colname_create(name), dat.column_from_floats(c)))
        return result

    def num_cols(self) -> int:
        return self.m_col_to_name.len()

    def column_as_datset_column(self, c: int) -> dat.Column:
        return dat.column_from_floats(self.column_as_floats(c))

    def termvecs(self) -> FloatRecords:
        return self.m_row_to_col_to_value

    def loosely_equals(self, other) -> bool:
        assert isinstance(other, NamedFloatRecords)
        if not self.m_col_to_name.equals(other.m_col_to_name):
            return False
        return self.termvecs().loosely_equals(other.termvecs())

    def num_rows(self) -> int:
        return self.termvecs().num_rows()

    def termvec(self, r: int) -> FloatRecord:
        return self.termvecs().termvec(r)

    def column_as_floats(self, term_num: int) -> arr.Floats:
        return self.termvecs().column_as_floats(term_num)

    def names(self) -> arr.Strings:
        return self.m_col_to_name


def categorical_name(cn: dat.Colname, vn: dat.Valname) -> str:
    return cn.string() + '_is_' + vn.string()


def categorical_names(cn: dat.Colname, encoding_to_valname: dat.Valnames) -> arr.Strings:
    nns = arr.strings_empty()
    for vn in encoding_to_valname.range():
        nns.add(categorical_name(cn, vn))
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


class Transformer(ABC):
    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def names(self) -> arr.Strings:
        pass

    @abstractmethod
    def transform_atom(self, a: dat.Atom) -> arr.Floats:
        pass

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
        self.m_names = categorical_names(cn, encoding_to_valname)
        self.m_valname_encoder = valname_encoder_create(encoding_to_valname)
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_names, arr.Strings)
        self.m_names.assert_ok()
        assert isinstance(self.m_valname_encoder, ValnameEncoder)
        self.m_valname_encoder.assert_ok()
        assert self.m_names.len() == self.m_valname_encoder.num_encodings()

    def names(self) -> arr.Strings:
        return self.m_names

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


class FloatTransformer(Transformer):
    def scaling_intervals(self) -> bas.Intervals:
        return bas.intervals_singleton(self.interval())

    def __init__(self, name: str, fs: arr.Floats):
        self.m_name = name
        self.m_interval = fs.extremes()
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_name, str)
        assert isinstance(self.m_interval, bas.Interval)
        self.m_interval.assert_ok()

    def name(self) -> str:
        return self.m_name

    def transform_float(self, f: float) -> arr.Floats:
        z = self.interval().fractional_from_absolute(f)
        return arr.floats_singleton(z)

    def interval(self) -> bas.Interval:
        return self.m_interval

    def names(self) -> arr.Strings:
        return arr.strings_singleton(self.name())

    def transform_atom(self, a: dat.Atom) -> arr.Floats:
        assert isinstance(a, dat.AtomFloat)
        return self.transform_float(a.float())

    def is_float(self) -> bool:
        return True


def float_transformer_create(nn: str, fs: arr.Floats) -> FloatTransformer:
    return FloatTransformer(nn, fs)


class BoolTransformer(Transformer):
    def scaling_intervals(self) -> bas.Intervals:
        return bas.intervals_all_unit(1)

    def __init__(self, nn: str):
        self.m_name = nn
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_name, str)

    def name(self) -> str:
        return self.m_name

    def transform_bool(self, b: bool) -> arr.Floats:
        assert isinstance(self, BoolTransformer)
        z: float
        if b:
            z = 1.0
        else:
            z = 0.0
        return arr.floats_singleton(z)

    def names(self) -> arr.Strings:
        return arr.strings_singleton(self.name())

    def transform_atom(self, a: dat.Atom) -> arr.Floats:
        assert isinstance(a, dat.AtomBool)
        return self.transform_bool(a.bool())


def bool_transformer_create(nn: str) -> BoolTransformer:
    return BoolTransformer(nn)


def bool_transformer_from_colname(cn: dat.Colname) -> BoolTransformer:
    return bool_transformer_create(cn.string())


def transformer_from_floats(cn: dat.Colname, fs: arr.Floats) -> FloatTransformer:
    return float_transformer_create(cn.string(), fs)


def transformer_from_column(cn: dat.Colname, c: dat.Column) -> Transformer:
    if isinstance(c, dat.ColumnCats):
        return transformer_from_cats(cn, c.cats())
    elif isinstance(c, dat.ColumnFloats):
        return transformer_from_floats(cn, c.floats())
    elif isinstance(c, dat.ColumnBools):
        return bool_transformer_from_colname(cn)
    else:
        bas.my_error("bad Transtype")


def transformer_from_named_column(nc: dat.NamedColumn) -> Transformer:
    return transformer_from_column(nc.colname(), nc.column())


def named_float_records_create(nns: arr.Strings, frs: FloatRecords) -> NamedFloatRecords:
    assert nns.len() == frs.num_cols()
    return NamedFloatRecords(nns, frs)


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


def termvec_create(fs: arr.Floats) -> FloatRecord:
    return FloatRecord(fs)


def termvec_from_numvec(nv: Numvec):
    fs = arr.floats_singleton(1.0)
    fs.append(nv.floats())
    return termvec_create(fs)


def termvecs_empty(n_terms) -> FloatRecords:
    return FloatRecords(n_terms, [])


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

    def names(self) -> arr.Strings:
        result = arr.strings_empty()
        for tf in self.range():
            result.append(tf.names())
        return result

    def transformer(self, i) -> Transformer:
        assert 0 <= i < self.len()
        return self.m_transformers[i]

    def len(self) -> int:
        return len(self.m_transformers)

    def numvec_from_row(self, rw: dat.Record) -> Numvec:
        result = arr.floats_empty()
        for t, a in zip(self.range(), rw.range()):
            fs = t.transform_atom(a)
            result.append(fs)
        return numvec_create(result)

    def transform_datset(self, ds: dat.Datset) -> NamedFloatRecords:
        nns = self.names()
        n_noomnames = nns.len()
        n_terms = n_noomnames + 1  # There's a constant term at the left
        frs = termvecs_empty(n_terms)
        for row in ds.range_records():
            z = self.float_record_from_record(row)
            frs.add(z)
        return named_float_records_create(nns, frs)

    def range(self) -> Iterator[Transformer]:
        for tf in self.m_transformers:
            yield tf

    def float_record_from_record(self, row: dat.Record) -> FloatRecord:
        return termvec_from_numvec(self.numvec_from_row(row))

    def named_float_records_from_datset(self, inputs):
        pass


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


def noomset_from_datset(ds: dat.Datset) -> NamedFloatRecords:
    tf = transformers_from_datset(ds)
    return tf.transform_datset(ds)


def float_records_empty(n_cols: int):
    return FloatRecords(n_cols, [])


def named_float_records_default() -> NamedFloatRecords:
    return named_float_records_create(arr.strings_empty(), float_records_empty(0))


def termvecs_from_fmat(fm: arr.Fmat) -> FloatRecords:
    result = termvecs_empty(fm.num_cols())
    for r in fm.range_rows():
        result.add(termvec_create(r))
    return result


def named_float_records_from_smat(sm: arr.Smat) -> Tuple[NamedFloatRecords, bas.Errmess]:
    if sm.num_rows() < 1:
        return named_float_records_default(), bas.errmess_error("Need at least 1 row")

    nns = sm.row(0)
    rest = sm.without_first_row()
    fm, err = arr.fmat_from_smat(rest)

    if err.is_error():
        return named_float_records_default(), err

    return named_float_records_create(nns, termvecs_from_fmat(fm)), bas.errmess_ok()


def named_float_records_from_multiline_string(s: str) -> Tuple[NamedFloatRecords, bas.Errmess]:
    sm, em = dat.smat_from_multiline_string(s)

    if em.is_error():
        return named_float_records_default(), em

    ns, em2 = named_float_records_from_smat(sm)

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

    ns, ns_errmess = named_float_records_from_multiline_string(ns_string)

    print(f'ns_errmess = {ns_errmess.string()}')
    assert ns_errmess.is_ok()
    ns2 = noomset_from_datset(ds)

    assert ns2.loosely_equals(ns)


class FloatRecord:
    def __init__(self, fs: arr.Floats):
        self.m_floats = fs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_floats, arr.Floats)
        self.m_floats.assert_ok()

    def floats(self) -> arr.Floats:
        return self.m_floats

    def len(self) -> int:
        return self.m_floats.len()

    def float(self, fr_index: int) -> float:
        return self.m_floats.float(fr_index)

    def loosely_equals(self, other: FloatRecord) -> bool:
        return self.floats().loosely_equals(other.floats())

    def times(self, fs: arr.Floats) -> float:
        return self.floats().dot_product(fs)


class FloatRecords:
    def __init__(self, n_terms: int, li: List[FloatRecord]):
        self.m_num_terms = n_terms
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for fr in self.m_list:
            assert isinstance(fr, FloatRecord)
            fr.assert_ok()
            assert fr.len() == self.m_num_terms

    def num_rows(self) -> int:
        return len(self.m_list)

    def num_cols(self) -> int:
        return self.m_num_terms

    def termvec(self, k: int) -> FloatRecord:
        assert 0 <= k < self.num_rows()
        return self.m_list[k]

    def float(self, row: int, term_num: int) -> float:
        return self.termvec(row).float(term_num)

    def loosely_equals(self, other: FloatRecords) -> bool:
        if self.num_rows() != other.num_rows():
            return False

        for s, t in zip(self.range(), other.range()):
            if not s.loosely_equals(t):
                return False

        return True

    def range(self) -> Iterator[FloatRecord]:
        for fr in self.m_list:
            yield fr

    def column_as_floats(self, term_num: int) -> arr.Floats:
        result = arr.floats_empty()
        for fr in self.range():
            result.add(fr.float(term_num))
        return result

    def add(self, fr: FloatRecord):
        assert self.num_cols() == fr.len()
        self.m_list.append(fr)

    def times(self, fs: arr.Floats) -> arr.Floats:
        result = arr.floats_empty()
        for fr in self.range():
            result.add(fr.times(fs))
        return result
