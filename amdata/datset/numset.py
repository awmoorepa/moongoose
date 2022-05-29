import enum

import datset.amarrays
import datset.dset as dat
import datset.ambasic as bas
import datset.amarrays as arr


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
        assert isinstance(self, Noomnames)
        for i in range(0, nns.len()):
            self.add(nns.noomname(i))

    def strings(self) -> arr.Strings:
        result = arr.strings_empty()
        for i in range(0, self.len()):
            result.add(self.noomname(i).string())
        return result

    def noomname(self, i: int) -> Noomname:
        assert 0 <= i < self.len()
        return self.m_noomnames[i]

    def equals(self, other) -> bool:
        assert isinstance(other, Noomnames)
        n = self.len()
        if n != other.len():
            return False
        for i in range(0, n):
            if not self.noomname(i).equals(other.noomname(i)):
                return False
        return True

    def contains(self, nn: Noomname) -> bool:
        col, ok = self.col_from_noomname(nn)
        return ok

    def col_from_noomname(self, nn: Noomname) -> tuple[int, bool]:
        for col in range(0, self.len()):
            if self.noomname(col).equals(nn):
                return col, True
        return -77, False


def noomnames_empty() -> Noomnames:
    return Noomnames([])


def noomname_create(name_as_string: str) -> Noomname:
    assert len(name_as_string) > 0
    spa = bas.character_space()
    und = bas.character_create('_')
    return Noomname(bas.string_replace(name_as_string, spa, und))


def noomnames_from_colname_and_valnames(cn: dat.Colname, encoding_to_valname: arr.Namer) -> Noomnames:
    result = noomnames_empty()
    for encoding in range(0, encoding_to_valname.len()):
        valname = encoding_to_valname.name_from_key(encoding)
        s = f'{cn.string()}_is_{valname}'
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
        self.m_row_to_col_to_value.assert_ok()
        if self.m_row_to_col_to_value.num_rows() > 0:
            assert self.noomnames().len() == self.m_row_to_col_to_value.num_cols()

    def noomnames(self) -> Noomnames:
        return self.m_noomnames

    def add_row(self, z: arr.Floats):
        self.m_row_to_col_to_value.add_row(z)

    def explain(self):
        print(self.pretty_string())

    def pretty_strings(self) -> arr.Strings:
        return self.datset().pretty_strings()

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def datset(self) -> dat.Datset:
        result = dat.datset_empty(self.num_rows())
        for i in range(0, self.num_cols()):
            nc = self.named_column(i)
            result.add(nc)
        return result

    def num_cols(self) -> int:
        return self.m_noomnames.len()

    def named_column(self, c: int) -> dat.NamedColumn:
        cn = self.noomname(c).colname()
        col = self.column(c)
        return dat.named_column_create(cn, col)

    def noomname(self, c: int) -> Noomname:
        return self.noomnames().noomname(c)

    def column(self, c: int):
        return dat.column_from_floats(self.col(c))

    def col(self, c: int) -> arr.Floats:
        return self.fmat().column(c)

    def fmat(self) -> arr.Fmat:
        return self.m_row_to_col_to_value

    def loosely_equals(self, other) -> bool:
        assert isinstance(other, Noomset)
        if not self.noomnames().equals(other.noomnames()):
            return False
        return self.fmat().loosely_equals(other.fmat())

    def num_rows(self) -> int:
        return self.fmat().num_rows()

    def row(self, r: int) -> arr.Floats:
        return self.fmat().row(r)


def noomset_empty(nns: Noomnames) -> Noomset:
    return Noomset(nns, arr.fmat_empty())


def categorical_noomname(cn: dat.Colname, vn: dat.Valname) -> Noomname:
    s = cn.string() + '_is_' + vn.string()
    return noomname_create(s)


def categorical_noomnames(cn: dat.Colname, encoding_to_valname: dat.Valnames) -> Noomnames:
    nns = noomnames_empty()
    for enc in range(0, encoding_to_valname.len()):
        nns.add(categorical_noomname(cn, encoding_to_valname.valname(enc)))
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
    for i in range(0, encoding_to_valname.len()):
        s = encoding_to_valname.valname(i).string()
        assert not nm.contains(s)
        nm.add(s)
    return nm


def valname_encoder_create(encoding_to_valname: dat.Valnames) -> ValnameEncoder:
    nm = namer_from_encoding_to_valname(encoding_to_valname)
    return ValnameEncoder(nm)


class CatTransformer:
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

    def num_encodings(self) -> int:
        return self.m_valname_encoder.num_encodings()

    def encoding_from_valname(self, vn: dat.Valname) -> tuple[int, bool]:
        return self.m_valname_encoder.encoding(vn)


def cat_transformer_create(cn: dat.Colname, encoding_to_valname: dat.Valnames) -> CatTransformer:
    return CatTransformer(cn, encoding_to_valname)


def cat_transformer_from_named_column(cn: dat.Colname, cts: dat.Cats) -> CatTransformer:
    value_to_frequency = cts.histogram()
    mcv = value_to_frequency.argmax()
    encoding_to_valname = dat.valnames_empty()
    for v in range(0, cts.num_values()):
        if v != mcv:
            encoding_to_valname.add(cts.valname_from_value(v))

    return cat_transformer_create(cn, encoding_to_valname)


class Transtype(enum.Enum):
    constant = 0
    bool = 1
    float = 2
    categorical = 3


def noomnames_singleton(nn: Noomname) -> Noomnames:
    nns = noomnames_empty()
    nns.add(nn)
    return nns


class FloatTransformer:
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


def float_transformer_create(nn: Noomname, fs: arr.Floats) -> FloatTransformer:
    return FloatTransformer(nn, fs)


class BoolTransformer:
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


def bool_transformer_create(nn: Noomname) -> BoolTransformer:
    return BoolTransformer(nn)


def bool_transformer_from_named_column(cn: dat.Colname) -> BoolTransformer:
    return bool_transformer_create(noomname_from_colname(cn))


class Transformer:
    def __init__(self, tt: Transtype, mc: bas.Maybeint, data):
        self.m_maybecol = mc
        self.m_transtype = tt
        self.m_data = data
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_maybecol, bas.Maybeint)
        assert isinstance(self.m_transtype, Transtype)
        tt = self.m_transtype

        assert (tt == Transtype.constant) == self.m_maybecol.is_undefined()

        if tt == Transtype.categorical:
            assert isinstance(self.m_data, CatTransformer)
            self.m_data.assert_ok()
        elif tt == Transtype.float:
            assert isinstance(self.m_data, FloatTransformer)
            self.m_data.assert_ok()
        elif tt == Transtype.bool:
            assert isinstance(self.m_data, BoolTransformer)
            self.m_data.assert_ok()
        elif tt == Transtype.constant:
            assert self.m_data is None
        else:
            bas.my_error("bad Transtype")

    def noomnames(self) -> Noomnames:
        tt = self.m_transtype
        if tt == Transtype.categorical:
            return self.cat_transformer().noomnames()
        elif tt == Transtype.float:
            return noomnames_singleton(self.float_transformer().noomname())
        elif tt == Transtype.bool:
            return noomnames_singleton(self.bool_transformer().noomname())
        elif tt == Transtype.constant:
            return noomnames_singleton(noomname_create("constant"))
        else:
            bas.my_error("bad Transtype")

    def cat_transformer(self) -> CatTransformer:
        assert self.transtype() == Transtype.categorical
        return self.m_data

    def float_transformer(self) -> FloatTransformer:
        assert self.transtype() == Transtype.float
        return self.m_data

    def bool_transformer(self) -> BoolTransformer:
        assert self.transtype() == Transtype.bool
        return self.m_data

    def transtype(self) -> Transtype:
        return self.m_transtype

    def transform_row(self, rw: dat.Row) -> arr.Floats:
        tt = self.transtype()
        if tt == Transtype.constant:
            return arr.floats_singleton(1.0)
        else:
            col, ok = self.col()
            assert ok
            a = rw.atom(col)
            if tt == Transtype.categorical:
                return self.cat_transformer().transform_valname(a.valname())
            elif tt == Transtype.float:
                return self.float_transformer().transform_float(a.float())
            elif tt == Transtype.bool:
                return self.bool_transformer().transform_bool(a.bool())
            else:
                bas.my_error("bad coltype")

    def col(self) -> tuple[int, bool]:
        return self.m_maybecol.int()


def noomname_from_colname(cn: dat.Colname) -> Noomname:
    return noomname_create(cn.string())


def float_transformer_from_named_column(cn: dat.Colname, fs: arr.Floats) -> FloatTransformer:
    return float_transformer_create(noomname_from_colname(cn), fs)


def transformer_from_named_column(ds: dat.Datset, col: int) -> Transformer:
    nc = ds.named_column(col)
    ct = nc.coltype()
    cn = nc.colname()
    c = nc.column()
    maybe_col = bas.maybeint_defined(col)
    if ct == dat.Coltype.categorical:
        return Transformer(Transtype.categorical, maybe_col, cat_transformer_from_named_column(cn, c.cats()))
    elif ct == dat.Coltype.float:
        return Transformer(Transtype.float, maybe_col, float_transformer_from_named_column(cn, c.floats()))
    elif ct == dat.Coltype.bool:
        return Transformer(Transtype.bool, maybe_col, bool_transformer_from_named_column(cn))
    else:
        bas.my_error("bad Transtype")


class Transformers:
    def __init__(self, tfs: list[Transformer]):
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
        for i in range(0, self.len()):
            tf = self.transformer(i)
            result.append(tf.noomnames())
        return result

    def transformer(self, i) -> Transformer:
        assert 0 <= i < self.len()
        return self.m_transformers[i]

    def len(self) -> int:
        return len(self.m_transformers)

    def transform_row(self, rw: dat.Row) -> arr.Floats:
        result = arr.floats_empty()
        for i in range(0, self.len()):
            t = self.transformer(i)
            fs = t.transform_row(rw)
            result.append(fs)
        return result

    def transform_datset(self, ds: dat.Datset) -> Noomset:
        nns = self.noomnames()
        ns = noomset_empty(nns)
        for r in range(0, ds.num_rows()):
            z = self.transform_row(ds.row(r))
            ns.add_row(z)
        return ns


def transformers_empty():
    return Transformers([])


def transformers_singleton(tf: Transformer) -> Transformers:
    result = transformers_empty()
    result.add(tf)
    return result


def transformer_constant_one():
    return Transformer(Transtype.constant, bas.maybeint_undefined(), None)


def transformers_from_datset(ds: dat.Datset) -> Transformers:
    tf = transformers_singleton(transformer_constant_one())
    for c in range(0, ds.num_cols()):
        tf.add(transformer_from_named_column(ds, c))
    return tf


def noomset_from_datset(ds: dat.Datset) -> Noomset:
    tf = transformers_from_datset(ds)
    return tf.transform_datset(ds)


def noomset_default() -> Noomset:
    nn = noomname_create("default")
    nns = noomnames_singleton(nn)
    return noomset_empty(nns)


def noomnames_from_strings(ss: arr.Strings) -> Noomnames:
    print(f'ss(noomnames_from_strings) = {ss.pretty_string()}')
    nns = noomnames_empty()
    for i in range(0, ss.len()):
        s = ss.string(i)
        print(f's[{i}] = [{s}]')
        nn = noomname_create(s)
        print(f'nn[{i}] = {nn.string()}')
        nns.add(nn)

    return nns


def noomset_from_smat(sm: datset.amarrays.Smat) -> tuple[Noomset, bas.Errmess]:
    if sm.num_rows() < 2:
        return noomset_default(), bas.errmess_error("Need at least 2 rows")

    if sm.num_cols() < 1:
        return noomset_default(), bas.errmess_error("Need at least 1 column")

    nns = noomnames_from_strings(sm.row(0))

    result = noomset_empty(nns)

    for r in range(1, sm.num_rows()):
        fs, ok = arr.floats_from_strings(sm.row(r))
        if not ok:
            em = bas.errmess_error(f'Error parsing line {r}: not a CSV list of floats')
            return noomset_default(), em

        if fs.len() != nns.len():
            s = f"""Error parsing line {r}:" 
            " Expected {nns.len()} floats, but found {fs.len()}."""
            em = bas.errmess_error(s)
            return noomset_default(), em

        result.add_row(fs)
    return result, bas.errmess_ok()


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
