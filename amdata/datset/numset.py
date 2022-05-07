import enum

import datset.dset as ds
import datset.ambasic as bas
import datset.amarrays as arr

class Noomname:
    def __init__(self,name_as_str: str):
        self.m_name = name_as_str
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_name,str)
        assert not len(str) == 0

class Noomnames:
    def __init__(self,li: list[Noomname]):
        self.m_noomnames = li
        self.assert_ok()

    def assert_ok(self):
        for nn in self.m_noomnames:
            assert isinstance(nn,Noomname)
            nn.assert_ok()

        assert not self.strings().contains_duplicates()

    def add(self, nn:Noomname):
        self.m_noomnames.append(nn)

    def len(self)->int:
        return len(self.m_noomnames)

    def append(self, nns):
        assert isinstance(self,Noomnames)
        for i in range(0,nns.len()):
            self.add(nns.noomname(i))

def noomnames_empty()->Noomnames:
    return Noomnames([])


def noomname_create(name_as_string:str)->Noomname:
    return Noomname(name_as_string)


def noomnames_from_colname_and_valnames(cn: ds.Colname, encoding_to_valname: arr.Namer)->Noomnames:
    result = noomnames_empty()
    for encoding in range(0,encoding_to_valname.len()):
        valname = encoding_to_valname.name_from_key(encoding)
        s = f'{cn.string()}_is_{valname}'
        result.add(noomname_create(s))
    return result




class StringTransformer:
    def __init__(self,nns:Noomnames,e2n: arr.Namer):
        self.m_noomnames = nns
        self.m_encoding_to_valname = e2n
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_noomnames,Noomnames)
        self.m_noomnames.assert_ok()
        assert isinstance(self.m_encoding_to_valname,arr.Namer)
        self.m_encoding_to_valname.assert_ok()
        assert self.m_noomnames.len() == self.m_encoding_to_valname.len()

    def noomnames(self)->Noomnames:
        return self.m_noomnames


def string_transformer_create(nns: Noomnames, encoding_to_valname: arr.Namer)->StringTransformer:
    return StringTransformer(nns,encoding_to_valname)


def string_transformer_from_named_column(cn:ds.Colname,row_to_valname:arr.Strings)->StringTransformer:
    vc_to_name = arr.namer_empty()
    vc_to_frequency = arr.ints_empty()
    for r in range(0,row_to_valname.len()):
        s = row_to_valname.string(r)
        vc,ok = vc_to_name.key_from_name(s)
        if ok:
            vc_to_frequency.increment(vc)
        else:
            vc = vc_to_name.len()
            vc_to_name.add(s)
            vc_to_frequency.add(1)
            assert vc_to_name.name_from_key(vc)==s
            assert vc_to_frequency.len() == vc+1

    mcv = vc_to_frequency.argmax()

    vc_to_encoding = arr.ints_empty()
    encoding_to_valname = arr.namer_empty()

    for vc in range(0,vc_to_frequency.len()):
        encoding: int
        if vc == mcv:
            encoding = -777
        else:
            encoding = encoding_to_valname.len()
            valname = vc_to_name.name_from_key(vc)
            assert not encoding_to_valname.contains(valname)
            encoding_to_valname.add(valname)

        vc_to_encoding.add(encoding)

    for vc in range(0,vc_to_encoding.len()):
        encoding = vc_to_encoding.int(vc)
        if vc==mcv:
            assert encoding < 0
        else:
            assert vc_to_name.name_from_key(vc) == encoding_to_valname.name_from_key(encoding)

    nns = noomnames_from_colname_and_valnames(cn,encoding_to_valname)

    return string_transformer_create(nns,encoding_to_valname)

class Transtype(enum.Enum):
    bool = 0
    float = 1
    string = 2


def noomnames_singleton(nn:Noomname)->Noomnames:
    nns = noomnames_empty()
    nns.add(nn)
    return nns

class FloatTransformer:
    def __init__(self,nn:Noomname,fs:arr.Floats):
        self.m_noomname = nn
        self.m_interval = fs.extremes()
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_noomname,Noomname)
        self.m_noomname.assert_ok()
        assert isinstance(self.m_interval,bas.Interval)
        self.m_interval.assert_ok()

    def noomname(self)->Noomname:
        return self.m_noomname


def float_transformer_create(nn:Noomname, fs:arr.Floats)->FloatTransformer:
    return FloatTransformer(nn,fs)

class BoolTransformer:
    def __init__(self,nn:Noomname):
        self.m_noomname = nn
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_noomname,Noomname)
        self.m_noomname.assert_ok()

    def noomname(self)->Noomname:
        return self.m_noomname


def bool_transformer_create(nn:Noomname)->BoolTransformer:
    return BoolTransformer(nn)


def bool_transformer_from_named_column(cn:ds.Colname)->BoolTransformer:
    return bool_transformer_create(noomname_from_colname(cn))




class Transformer:
    def __init__(self,tt:Transtype,data):
        self.m_transtype = tt
        self.m_data = data
        self.assert_ok()

    def assert_ok(self):
        tt = self.m_transtype
        if tt == Transtype.string:
            assert isinstance(self.m_data,StringTransformer)
            self.m_data.assert_ok()
        elif tt == Transtype.float:
            assert isinstance(self.m_data,FloatTransformer)
            self.m_data.assert_ok()
        elif tt == Transtype.bool:
            assert isinstance(self.m_data,BoolTransformer)
            self.m_data.assert_ok()
        else:
            bas.my_error("bad Transtype")

    def noomnames(self)->Noomnames:
        tt = self.m_transtype
        if tt == Transtype.string:
            return self.string_transformer().noomnames()
        elif tt == Transtype.float:
            return noomnames_singleton(self.float_transformer().noomname())
        elif tt == Transtype.bool:
            return noomnames_singleton(self.bool_transformer().noomname())
        else:
            bas.my_error("bad Transtype")

    def string_transformer(self)->StringTransformer:
        assert self.transtype() == Transtype.string
        return self.m_data

    def float_transformer(self)->FloatTransformer:
        assert self.transtype() == Transtype.float
        return self.m_data

    def bool_transformer(self)->BoolTransformer:
        assert self.transtype() == Transtype.bool
        return self.m_data

    def transtype(self)->Transtype:
        return self.m_transtype


def noomname_from_colname(cn:ds.Colname)->Noomname:
    return noomname_create(cn.string())


def float_transformer_from_named_column(cn:ds.Colname,fs:arr.Floats)->FloatTransformer:
    return float_transformer_create(noomname_from_colname(cn),fs)


def transformer_from_named_column(nc:ds.NamedColumn)->Transformer:
    ct = nc.coltype()
    cn = nc.colname()
    c = nc.column()
    if ct == ds.Coltype.string:
        return Transformer(Transtype.string,string_transformer_from_named_column(cn,c.strings()))
    elif ct == ds.Coltype.float:
        return Transformer(Transtype.float,float_transformer_from_named_column(cn,c.floats()))
    elif ct == ds.Coltype.bool:
        return Transformer(Transtype.bool,bool_transformer_from_named_column(cn))
    else:
        bas.my_error("bad Transtype")


class Transformers:
    def __init__(self,tfs:list[Transformer]):
        self.m_transformers = tfs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_transformers,list)
        for tf in self.m_transformers:
            assert isinstance(tf,Transformer)
            tf.assert_ok()

    def add(self, tf:Transformer):
        self.m_transformers.append(tf)

    def noomnames(self)->Noomnames:
        result = noomnames_empty()
        for i in range(0,self.len()):
            tf = self.transformer(i)
            result.append(tf.noomnames())
        return result

    def transformer(self, i)->Transformer:
        assert 0 <= i < self.len()
        return self.m_transformers[i]

    def len(self)->int:
        return len(self.m_transformers)

    def transform_row(self, rw:ds.Row)->arr.Floats:
        assert rw.len()==self.len()
        result = arr.floats_empty()
        for i in range(0,self.len()):
            t = self.transformer(i)
            fs = t.transform_atom(rw.atom(i))
            result.append(fs)
        return result



def transformers_empty():
    return Transformers([])


def transformers_from_datset(ds:ds.Datset)->Transformers:
    tf = transformers_empty()
    for c in range(0,ds.num_cols()):
        tf.add(transformer_from_named_column(ds.named_column(c)))
    return tf


class Noomset:
    def __init__(self,nns:Noomnames,fm:arr.Fmat):
        self.m_noomnames = nns
        self.m_row_to_col_to_value = fm
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_noomnames,Noomnames)
        self.m_noomnames.assert_ok()
        assert isinstance(self.m_row_to_col_to_value,arr.Fmat)
        self.m_row_to_col_to_value.assert_ok()
        assert self.noomnames().len() == self.m_row_to_col_to_value.num_cols()

    def noomnames(self)->Noomnames:
        return self.m_noomnames

    def add_row(self, z:arr.Floats):
        self.m_row_to_col_to_value.add_row(z)


def noomset_empty(nns:Noomnames)->Noomset:
    return Noomset(nns,arr.fmat_empty())


def noomset_from_datset(ds:ds.Datset)->Noomset:
    tf = transformers_from_datset(ds)
    nns = tf.noomnames()
    ns = noomset_empty(nns)
    for r in range(0,ds.num_rows()):
        z = tf.transform_row(ds.row(r))
        ns.add_row(z)
    return ns
