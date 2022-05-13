import enum
import sys
from typing import TextIO

import datset.amarrays as arr
import datset.ambasic as bas
import datset.amcsv as csv
import requests


class Coltype(enum.Enum):
    bool = 0
    float = 1
    string = 2


class Colname:
    def __init__(self, s: str):
        self.m_string = s
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_string, str)
        assert self.m_string != ""
        assert not bas.string_contains(self.m_string, bas.character_space())

    def string(self) -> str:
        return self.m_string

    def equals(self, cn) -> bool:
        assert isinstance(cn, Colname)
        return self.m_string == cn.string()


class Atom:
    def __init__(self, ct: Coltype, data):
        self.m_coltype = ct
        self.m_data = data
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_coltype, Coltype)
        ct = self.m_coltype
        if ct == Coltype.string:
            assert isinstance(self.m_data, str)
        elif ct == Coltype.float:
            assert isinstance(self.m_data, float)
        elif ct == Coltype.bool:
            assert isinstance(self.m_data, bool)
        else:
            bas.my_error("bad coltype")

    def coltype(self) -> Coltype:
        return self.m_coltype

    def string(self) -> str:
        assert self.coltype() == Coltype.string
        assert isinstance(self.m_data, str)
        return self.m_data

    def float(self) -> float:
        assert self.coltype() == Coltype.float
        assert isinstance(self.m_data, float)
        return self.m_data

    def bool(self) -> bool:
        assert self.coltype() == Coltype.bool
        assert isinstance(self.m_data, bool)
        return self.m_data


def atom_from_string(s: str) -> Atom:
    return Atom(Coltype.string, s)


def atom_from_float(f: float) -> Atom:
    return Atom(Coltype.float, f)


def atom_from_bool(b: bool) -> Atom:
    return Atom(Coltype.bool, b)


class Column:
    def __init__(self, ct: Coltype, items: any):
        self.m_coltype = ct
        self.m_list = items
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_coltype, Coltype)
        if self.m_coltype == Coltype.bool:
            assert isinstance(self.m_list, arr.Bools)
            self.m_list.assert_ok()
        elif self.m_coltype == Coltype.float:
            assert isinstance(self.m_list, arr.Floats)
            self.m_list.assert_ok()
        elif self.m_coltype == Coltype.string:
            assert isinstance(self.m_list, arr.Strings)
            self.m_list.assert_ok()
        else:
            bas.my_error("bad column type")

    def num_rows(self) -> int:
        return self.m_list.len()

    def value_as_string(self, r: int) -> str:
        if self.coltype() == Coltype.string:
            return self.strings().string(r)
        elif self.coltype() == Coltype.float:
            return self.floats().value_as_string(r)
        else:
            assert self.coltype() == Coltype.bool
            return self.bools().value_as_string(r)

    def coltype(self) -> Coltype:
        return self.m_coltype

    def strings(self) -> arr.Strings:
        assert self.coltype() == Coltype.string
        return self.m_list

    def floats(self) -> arr.Floats:
        assert self.coltype() == Coltype.float
        return self.m_list

    def bools(self) -> arr.Bools:
        assert self.coltype() == Coltype.bool
        return self.m_list

    def string(self, r: int) -> str:
        return self.strings().string(r)

    def float(self, r: int) -> float:
        return self.floats().float(r)

    def bool(self, r: int) -> bool:
        return self.bools().bool(r)

    def atom(self, r: int) -> Atom:
        ct = self.coltype()
        if ct == Coltype.string:
            return atom_from_string(self.string(r))
        elif ct == Coltype.float:
            return atom_from_float(self.float(r))
        elif ct == Coltype.bool:
            return atom_from_bool(self.bool(r))
        else:
            bas.my_error("bad coltype")


class NamedColumn:
    def __init__(self, cn: Colname, col: Column):
        self.m_colname = cn
        self.m_column = col
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_colname, Colname)
        self.m_colname.assert_ok()
        assert isinstance(self.m_column, Column)
        self.m_column.assert_ok()

    def colname(self) -> Colname:
        return self.m_colname

    def column(self) -> Column:
        return self.m_column

    def num_rows(self) -> int:
        return self.column().num_rows()

    def coltype(self) -> Coltype:
        return self.column().coltype()

    def atom(self, r:int)->Atom:
        return self.column().atom(r)


class Colnames:
    def __init__(self, cs: list[Colname]):
        self.m_colnames = cs
        self.assert_ok()

    def contains_duplicates(self) -> bool:
        return self.strings().contains_duplicates()

    def strings(self) -> arr.Strings:
        result = arr.strings_empty()
        for i in range(0, self.len()):
            result.add(self.string(i))
        return result

    def assert_ok(self):
        for cn in self.m_colnames:
            assert isinstance(cn, Colname)
            cn.assert_ok()

    def add(self, cn: Colname):
        self.m_colnames.append(cn)

    def len(self) -> int:
        return len(self.m_colnames)

    def string(self, i: int) -> str:
        return self.colname(i).string()

    def colname(self, i: int) -> Colname:
        assert 0 <= i < self.len()
        return self.m_colnames[i]

    def pretty_string(self) -> str:
        ss = self.strings()
        return ss.concatenate_fancy('{', ',', '}')

    def minus(self, cn: Colname):  # returns Colnames
        result = colnames_empty()
        for i in range(0, self.len()):
            cn_i = self.colname(i)
            if not cn.equals(cn_i):
                result.add(cn_i)
        return result


def colnames_empty() -> Colnames:
    return Colnames([])


def colname_from_string(s: str) -> Colname:
    spa = bas.character_space()
    und = bas.character_create('_')
    return Colname(bas.string_replace(s, spa, und))


def colnames_from_list_of_strings(*strs: str) -> Colnames:
    result = colnames_empty()
    for s in strs:
        assert isinstance(s, str)
        result.add(colname_from_string(s))
    return result


class Row:
    def __init__(self, la: list[Atom]):
        self.m_atoms = la
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_atoms, list)
        for a in self.m_atoms:
            assert isinstance(a, Atom)
            a.assert_ok()

    def add(self, a: Atom):
        self.m_atoms.append(a)

    def len(self) -> int:
        return len(self.m_atoms)

    def atom(self, i: int) -> Atom:
        assert 0 <= i < self.len()
        return self.m_atoms[i]


def row_empty() -> Row:
    return Row([])


class Datset:
    def __init__(self, ncs: list[NamedColumn]):
        self.m_named_columns = ncs
        self.assert_ok()

    def string(self, r: int, c: int) -> str:
        return self.column(c).string(r)

    def float(self, r: int, c: int) -> float:
        return self.column(c).float(r)

    def bool(self, r: int, c: int) -> bool:
        return self.column(c).bool(r)

    def assert_ok(self):
        for nc in self.m_named_columns:
            assert isinstance(nc, NamedColumn)
            nc.assert_ok()
        assert not self.colnames().contains_duplicates()

    def colnames(self) -> Colnames:
        result = colnames_empty()
        for nc in self.m_named_columns:
            result.add(nc.colname())
        return result

    def pretty_strings(self) -> arr.Strings:
        return self.strings_array().pretty_strings()

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('','\n','')

    def strings_array(self) -> arr.StringsArray:
        result = arr.strings_array_empty()
        result.add(self.colnames().strings())
        for r in range(0, self.num_rows()):
            result.add(self.row_as_strings(r))

        return result

    def num_rows(self) -> int:
        assert self.num_cols() > 0
        return self.column(0).num_rows()

    def num_cols(self) -> int:
        return self.colnames().len()

    def row_as_strings(self, r: int) -> arr.Strings:
        result = arr.strings_empty()
        for c in range(0, self.num_cols()):
            result.add(self.value_as_string(r, c))
        return result

    def value_as_string(self, r: int, c: int) -> str:
        return self.column(c).value_as_string(r)

    def column(self, c: int) -> Column:
        return self.named_column(c).column()

    def named_column(self, c: int) -> NamedColumn:
        assert 0 <= c < self.num_cols()
        return self.m_named_columns[c]

    def explain(self):
        print(self.pretty_string())

    def colid_from_colname(self, cn: Colname) -> tuple[int, bool]:
        for c in range(0, self.num_cols()):
            if self.colname(c).equals(cn):
                return c, True
        return 0, False

    def colname(self, c: int) -> Colname:
        return self.named_column(c).colname()

    def subcols_from_ints(self, cs: arr.Ints):  # returns Datset
        ds = datset_empty()
        for i in range(0, cs.len()):
            ds.add(self.named_column(cs.int(i)))
        return ds

    def subcols(self, *strs: str):  # returns Datset
        cns = colnames_from_list_of_strings(*strs)
        print(f'colnames = {cns.pretty_string()}')
        cis, ok = self.colids_from_colnames(cns)
        assert ok
        return self.subcols_from_ints(cis)

    def colids_from_colnames(self, cns: Colnames) -> tuple[arr.Ints, bool]:
        result = arr.ints_empty()
        for i in range(0, cns.len()):
            i, ok = self.colid_from_colname(cns.colname(i))
            if not ok:
                return arr.ints_empty(), False
            result.add(i)
        return result, True

    def contains_colname(self, cn: Colname) -> bool:
        ci, ok = self.colid_from_colname(cn)
        return ok

    def add(self, nc: NamedColumn):
        if self.num_cols() > 0:
            assert nc.num_rows() == self.num_rows()

        assert not self.contains_colname(nc.colname())

        self.m_named_columns.append(nc)

    def row(self, r: int) -> Row:
        result = row_empty()
        for c in range(0, self.num_cols()):
            a = self.atom(r, c)
            result.add(a)
        return result

    def atom(self, r: int, c: int) -> Atom:
        return self.column(c).atom(r)

    def colid_from_string(self, colname_as_string: str) -> tuple[int, bool]:
        cn = colname_from_string(colname_as_string)
        return self.colid_from_colname(cn)

    def named_column_from_string(self, colname_as_string: str) -> NamedColumn:
        col, ok = self.colid_from_string(colname_as_string)
        assert ok
        return self.named_column(col)

    def without_column(self, exclude_me: NamedColumn):  # returns Datset
        return self.without_colname(exclude_me.colname())

    def without_colname(self, exclude_me: Colname):  # returns Datset
        col, ok = self.colid_from_colname(exclude_me)
        assert ok
        return self.without_colid(col)

    def without_colid(self, col: int):  # returns datset
        result = datset_empty()
        for c in range(0, self.num_cols()):
            if not c == col:
                result.add(self.named_column(c))
        return result


def datset_empty() -> Datset:
    return Datset([])


def is_legal_datid(datid_as_string: str) -> bool:
    assert isinstance(datid_as_string, str)
    return len(datid_as_string) > 0


class Filename:
    def __init__(self, s: str):
        self.m_string = s
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_string, str)
        assert self.string() != ""

    def open(self, readwrite: str) -> tuple[TextIO, bool]:
        try:
            f = open(self.string(), readwrite)
        except FileNotFoundError:
            return TextIO(), False

        return f, True

    def string(self) -> str:
        return self.m_string


def is_legal_filename(f_name: str) -> bool:
    return len(f_name) > 0


def filename_default() -> Filename:
    return Filename("default.txt")


def filename_from_string(f_name: str) -> tuple[Filename, bas.Errmess]:
    if is_legal_filename(f_name):
        return Filename(f_name), bas.errmess_ok()
    else:
        return filename_default(), bas.errmess_error(f"{f_name} is not a legal filename")


class RowIndexedSmat:
    def __init__(self, first_row: arr.Strings):
        self.m_row_to_col_to_string = arr.strings_array_empty()
        self.m_row_to_col_to_string.add(first_row)
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_row_to_col_to_string, arr.StringsArray)
        assert self.num_rows() > 0
        assert self.num_cols() > 0

    def num_cols(self) -> int:
        assert self.num_rows() > 0
        return self.m_row_to_col_to_string.strings(0).len()

    def column(self, c: int) -> arr.Strings:
        assert 0 <= c < self.num_cols()
        result = arr.strings_empty()
        for r in range(0, self.num_rows()):
            result.add(self.string(r, c))
        return result

    def num_rows(self) -> int:
        return self.m_row_to_col_to_string.len()

    def add(self, ss: arr.Strings):
        assert self.num_cols() == ss.len()
        self.m_row_to_col_to_string.add(ss)

    def string(self, r: int, c: int) -> str:
        assert 0 <= r < self.num_rows()
        assert 0 <= c < self.num_cols()
        return self.m_row_to_col_to_string.strings(r).string(c)

    def strings_from_row(self, r: int) -> arr.Strings:
        assert 0 <= r < self.num_rows()
        return self.m_row_to_col_to_string.strings(r)

    def pretty_strings(self)->arr.Strings:
        return self.m_row_to_col_to_string.pretty_strings()

    def pretty_string(self)->str:
        return self.pretty_strings().concatenate_fancy('','\n','')


def row_indexed_smat_transpose(ris: RowIndexedSmat) -> RowIndexedSmat:
    assert ris.num_cols() > 0

    result = row_indexed_smat_singleton(ris.column(0))

    for c in range(1, ris.num_cols()):
        result.add(ris.column(c))

    result.assert_ok()
    return result


class Smat:
    def __init__(self, row_to_col_to_string: RowIndexedSmat):
        self.m_row_to_col_to_string = row_to_col_to_string
        self.m_col_to_row_to_string = row_indexed_smat_transpose(row_to_col_to_string)
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_row_to_col_to_string, RowIndexedSmat)
        self.m_row_to_col_to_string.assert_ok()
        assert isinstance(self.m_row_to_col_to_string, RowIndexedSmat)
        self.m_col_to_row_to_string.assert_ok()

        assert self.m_row_to_col_to_string.num_rows() == self.m_col_to_row_to_string.num_cols()
        assert self.m_row_to_col_to_string.num_cols() == self.m_col_to_row_to_string.num_rows()

        for r in range(0, self.num_rows()):
            for c in range(0, self.num_cols()):
                assert self.m_row_to_col_to_string.string(r, c) == self.m_col_to_row_to_string.string(c, r)

    def num_rows(self) -> int:
        return self.m_row_to_col_to_string.num_rows()

    def num_cols(self) -> int:
        return self.m_row_to_col_to_string.num_cols()

    def column(self, c: int) -> arr.Strings:
        result = arr.strings_empty()
        for r in range(0, self.num_rows()):
            result.add(self.string(r, c))
        return result

    def string(self, r: int, c: int) -> str:
        return self.row(r).string(c)

    def row(self, r: int) -> arr.Strings:
        assert 0 <= r < self.num_rows()
        return self.m_row_to_col_to_string.strings_from_row(r)

    def pretty_string(self) -> str:
        return self.m_row_to_col_to_string.pretty_string()


class StringsLoadResult:
    def __init__(self, ss, em: bas.Errmess, source_unavailable: bool):
        self.m_strings = ss
        self.m_errmess = em
        self.m_source_unavailable = source_unavailable
        self.assert_ok()

    def has_result(self) -> bool:
        return self.is_ok() or self.has_errmess()

    def result(self) -> tuple[arr.Strings, bas.Errmess]:
        assert self.has_result()
        return self.m_strings, self.m_errmess

    def assert_ok(self):
        assert self.m_strings is None or isinstance(self.m_strings, arr.Strings)
        assert isinstance(self.m_errmess, bas.Errmess)
        assert isinstance(self.m_source_unavailable, bool)

        n = 0
        if self.m_strings is not None:
            n += 1

        if self.m_errmess.is_error():
            n += 1

        if self.m_source_unavailable:
            n += 1

        assert n == 1

    def has_errmess(self) -> bool:
        return self.m_errmess.is_error()

    def is_ok(self) -> bool:
        return self.m_strings is not None


def strings_load_result_error(em: bas.Errmess) -> StringsLoadResult:
    return StringsLoadResult(None, em, False)


def strings_load_result_no_file():
    return StringsLoadResult(None, bas.errmess_ok(), True)


def strings_load_result_ok(ss: arr.Strings) -> StringsLoadResult:
    return StringsLoadResult(ss, bas.errmess_ok(), False)


def smat_default():
    return smat_unit("default")


def datset_default() -> Datset:
    return datset_empty()


def test_string():
    s = """date,hour,person,is_happy\n
        4/22/22, 15, ann, True\n
        4/22/22, 15, bob robertson, True\n
        4/22/22, 16, jan, False\n
        4/22/22, 09, jan, True\n
        4/22/22, 12, ann, False\n"""
    return s


class Datid:
    def __init__(self, s: str):
        self.m_string = s
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_string, str)
        assert is_legal_datid(self.m_string)

    def string(self) -> str:
        return self.m_string

    def smat_load(self) -> tuple[Smat, bas.Errmess]:
        ss, em = self.strings_load()
        if em.is_error():
            return smat_default(), em

        return smat_from_strings(ss)

    def datset_load(self) -> tuple[Datset, bas.Errmess]:
        sm, em = self.smat_load()
        if em.is_error():
            return datset_default(), em

        return datset_from_smat(sm)

    def strings_load(self) -> tuple[arr.Strings, bas.Errmess]:
        slr = self.strings_load_using_code()
        if slr.has_result():
            return slr.result()

        slr = self.strings_load_result_using_filename()
        if slr.has_result():
            return slr.result()

        slr = self.strings_load_result_using_url()
        if slr.has_result():
            return slr.result()

        return arr.strings_default(), bas.errmess_error(f'Cannot find a data source using this string: {self.string()}')

    def strings_load_result_using_filename(self) -> StringsLoadResult:
        fn, em = filename_from_string(self.string())
        if em.is_error():
            return strings_load_result_error(em)

        f, ok = fn.open('r')
        if not ok:
            return strings_load_result_no_file()

        finished = False
        result = arr.strings_empty()
        current_line = ""
        while not finished:
            c = f.read()
            if not c:
                finished = True
                if current_line != "":
                    result.add(current_line)
            elif c == '\n':
                result.add(current_line)
                current_line = ""
            else:
                current_line += c

        f.close()
        return strings_load_result_ok(result)

    def string_load_using_url(self) -> tuple[str, bool]:
        url = "https://github.com/awmoorepa/accumulate/blob/master/" + self.string() + "?raw=true"
        response = requests.get(url, stream=True)

        if not response.ok:
            return "", False

        result = ""
        for chunk in response.iter_content(chunk_size=1024):
            s = bas.string_from_bytes(chunk)
            print(f'chunk = [{s}]')
            result = result + s

        print(f'result = [{result}]')
        return result, True

    def strings_load_result_using_url(self) -> StringsLoadResult:
        s, ok = self.string_load_using_url()
        if not ok:
            return strings_load_result_no_file()
        return strings_load_result_ok(arr.strings_from_lines_in_string(s))

    def strings_load_using_code(self) -> StringsLoadResult:
        if self.equals_string("test"):
            s = test_string()
            return strings_load_result_ok(arr.strings_from_lines_in_string(s))
        else:
            return strings_load_result_no_file()

    def equals_string(self, test: str) -> bool:
        return self.string() == test


def datid_default() -> Datid:
    return Datid("default")


def datid_from_string(datid_as_string: str) -> tuple[Datid, bas.Errmess]:
    if is_legal_datid(datid_as_string):
        return Datid(datid_as_string), bas.errmess_ok()
    else:
        return datid_default(), bas.errmess_error(f"{datid_as_string} is not a legal filename")


def row_indexed_smat_singleton(first_row: arr.Strings) -> RowIndexedSmat:
    return RowIndexedSmat(first_row)


def row_indexed_smat_unit(s: str) -> RowIndexedSmat:
    return row_indexed_smat_singleton(arr.strings_singleton(s))


def row_indexed_smat_default():
    return row_indexed_smat_unit('default')


def row_indexed_smat_from_strings_array(ssa: arr.StringsArray) -> tuple[RowIndexedSmat, bas.Errmess]:
    if not ssa:
        return row_indexed_smat_default(), bas.errmess_error(
            "Can't make a row indexed smat from an empty array of strings")

    result = row_indexed_smat_singleton(ssa.strings(0))
    for r in range(1, ssa.len()):
        ss = ssa.strings(r)
        if ss.len() != result.num_cols():
            return result, bas.errmess_error(
                f'first row of csv file has {result.num_cols()} items, but row {r} has {ss.len()} items')
        result.add(ss)

    return result, bas.errmess_ok()


def row_indexed_smat_from_strings(ss: arr.Strings) -> tuple[RowIndexedSmat, bas.Errmess]:
    ssa, em = csv.strings_array_from_strings_csv(ss)

    if em.is_error():
        return row_indexed_smat_default(), em

    print(f'ssa =\n{ssa.pretty_string()}')

    return row_indexed_smat_from_strings_array(ssa)


def smat_from_row_indexed_smat(rsm: RowIndexedSmat) -> Smat:
    return Smat(rsm)


def smat_unit(s: str) -> Smat:
    return smat_from_row_indexed_smat(row_indexed_smat_unit(s))


def smat_from_strings(ss: arr.Strings) -> tuple[Smat, bas.Errmess]:
    ris, em = row_indexed_smat_from_strings(ss)
    if em.is_error():
        return smat_default(), em

    return smat_from_row_indexed_smat(ris), bas.errmess_ok()


def named_column_create(cn: Colname, c: Column) -> NamedColumn:
    return NamedColumn(cn, c)


def colname_create(s: str) -> Colname:
    return Colname(s)


def column_from_bools(bs: arr.Bools) -> Column:
    return Column(Coltype.bool, bs)


def column_default() -> Column:
    return column_from_bools(arr.bools_singleton(False))


def named_column_default() -> NamedColumn:
    return named_column_create(colname_create("default"), column_default())


def coltype_from_strings(ss: arr.Strings) -> Coltype:
    assert ss.len() > 0
    could_be_floats = True
    could_be_bools = True

    for i in range(0, ss.len()):
        s = ss.string(i)
        if could_be_floats and not bas.string_is_float(s):
            could_be_floats = False
        if could_be_bools and not bas.string_is_bool(s):
            could_be_bools = False
        if not (could_be_bools or could_be_floats):
            return Coltype.string

    assert could_be_bools != could_be_floats

    if could_be_bools:
        return Coltype.bool
    else:
        return Coltype.float


def column_from_floats(fs: arr.Floats) -> Column:
    return Column(Coltype.float, fs)


def column_of_type_strings(ss: arr.Strings) -> Column:
    return Column(Coltype.string, ss)


def column_from_strings_choosing_coltype(ss: arr.Strings) -> Column:
    ct = coltype_from_strings(ss)
    if ct == Coltype.bool:
        bs, ok = arr.bools_from_strings(ss)
        assert ok
        return column_from_bools(bs)
    elif ct == Coltype.float:
        fs, ok = arr.floats_from_strings(ss)
        assert ok
        return column_from_floats(fs)
    else:
        assert ct == Coltype.string
        return column_of_type_strings(ss)


def named_column_from_strings(ss: arr.Strings) -> tuple[NamedColumn, bas.Errmess]:
    if ss.len() < 2:
        return named_column_default(), bas.errmess_error("A file with named columns needs at least two rows")

    rest = arr.strings_without_first_n_elements(ss, 1)
    c = column_from_strings_choosing_coltype(rest)

    return named_column_create(colname_create(ss.string(0)), c), bas.errmess_ok()


def named_column_from_smat_column(sm: Smat, c: int) -> tuple[NamedColumn, bas.Errmess]:
    return named_column_from_strings(sm.column(c))


def named_columns_from_smat(sm: Smat) -> tuple[list[NamedColumn], bas.Errmess]:
    result = []
    for c in range(0, sm.num_cols()):
        nc, em = named_column_from_smat_column(sm, c)
        if em.is_error():
            return [], em
        result.append(nc)

    return result, bas.errmess_ok()


def datset_from_string(datid_as_string: str) -> tuple[Datset, bas.Errmess]:
    did, em = datid_from_string(datid_as_string)
    if em.is_error():
        return datset_default(), em
    else:
        return did.datset_load()


def load(datid_as_string: str) -> Datset:
    print(f'datid_as_string = {datid_as_string}')
    ds, em = datset_from_string(datid_as_string)

    print(f'em = {em.string()}')

    if em.is_error():
        sys.exit(em.string())

    print(f'ds =\n{ds.pretty_string()}')
    return ds


def datset_from_smat(sm: Smat) -> tuple[Datset, bas.Errmess]:
    ncs, em = named_columns_from_smat(sm)
    if em.is_error():
        return datset_default(), em

    ds = datset_empty()
    for nc in ncs:
        if ds.contains_colname(nc.colname()):
            return datset_default(), bas.errmess_error("datset has multiple columns with same name")
        ds.add(nc)

    return ds, bas.errmess_ok()


def datset_from_strings_csv(ss: arr.Strings) -> tuple[Datset, bas.Errmess]:
    sm, em = smat_from_strings(ss)
    if em.is_error():
        return datset_default(), em

    return datset_from_smat(sm)


def datset_from_multiline_string(s: str) -> tuple[Datset, bas.Errmess]:
    ss = arr.strings_from_lines_in_string(s)
    return datset_from_strings_csv(ss)


def unit_test():
    s = """date,hour,person,is_happy\n
        4/22/22, 15, ann, True\n
        4/22/22, 15, bob robertson, True\n
        4/22/22, 16, jan, False\n
        4/22/22, 09, jan, True\n
        4/22/22, 12, ann, False\n"""

    ds, em = datset_from_multiline_string(s)

    print(f'em = {em.string()}')
    print(f'ds = \n{ds.pretty_string()}')

    assert ds.string(1, 0) == '4/22/22'
    assert ds.string(1, 2) == 'bob robertson'
    assert not ds.bool(2, 3)
    assert ds.num_rows() == 5
    assert ds.num_cols() == 4


def smat_from_multiline_string(s: str) -> tuple[Smat, bas.Errmess]:
    ss = arr.strings_from_lines_in_string(s)
    return smat_from_strings(ss)
