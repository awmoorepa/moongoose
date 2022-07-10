from __future__ import annotations

import enum
import math
import sys
from abc import ABC, abstractmethod

from typing import TextIO, List, Tuple, Iterator

import requests

import datset.amarrays as arr
import datset.ambasic as bas
import datset.amcsv as csv


# class Coltype(enum.Enum):
#     bool = 0
#     float = 1
#     categorical = 2


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


class Valname:
    def __init__(self, s: str):
        self.m_string = s
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_string, str)

    def string(self) -> str:
        return self.m_string

    def equals(self, other) -> bool:  # other is type Valname
        assert isinstance(other, Valname)
        return self.string() == other.string()


class Coltype(enum.Enum):
    bools = 0
    floats = 1
    cats = 2


class ColumnDescription:
    def __init__(self, cn: Colname, ct: Coltype, vns: Valnames):
        self.m_colname = cn
        self.m_coltype = ct
        self.m_valnames = vns
        self.assert_ok()

    def coltype(self) -> Coltype:
        return self.m_coltype

    def valnames(self) -> Valnames:
        assert self.coltype() == Coltype.cats
        return self.m_valnames

    def assert_ok(self):
        assert isinstance(self.m_colname, Colname)
        self.m_colname.assert_ok()
        assert isinstance(self.m_coltype, Coltype)
        assert isinstance(self.m_valnames, Valnames)
        self.m_valnames.assert_ok()
        assert (self.m_valnames.len() == 0) == (self.m_coltype != Coltype.cats)

    def colname(self) -> Colname:
        return self.m_colname

    def value_from_valname(self, target: Valname) -> Tuple[int, bool]:
        return self.valnames().value(target)

    def equals(self, other: ColumnDescription) -> bool:
        if not self.m_colname.equals(other.m_colname):
            return False

        if self.m_coltype != other.coltype():
            return False

        if self.m_coltype == Coltype.cats and not self.m_valnames.equals(other.m_valnames):
            return False

        return True


def valname_default() -> Valname:
    return valname_from_string("plop")


class Atom(ABC):
    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def pretty_string(self) -> str:
        pass

    @abstractmethod
    def float(self) -> float:
        pass

    @abstractmethod
    def valname(self) -> Valname:
        pass

    @abstractmethod
    def bool(self) -> bool:
        pass


class AtomCategorical(Atom):
    def assert_ok(self):
        assert isinstance(self.m_valname, Valname)
        self.m_valname.assert_ok()

    def float(self) -> float:
        bas.my_error("Can't ask a AtomCategorical for a float")
        return -7e77

    def bool(self) -> bool:
        bas.my_error("Can't ask a AtomCategorical for a bool")
        return False

    def __init__(self, v: Valname):
        self.m_valname = v

    def pretty_string(self) -> str:
        return self.m_valname.string()

    def valname(self) -> Valname:
        return self.m_valname


class AtomFloat(Atom):
    def assert_ok(self):
        assert isinstance(self.m_float, float)

    def valname(self) -> Valname:
        bas.my_error("Can't ask an AtomFloat for a valname")
        return valname_default()

    def bool(self) -> bool:
        bas.my_error("Can't ask an AtomFloat for a bool")
        return False

    def __init__(self, f: float):
        self.m_float = f

    def pretty_string(self) -> str:
        return bas.string_from_float(self.m_float)

    def float(self) -> float:
        return self.m_float


class AtomBool(Atom):
    def assert_ok(self):
        assert isinstance(self.m_bool, bool)

    def float(self) -> float:
        bas.my_error("Can't ask an AtomBool for a float")
        return False

    def valname(self) -> Valname:
        bas.my_error("Can't ask an AtomBool for a valname")
        return valname_default()

    def __init__(self, b: bool):
        self.m_bool = b

    def bool(self) -> bool:
        return self.m_bool

    def pretty_string(self) -> str:
        return bas.string_from_bool(self.m_bool)


# class Atom:
#     def __init__(self, ct: Coltype, data):
#         self.m_coltype = ct
#         self.m_data = data
#         self.assert_ok()
#
#     def assert_ok(self):
#         assert isinstance(self.m_coltype, Coltype)
#         ct = self.m_coltype
#         if ct == Coltype.categorical:
#             assert isinstance(self.m_data, Valname)
#         elif ct == Coltype.float:
#             assert isinstance(self.m_data, float)
#         elif ct == Coltype.bool:
#             assert isinstance(self.m_data, bool)
#         else:
#             bas.my_error("bad coltype")
#
#     def coltype(self) -> Coltype:
#         return self.m_coltype
#
#     def valname(self) -> Valname:
#         assert self.coltype() == Coltype.categorical
#         assert isinstance(self.m_data, Valname)
#         return self.m_data
#
#     def float(self) -> float:
#         assert self.coltype() == Coltype.float
#         assert isinstance(self.m_data, float)
#         return self.m_data
#
#     def bool(self) -> bool:
#         assert self.coltype() == Coltype.bool
#         assert isinstance(self.m_data, bool)
#         return self.m_data
#
#     def pretty_string(self) -> str:
#         ct = self.coltype()
#         if ct == Coltype.categorical:
#             return self.valname().string()
#         elif ct == Coltype.float:
#             return bas.string_from_float(self.float())
#         elif ct == Coltype.bool:
#             return bas.string_from_bool(self.bool())
#         else:
#             bas.my_error("bad coltype")


def atom_from_valname(vn: Valname) -> Atom:
    result = AtomCategorical(vn)
    assert isinstance(result, AtomCategorical)
    assert isinstance(result, Atom)
    return result


def atom_from_float(f: float) -> Atom:
    result = AtomFloat(f)
    assert isinstance(result, AtomFloat)
    assert isinstance(result, Atom)
    return result


def atom_from_bool(b: bool) -> Atom:
    return AtomBool(b)


class Valnames:
    def __init__(self, li: List[Valname]):
        self.m_valnames = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_valnames, list)
        for vn in self.m_valnames:
            assert isinstance(vn, Valname)
            vn.assert_ok()

    def len(self) -> int:
        return len(self.m_valnames)

    def valname(self, value: int) -> Valname:
        assert 0 <= value < self.len()
        return self.m_valnames[value]

    def add(self, vn: Valname):
        self.m_valnames.append(vn)

    def is_sorted(self) -> bool:
        return self.strings().is_sorted()

    def indexes_of_sorted(self) -> arr.Ints:
        return self.strings().indexes_of_sorted()

    def strings(self) -> arr.Strings:
        result = arr.strings_empty()
        for v in self.range():
            result.add(v.string())
        return result

    def subset(self, indexes: arr.Ints):  # returns Valnames
        result = valnames_empty()
        for index in indexes.range():
            result.add(self.valname(index))
        return result

    def contains_duplicates(self) -> bool:
        return self.strings().contains_duplicates()

    def range(self) -> Iterator[Valname]:
        for vn in self.m_valnames:
            yield vn

    def list(self) -> List[str]:
        return self.strings().list()

    def value(self, target):
        for value, vn in enumerate(self.range()):
            if vn.equals(target):
                return value, True
        return -77, False

    def deep_copy(self) -> Valnames:
        return valnames_from_strings(self.strings().deep_copy())

    def equals(self, other: Valnames) -> bool:
        if self.len() != other.len():
            return False

        for me, ot in zip(self.range(), other.range()):
            if not me.equals(ot):
                return False

        return True


class Cats:
    def __init__(self, row_to_value: arr.Ints, vns: Valnames):
        self.m_row_to_value = row_to_value
        self.m_value_to_valname = vns
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_row_to_value, arr.Ints)
        self.m_row_to_value.assert_ok()
        assert isinstance(self.m_value_to_valname, Valnames)
        self.m_value_to_valname.assert_ok()
        assert not self.m_value_to_valname.contains_duplicates()
        hist = self.m_row_to_value.histogram()
        assert isinstance(hist, arr.Ints)
        assert hist.len() == self.m_value_to_valname.len()
        assert hist.min() > 0

    def values(self) -> arr.Ints:
        return self.m_row_to_value

    def valnames(self) -> Valnames:
        return self.m_value_to_valname

    def value_as_string(self, r) -> str:
        return self.valname_from_row(r).string()

    def valname_from_row(self, r: int) -> Valname:
        return self.valname_from_value(self.value(r))

    def valname_from_value(self, value: int) -> Valname:
        assert 0 <= value < self.num_values()
        return self.valnames().valname(value)

    def num_values(self) -> int:
        return self.valnames().len()

    def value(self, r) -> int:
        assert 0 <= r < self.num_rows()
        return self.m_row_to_value.int(r)

    def num_rows(self) -> int:
        return self.m_row_to_value.len()

    def histogram(self) -> arr.Ints:
        return self.m_row_to_value.histogram()

    def range_valnames_by_row(self) -> Iterator[Valname]:
        for val in self.range_values_by_row():
            yield self.valname_from_value(val)

    def range_values_by_row(self) -> Iterator[int]:
        return self.m_row_to_value.range()

    def range_valnames_by_value(self) -> Iterator[Valname]:
        return self.m_value_to_valname.range()

    def range_values_by_value(self) -> Iterator[int]:
        for v in range(0, self.num_values()):
            yield v

    def subset(self, indexes) -> Cats:
        return cats_from_strings(self.strings_by_row().subset(indexes))

    def strings_by_row(self) -> arr.Strings:
        result = arr.strings_empty()
        for vn in self.range_valnames_by_row():
            result.add(vn.string())
        return result


def cats_create(row_to_value: arr.Ints, vns: Valnames) -> Cats:
    return Cats(row_to_value, vns)


def valnames_empty() -> Valnames:
    return Valnames([])


def valname_from_string(s: str) -> Valname:
    return Valname(s)


def valnames_from_strings(ss: arr.Strings) -> Valnames:
    result = valnames_empty()
    for s in ss.range():
        result.add(valname_from_string(s))
    return result


def cats_from_strings(row_to_string: arr.Strings) -> Cats:
    nm = arr.namer_empty()
    row_to_unsorted_value = arr.ints_empty()
    for vn in row_to_string.range():
        unsorted_val, ok = nm.key_from_name(vn)
        if not ok:
            unsorted_val = nm.len()
            nm.add(vn)
        row_to_unsorted_value.add(unsorted_val)

    unsorted_names_as_strings = nm.names()
    unsorted_value_to_name = valnames_from_strings(unsorted_names_as_strings)

    sorted_value_to_unsorted_value = unsorted_value_to_name.indexes_of_sorted()
    unsorted_value_to_sorted_value = sorted_value_to_unsorted_value.invert_index()
    sorted_value_to_name = unsorted_value_to_name.subset(sorted_value_to_unsorted_value)
    assert sorted_value_to_name.is_sorted()

    row_to_sorted_value = arr.ints_empty()
    for unsorted_value in row_to_unsorted_value.range():
        sorted_value = unsorted_value_to_sorted_value.int(unsorted_value)
        row_to_sorted_value.add(sorted_value)

    result = cats_create(row_to_sorted_value, sorted_value_to_name)

    if bas.expensive_assertions:
        for original_string, final_valname in zip(row_to_string.range(), result.range_valnames_by_row()):
            assert original_string == final_valname.string()

    return result


# class Column:
#     def __init__(self, ct: Coltype, items: any):
#         self.m_coltype = ct
#         self.m_list = items
#         self.assert_ok()
#
#     def assert_ok(self):
#         assert isinstance(self.m_coltype, Coltype)
#         if self.m_coltype == Coltype.bool:
#             assert isinstance(self.m_list, arr.Bools)
#             self.m_list.assert_ok()
#         elif self.m_coltype == Coltype.float:
#             assert isinstance(self.m_list, arr.Floats)
#             self.m_list.assert_ok()
#         elif self.m_coltype == Coltype.categorical:
#             assert isinstance(self.m_list, Cats)
#             self.m_list.assert_ok()
#         else:
#             bas.my_error("bad column type")
#
#     def num_rows(self) -> int:
#         assert isinstance(self.m_coltype, Coltype)
#         if self.m_coltype == Coltype.bool:
#             assert isinstance(self.m_list, arr.Bools)
#             return self.m_list.len()
#         elif self.m_coltype == Coltype.float:
#             assert isinstance(self.m_list, arr.Floats)
#             return self.m_list.len()
#         elif self.m_coltype == Coltype.categorical:
#             assert isinstance(self.m_list, Cats)
#             self.m_list.assert_ok()
#             return self.m_list.num_rows()
#         else:
#             bas.my_error("bad column type")
#
#     def coltype(self) -> Coltype:
#         return self.m_coltype
#
#     def cats(self) -> Cats:
#         assert self.coltype() == Coltype.categorical
#         return self.m_list
#
#     def floats(self) -> arr.Floats:
#         assert self.coltype() == Coltype.float
#         return self.m_list
#
#     def bools(self) -> arr.Bools:
#         assert self.coltype() == Coltype.bool
#         return self.m_list
#
#     def valname_from_row(self, r: int) -> Valname:
#         return self.cats().valname_from_row(r)
#
#     def float(self, r: int) -> float:
#         return self.floats().float(r)
#
#     def bool(self, r: int) -> bool:
#         return self.bools().bool(r)
#
#     def atom(self, r: int) -> Atom:
#         ct = self.coltype()
#         if ct == Coltype.categorical:
#             return atom_from_valname(self.valname_from_row(r))
#         elif ct == Coltype.float:
#             return atom_from_float(self.float(r))
#         elif ct == Coltype.bool:
#             return atom_from_bool(self.bool(r))
#         else:
#             bas.my_error("bad coltype")
#
#     def valnames(self) -> Valnames:
#         return self.cats().valnames()


def cats_default() -> Cats:
    return cats_create(arr.ints_empty(), valnames_empty())


class Column(ABC):
    def assert_ok(self):
        bas.my_error(f'{self} base class')

    @abstractmethod
    def num_rows(self) -> int:
        pass

    # def coltype(self) -> Coltype:
    #     bas.my_error(f'{self} base class')
    #     return Coltype.bool

    @abstractmethod
    def atom(self, r: int) -> Atom:
        pass

    def floats(self) -> arr.Floats:
        if isinstance(self, ColumnFloats):
            return self.floats()
        else:
            bas.my_error('You called the floats method on a Column not implemented by ColumnFloats')

    def cats(self) -> Cats:
        if isinstance(self, ColumnCats):
            return self.cats()
        else:
            bas.my_error('You called the cats method on a Column not implemented by ColumnCats')

    def bools(self) -> arr.Bools:
        if isinstance(self, ColumnBools):
            return self.bools()
        else:
            bas.my_error('You called the bools method on a Column not implemented by ColumnFloats')

    @abstractmethod
    def range(self):
        pass

    def is_floats(self) -> bool:
        return False  # overridden in ColumnFloats subclass

    def is_cats(self):
        return False

    def is_bools(self):
        return False

    @abstractmethod
    def subset(self, row_indexes: arr.Ints) -> Column:
        pass

    @abstractmethod
    def column_description(self, cn: Colname) -> ColumnDescription:
        pass

    def type_as_string(self) -> str:
        return self.coltype().string()

    @abstractmethod
    def coltype(self):
        pass


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

    # def coltype(self) -> Coltype:
    #     return self.column().coltype()

    def atom(self, r: int) -> Atom:
        return self.column().atom(r)

    def floats(self) -> arr.Floats:
        return self.column().floats()

    def cats(self) -> Cats:
        return self.column().cats()

    def valnames(self) -> Valnames:
        return self.cats().valnames()

    def is_cats(self) -> bool:
        return self.column().is_cats()

    def is_bools(self) -> bool:
        return self.column().is_bools()

    def bools(self) -> arr.Bools:
        return self.column().bools()

    def is_floats(self) -> bool:
        return self.column().is_floats()

    def subset(self, row_indexes: arr.Ints) -> NamedColumn:
        return named_column_create(self.colname(), self.column().subset(row_indexes))

    def column_description(self):
        return self.column().column_description(self.colname())

    def coltype(self) -> Coltype:
        return self.column().coltype()


class Colnames:
    def __init__(self, cs: List[Colname]):
        self.m_colnames = cs
        self.assert_ok()

    def contains_duplicates(self) -> bool:
        return self.strings().contains_duplicates()

    def strings(self) -> arr.Strings:
        result = arr.strings_empty()
        for cn in self.range():
            result.add(cn.string())
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
        for cn_i in self.range():
            if not cn.equals(cn_i):
                result.add(cn_i)
        return result

    def range(self) -> Iterator[Colname]:
        for cn in self.m_colnames:
            yield cn

    def colid_from_colname(self, target: Colname) -> Tuple[int, bool]:
        for c, cln in enumerate(self.range()):
            if cln.equals(target):
                return c, True
        return -77, False


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


class Record:
    def __init__(self, la: List[Atom]):
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

    def string(self) -> str:
        return self.strings().concatenate_fancy('{', ',', '}')

    def strings(self) -> arr.Strings:
        result = arr.strings_empty()
        for a in self.range():
            result.add(a.pretty_string())
        return result

    def range(self) -> Iterator[Atom]:
        for a in self.m_atoms:
            yield a

    def range_floats(self) -> Iterator[float]:
        for a in self.range():
            yield a.float()


def record_empty() -> Record:
    return Record([])


def datset_from_single_named_column(nc: NamedColumn) -> Datset:
    result = datset_empty(nc.num_rows())
    result.add(nc)
    return result


def datset_from_single_column(cn: Colname, co: Column) -> Datset:
    return datset_from_single_named_column(named_column_create(cn, co))


def column_descriptions_empty() -> ColumnDescriptions:
    return ColumnDescriptions([])


class Operator(ABC):
    def assert_ok(self):
        pass  # nothing to do

    @abstractmethod
    def evaluate_record(self, rec: Record) -> Atom:
        pass

    def evaluate_columns(self, cs: Columns) -> Column:
        result = record_empty()
        for rec in cs.range_records():
            a = self.evaluate_record(rec)
            result.add(a)
        return column_from_record(result)


class OpSpec:
    def __init__(self, op: Operator, cs: Columns):
        self.m_operator = op
        self.m_columns = cs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_operator, Operator)
        self.m_operator.assert_ok()
        assert isinstance(self.m_columns, Columns)
        self.m_columns.assert_ok()

    def evaluate(self) -> Column:
        return self.operator().evaluate_columns(self.columns())

    def operator(self) -> Operator:
        return self.m_operator

    def columns(self) -> Columns:
        return self.m_columns


class Columns:
    def __init__(self, n_records: int, li: List[Column]):
        self.m_num_records = n_records
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for c in self.m_list:
            assert isinstance(c, Column)
            c.assert_ok()
            assert c.num_rows() == self.num_records()

    def add(self, c: Column):
        assert self.num_records() == c.num_rows()
        self.m_list.append(c)

    def range_records(self) -> Iterator[Record]:
        for r in range(self.num_records()):
            yield self.record(r)

    def num_records(self) -> int:
        return self.m_num_records

    def record(self, row: int) -> Record:
        result = record_empty()
        for c in self.range_columns():
            result.add(c.atom(row))
        return result

    def range_columns(self) -> Iterator[Column]:
        for c in self.m_list:
            yield c


def opspec_create(op: Operator, cs: Columns) -> OpSpec:
    return OpSpec(op, cs)


class OperatorPlus(Operator):
    def __init__(self):
        pass

    def evaluate_record(self, rec: Record) -> Atom:
        result = 0.0
        for x in rec.range_floats():
            result += x
        return atom_from_float(result)


def columns_empty(n_records: int) -> Columns:
    return Columns(n_records, [])


def columns_default() -> Columns:
    return columns_empty(0)


def opspec_default() -> OpSpec:
    return opspec_create(OperatorPlus(), columns_default())


class OperatorTimes(Operator):
    def evaluate_record(self, rec: Record) -> Atom:
        result = 1.0
        for a in rec.range_floats():
            result *= a
        return atom_from_float(result)


def operator_from_op_name(op_name: str) -> Tuple[Operator, bool]:
    if op_name == '+':
        return OperatorPlus(), True
    elif op_name == '*':
        return OperatorTimes(), True
    else:
        return OperatorPlus(), False


def named_column_from_bools(cn: Colname, bs: arr.Bools) -> NamedColumn:
    return named_column_create(cn, column_from_bools(bs))


class Datset:
    """
    Represents a set of named columns. Each column must be of the same length. At the moment, a
    column can be an array of booleans, an array of floats or an array of categorical values.

    To create a Datset use load(filename), or make a zero-column datset (you must give number
    of records) and then add named columns to it with the ds.add(nc) method.

    A datset can be considered to be a set of NamedColumn. It can equivalently be considered to
    be a set of Record.
    """

    def __init__(self, n_records: int, ncs: List[NamedColumn]):  # n_records must == col length
        self.m_num_rows = n_records
        self.m_named_columns = ncs
        self.assert_ok()

    def valname_from_row(self, r: int, c: int) -> Valname:
        return self.cats(c).valname_from_row(r)

    def float(self, r: int, c: int) -> float:
        return self.floats(c).float2(r)

    def bool(self, r: int, c: int) -> bool:
        return self.bools(c).bool(r)

    def assert_ok(self):
        for nc in self.m_named_columns:
            assert isinstance(nc, NamedColumn)
            nc.assert_ok()
            assert self.m_num_rows == nc.num_rows()

        assert not self.colnames().contains_duplicates()

    def colnames(self) -> Colnames:
        result = colnames_empty()
        for nc in self.m_named_columns:
            result.add(nc.colname())
        return result

    def pretty_strings(self) -> arr.Strings:
        if self.num_cols() == 0:
            return arr.strings_singleton(f'datset with {self.num_records()} row(s) and no columns')
        return self.strings_array().pretty_strings()

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def strings_array(self) -> arr.StringsArray:
        result = arr.strings_array_empty()
        result.add(self.colnames().strings())
        for ss in self.range_records():
            result.add(ss.strings())

        return result

    def num_records(self) -> int:
        return self.m_num_rows

    def num_cols(self) -> int:
        return self.colnames().len()

    def column(self, c: int) -> Column:
        return self.named_column(c).column()

    def named_column(self, c: int) -> NamedColumn:
        assert 0 <= c < self.num_cols()
        return self.m_named_columns[c]

    def explain(self):
        print(self.pretty_string())

    def colid_from_colname(self, cn: Colname) -> Tuple[int, bool]:
        return self.colnames().colid_from_colname(cn)

    def colname(self, c: int) -> Colname:
        return self.named_column(c).colname()

    def subcols_from_ints(self, cs: arr.Ints):  # returns Datset
        ds = datset_empty(self.num_records())
        for c in cs.range():
            ds.add(self.named_column(c))
        return ds

    def contains_colname(self, cn: Colname) -> bool:
        ci, ok = self.colid_from_colname(cn)
        return ok

    def add(self, nc: NamedColumn):
        if self.num_cols() > 0:
            assert nc.num_rows() == self.num_records()

        assert not self.contains_colname(nc.colname())

        self.m_named_columns.append(nc)

    def record(self, r: int) -> Record:
        result = record_empty()
        for c in self.range_columns():
            result.add(c.atom(r))
        return result

    def atom(self, r: int, c: int) -> Atom:
        return self.column(c).atom(r)

    def named_column_from_colname(self, target: Colname) -> Tuple[NamedColumn, bool]:
        for nc in self.range_named_columns():
            if nc.colname().equals(target):
                return nc, True
        return named_column_default(), False

    def without_column(self, exclude_me: NamedColumn):  # returns Datset
        return self.without_colname(exclude_me.colname())

    def without_colname(self, exclude_me: Colname):  # returns Datset
        col, ok = self.colid_from_colname(exclude_me)
        assert ok
        return self.without_colid(col)

    def without_colid(self, col: int):  # returns datset
        result = datset_empty(self.num_records())
        for c in range(self.num_cols()):
            if not c == col:
                result.add(self.named_column(c))
        return result

    def range_records(self) -> Iterator[Record]:
        for r in range(self.num_records()):
            yield self.record(r)

    def range_named_columns(self) -> Iterator[NamedColumn]:
        for nc in self.m_named_columns:
            yield nc

    def range_columns(self) -> Iterator[Column]:
        for nc in self.range_named_columns():
            yield nc.column()

    def subset_with_checks(self, cns: Colnames) -> Tuple[Datset, bas.Errmess]:
        assert isinstance(cns, Colnames)
        result = datset_empty(self.num_records())
        for cn in cns.range():
            nc, ok = self.named_column_from_colname(cn)
            if not ok:
                return datset_default(), bas.errmess_error(
                    f'datset has columns {self.colnames().pretty_string()} and does not contain column named {cn.string()}')
            if result.contains_colname(nc.colname()):
                return datset_default(), bas.errmess_error(f'you cannot repeat column named {cn.string()}')
            result.add(nc)
        return result, bas.errmess_ok()

    def subset(self, *colnames_as_strings: str) -> Datset:
        cns = colnames_from_list_of_strings(*colnames_as_strings)
        result, em = self.subset_with_checks(cns)
        em.abort_if_error()
        return result

    def column_is_floats(self, col: int) -> bool:
        return self.column(col).is_floats()

    def floats(self, col) -> arr.Floats:
        return self.column(col).floats()

    def with_named_column(self, new_nc: NamedColumn):  # returns Datset
        assert self.num_records() == new_nc.num_rows()
        result = datset_empty(self.num_records())
        for nc in self.range_named_columns():
            result.add(nc)
        result.add(new_nc)
        assert isinstance(result, Datset)
        return result

    def without(self, ignore):  # ignore type is Datset and returns Datset
        assert isinstance(ignore, Datset)
        result = datset_empty(self.num_records())
        for nc in self.range_named_columns():
            if not ignore.contains_colname(nc.colname()):
                result.add(nc)
        assert isinstance(result, Datset)
        return result

    def split(self, fraction_in_train: float) -> Tuple[Datset, Datset]:
        train_rows, test_rows = indexes_for_train_and_test(self.num_records(), fraction_in_train)
        return self.subset_records(train_rows), self.subset_records(test_rows)

    def subset_records(self, row_indexes: arr.Ints) -> Datset:
        result = datset_empty(row_indexes.len())
        for nc in self.range_named_columns():
            result.add(nc.subset(row_indexes))
        return result

    def cats(self, c: int) -> Cats:
        return self.column(c).cats()

    def bools(self, c: int) -> arr.Bools:
        return self.column(c).bools()

    def discretize(self, colname_as_string: str, n_buckets: int) -> Datset:
        assert self.num_cols() == 1
        co = self.column(0)
        assert isinstance(co, ColumnFloats)
        return datset_from_single_column(colname_from_string(colname_as_string), co.discretize(n_buckets))

    def binarize(self, colname_as_string: str) -> Datset:
        assert self.num_cols() == 1
        co = self.column(0)
        assert isinstance(co, ColumnFloats)
        return datset_from_single_column(colname_from_string(colname_as_string), co.binarize())

    def appended_with(self, other: Datset) -> Datset:
        assert self.num_records() == other.num_records()
        result = datset_empty(self.num_records())
        for nc in self.range_named_columns():
            result.add(nc)
        for nc in other.range_named_columns():
            if self.contains_colname(nc.colname()):
                bas.my_error(f"Can't append two datasets which share a column name ({nc.colname().string()})")
            result.add(nc)
        return result

    def column_descriptions(self) -> ColumnDescriptions:
        result = column_descriptions_empty()
        for nc in self.range_named_columns():
            result.add(nc.column_description())
        return result

    def add_column_using_function(self, col_name: str, op_name: str, cns: Colnames):
        ops, em = self.opspec_from_op_name(op_name, cns)
        em.abort_if_error()
        self.add_column_using_opspec(colname_from_string(col_name), ops)

    def opspec_from_op_name(self, op_name: str, cns: Colnames) -> Tuple[OpSpec, bas.Errmess]:
        cs, em = self.columns_from_colnames(cns)
        if em.is_error():
            return opspec_default(), em
        op, ok = operator_from_op_name(op_name)
        if not ok:
            return opspec_default(), bas.errmess_error(f'No such operator as {op_name}')

        return opspec_create(op, cs), bas.errmess_ok()

    def add_column_using_opspec(self, cn: Colname, ops: OpSpec):
        self.add(named_column_create(cn, ops.evaluate()))

    def columns_from_colnames(self, cns: Colnames) -> Tuple[Columns, bas.Errmess]:
        #  why not implement this by making a datset and taking the columns? Because we allow duplicate columns
        result = columns_empty(self.num_records())
        for cn in cns.range():
            nc, ok = self.named_column_from_colname(cn)
            if not ok:
                return columns_default(), bas.errmess_error(f'no column named {cn.string()}')
            result.add(nc.column())
        return result, bas.errmess_ok()

    def columns(self) -> Columns:
        result = columns_empty(self.num_records())
        for c in self.range_columns():
            result.add(c)
        return result

    def define_column(self, new_name: str, op_name: str, *colnames_as_strings: str):
        cns = colnames_from_list_of_strings(*colnames_as_strings)
        self.add_column_using_function(new_name, op_name, cns)

    def range_floats(self, colname_as_string: str) -> Iterator[float]:
        nc, ok = self.named_column_from_colname(colname_create(colname_as_string))
        assert ok
        return nc.floats().range2()

    def add_bools_column(self, colname_as_string, bs: arr.Bools):
        self.add(named_column_from_bools(colname_create(colname_as_string), bs))


def indexes_for_train_and_test(n_records: int, fraction_in_train: float) -> Tuple[arr.Ints, arr.Ints]:
    assert n_records > 1
    n_in_train = math.floor(n_records * fraction_in_train)
    if n_in_train < 1:
        n_in_train = 1
    elif n_in_train == n_records:
        n_in_train = n_records - 1

    indexes = arr.ints_random_permutation(n_records)
    train_rows = indexes.first_n_elements(n_in_train).sort()
    test_rows = indexes.last_n_elements(n_records - n_in_train).sort()

    return train_rows, test_rows


def datset_empty(n_records: int) -> Datset:
    return Datset(n_records, [])


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

    def open(self, readwrite: str) -> Tuple[TextIO, bool]:
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


def filename_from_string(f_name: str) -> Tuple[Filename, bas.Errmess]:
    if is_legal_filename(f_name):
        return Filename(f_name), bas.errmess_ok()
    else:
        return filename_default(), bas.errmess_error(f"{f_name} is not a legal filename")


class StringsLoadResult:
    def __init__(self, full_filename: str, ss: arr.Strings, em: bas.Errmess, source_unavailable: bool, method: str):
        self.m_full_filename = full_filename
        self.m_strings = ss
        self.m_errmess = em
        self.m_source_unavailable = source_unavailable
        self.m_method = method
        self.assert_ok()

    def was_successfully_loaded(self) -> bool:
        return self.m_errmess.is_ok() and not self.m_source_unavailable

    def file_not_found(self) -> bool:
        return self.m_source_unavailable

    def file_found_but_unparseable(self) -> bool:
        return self.m_errmess.is_error()

    def result(self) -> arr.Strings:
        assert self.was_successfully_loaded()
        return self.m_strings

    def assert_ok(self):
        assert isinstance(self.m_full_filename, str)
        assert self.m_full_filename != ""
        assert isinstance(self.m_strings, arr.Strings)
        self.m_strings.assert_ok()
        assert isinstance(self.m_errmess, bas.Errmess)
        self.m_errmess.assert_ok()
        assert isinstance(self.m_source_unavailable, bool)
        assert isinstance(self.m_method, str)

        n = 0
        if self.was_successfully_loaded():
            n += 1
        if self.file_not_found():
            n += 1
        if self.file_found_but_unparseable():
            n += 1

        assert n == 1

    def has_errmess(self) -> bool:
        return self.m_errmess.is_error()

    def is_ok(self) -> bool:
        return self.m_strings is not None

    def file_with_path(self) -> str:
        return self.m_full_filename

    def errmess(self) -> bas.Errmess:
        return self.m_errmess

    def should_return(self) -> bool:
        return not self.file_not_found()

    def return_result(self) -> Tuple[arr.Strings, bas.Errmess]:
        report = self.errmess().string() if self.file_found_but_unparseable() else 'successfully loaded'
        print(f'found data using {self.m_method} {self.file_with_path()}. {report}')
        assert not self.file_not_found()
        if self.was_successfully_loaded():
            return self.result(), bas.errmess_ok()
        else:
            assert self.file_found_but_unparseable()
            return arr.strings_empty(), self.errmess()


def strings_load_result_error(em: bas.Errmess, method: str) -> StringsLoadResult:
    return StringsLoadResult("", arr.strings_empty(), em, False, method)


def strings_load_result_no_file(full_filename: str, method: str):
    return StringsLoadResult(full_filename, arr.strings_empty(), bas.errmess_ok(), True, method)


def strings_load_result_ok(full_file_name: str, ss: arr.Strings, method: str) -> StringsLoadResult:
    return StringsLoadResult(full_file_name, ss, bas.errmess_ok(), False, method)


def datset_default() -> Datset:
    return datset_empty(1)


def test_string():
    s = """date,hour,person,weight,is_late\n
        4/22/22, 12, ann, 200,f\n
        4/22/22, 14, bob robertson, 170,f\n
        4/22/22, 14.5, ann, 180,f\n
        4/22/22, 17, jan, 130,t\n
        4/22/22, 19, jan, 130,t\n"""
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

    def smat_load(self) -> Tuple[arr.Smat, bas.Errmess]:
        ss, em = self.strings_load()
        if em.is_error():
            return arr.smat_default(), em

        assert ss.len() > 0

        return csv.smat_from_strings(ss)

    def datset_load(self) -> Tuple[Datset, bas.Errmess]:
        sm, em = self.smat_load()
        if em.is_error():
            return datset_default(), em

        return datset_from_smat(sm)

    def strings_load(self) -> Tuple[arr.Strings, bas.Errmess]:
        slr = self.strings_load_using_code()
        if slr.should_return():
            return slr.return_result()

        slr = self.strings_load_result_using_filename()
        if slr.should_return():
            return slr.return_result()

        slr = self.strings_load_result_using_url()
        if slr.should_return():
            return slr.return_result()

        return arr.strings_default(), bas.errmess_error(f'Cannot find a data source using this string: {self.string()}')

    def strings_load_result_using_filename(self) -> StringsLoadResult:
        fn, em = filename_from_string(self.string())
        if em.is_error():
            return strings_load_result_error(em, 'local file system')

        f, ok = fn.open('r')
        if not ok:
            return strings_load_result_no_file(fn.string(), 'local file system')

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
        return strings_load_result_ok(fn.string(), result, 'local file system')

    def string_load_using_url(self, git_url: str) -> Tuple[str, bool]:
        file_url = git_url + self.string()
        url = file_url + "?raw=true"
        response = requests.get(url, stream=True)

        if not response.ok:
            return "", False

        result = ""
        for chunk in response.iter_content(chunk_size=1024):
            s = bas.string_from_bytes(chunk)
            result = result + s

        return result, True

    def strings_load_result_using_url(self) -> StringsLoadResult:
        git_url = "https://github.com/awmoorepa/accumulate/blob/master/"
        s, ok = self.string_load_using_url(git_url)
        file_url = git_url + self.string()
        if not ok:
            return strings_load_result_no_file(file_url, 'URL')
        return strings_load_result_ok(file_url, arr.strings_from_lines_in_string(s), 'URL')

    def strings_load_using_code(self) -> StringsLoadResult:
        if self.equals_string("test"):
            s = test_string()
            return strings_load_result_ok("test", arr.strings_from_lines_in_string(s), "test code")
        else:
            return strings_load_result_no_file(self.string(), 'test code')

    def equals_string(self, test: str) -> bool:
        return self.string() == test


def datid_default() -> Datid:
    return Datid("default")


def datid_from_string(datid_as_string: str) -> Tuple[Datid, bas.Errmess]:
    if is_legal_datid(datid_as_string):
        return Datid(datid_as_string), bas.errmess_ok()
    else:
        return datid_default(), bas.errmess_error(f"{datid_as_string} is not a legal filename")


def named_column_create(cn: Colname, c: Column) -> NamedColumn:
    return NamedColumn(cn, c)


def colname_create(s: str) -> Colname:
    return Colname(s)


def column_description_create(cn: Colname, ct: Coltype, vns: Valnames) -> ColumnDescription:
    return ColumnDescription(cn, ct, vns)


def column_description_of_type_bools(cn: Colname) -> ColumnDescription:
    return column_description_create(cn, Coltype.bools, valnames_empty())


class ColumnBools(Column):
    def coltype(self) -> Coltype:
        return Coltype.bools

    def column_description(self, cn: Colname) -> ColumnDescription:
        return column_description_of_type_bools(cn)

    def __init__(self, bs: arr.Bools):
        self.m_bools = bs
        self.assert_ok()

    def assert_ok(self):
        assert (isinstance(self.m_bools, arr.Bools))
        self.m_bools.assert_ok()

    def num_rows(self) -> int:
        return self.m_bools.len()

    def atom(self, r: int) -> Atom:
        return atom_from_bool(self.bool(r))

    def bool(self, r: int) -> bool:
        return self.m_bools.bool(r)

    def range(self):
        for b in self.m_bools.range():
            yield atom_from_bool(b)

    def is_bools(self) -> bool:
        return True

    def bools(self) -> arr.Bools:
        return self.m_bools

    def subset(self, indexes: arr.Ints) -> ColumnBools:
        return ColumnBools(self.bools().subset(indexes))


def column_from_bools(bs: arr.Bools) -> ColumnBools:
    return ColumnBools(bs)


def column_default() -> Column:
    return column_from_bools(arr.bools_singleton(False))


def named_column_default() -> NamedColumn:
    return named_column_create(colname_create("default"), column_default())


def column_from_strings(ss: arr.Strings) -> Column:
    bs, ok = arr.bools_from_strings(ss)
    if ok:
        return column_from_bools(bs)

    fs, ok = arr.floats_from_strings(ss)
    if ok:
        return column_from_floats(fs)

    return column_from_cats(cats_from_strings(ss))


def column_description_of_type_floats(cn: Colname) -> ColumnDescription:
    return column_description_create(cn, Coltype.floats, valnames_empty())


def cats_from_discretized_floats(fs: arr.Floats, n_buckets: int) -> Cats:
    rank_to_index = fs.indexes_of_sorted()
    n_elements = fs.len()
    elements_per_bucket_lo = math.floor(n_elements / n_buckets)
    remaining = n_elements - elements_per_bucket_lo * n_buckets
    assert 0 <= remaining < n_buckets
    bucket_number_to_len = arr.ints_empty()
    for bucket_number in range(0, remaining):
        bucket_number_to_len.add(elements_per_bucket_lo + 1)
    for bucket_number in range(remaining, n_buckets):
        bucket_number_to_len.add(elements_per_bucket_lo)

    assert bucket_number_to_len.len() == n_buckets
    assert bucket_number_to_len.min() == elements_per_bucket_lo
    assert bucket_number_to_len.max() <= elements_per_bucket_lo + 1
    assert bucket_number_to_len.sum() == n_elements

    rank_to_bucket_number = arr.ints_empty()
    bucket_number = 0
    index_within_bucket = 0
    for rank in range(n_elements):
        rank_to_bucket_number.add(bucket_number)
        index_within_bucket += 1
        if index_within_bucket >= bucket_number_to_len.int(bucket_number):
            index_within_bucket = 0
            bucket_number += 1

    assert bucket_number == n_buckets
    assert index_within_bucket == 0

    index_to_rank = rank_to_index.invert_index()
    index_to_bucket_number = arr.ints_empty()

    for rank in index_to_rank.range():
        bucket_number = rank_to_bucket_number.int(rank)
        index_to_bucket_number.add(bucket_number)

    vns = valnames_empty()
    for bucket_number in range(n_buckets):
        vns.add(valname_from_string(f'segment{bucket_number}'))

    return cats_create(index_to_bucket_number, vns)


def bools_from_binarized_floats(fs: arr.Floats) -> arr.Bools:
    middle = fs.median()
    result = arr.bools_empty()
    for f in fs.range2():
        result.add(f > middle)
    return result


class ColumnFloats(Column):
    def coltype(self) -> Coltype:
        return Coltype.floats

    def column_description(self, cn: Colname) -> ColumnDescription:
        return column_description_of_type_floats(cn)

    def __init__(self, fs: arr.Floats):
        self.m_floats = fs

    def num_rows(self) -> int:
        return self.m_floats.len()

    def assert_ok(self):
        assert isinstance(self.m_floats, arr.Floats)

    def atom(self, r: int) -> Atom:
        return atom_from_float(self.float(r))

    def float(self, r: int) -> float:
        return self.m_floats.float2(r)

    def floats(self) -> arr.Floats:
        return self.m_floats

    def range(self):  # elements of the range are atoms
        for f in self.m_floats.range2():
            yield atom_from_float(f)

    def is_floats(self) -> bool:
        return True

    def subset(self, indexes: arr.Ints) -> Column:
        return ColumnFloats(self.floats().subset(indexes))

    def discretize(self, n_buckets) -> ColumnCats:
        cs = cats_from_discretized_floats(self.floats(), n_buckets)
        return column_from_cats(cs)

    def binarize(self) -> ColumnBools:
        bs = bools_from_binarized_floats(self.floats())
        return column_from_bools(bs)


def column_from_floats(fs: arr.Floats) -> Column:
    return ColumnFloats(fs)


def column_description_of_type_cats(cn: Colname, vns: Valnames) -> ColumnDescription:
    return column_description_create(cn, Coltype.cats, vns)


class ColumnCats(Column):
    def coltype(self):
        return Coltype.cats

    def column_description(self, cn: Colname) -> ColumnDescription:
        return column_description_of_type_cats(cn, self.cats().valnames())

    def __init__(self, cs: Cats):
        self.m_cats = cs

    def assert_ok(self):
        assert isinstance(self.m_cats, Cats)
        self.m_cats.assert_ok()

    def num_rows(self) -> int:
        return self.m_cats.num_rows()

    def atom(self, r: int) -> Atom:
        return atom_from_valname(self.valname(r))

    def valname(self, r: int) -> Valname:
        return self.m_cats.valname_from_row(r)

    def cats(self) -> Cats:
        return self.m_cats

    def range(self) -> Iterator[Atom]:
        for v in self.m_cats.range_valnames_by_row():
            yield atom_from_valname(v)

    def is_cats(self) -> bool:
        return True

    def subset(self, indexes: arr.Ints) -> ColumnCats:
        return ColumnCats(self.cats().subset(indexes))


def column_from_cats(cs: Cats) -> ColumnCats:
    return ColumnCats(cs)


def named_column_from_strings(ss: arr.Strings) -> Tuple[NamedColumn, bas.Errmess]:
    if ss.len() < 2:
        return named_column_default(), bas.errmess_error("A file with named columns needs at least two rows")

    rest = arr.strings_without_first_n_elements(ss, 1)
    c = column_from_strings(rest)

    return named_column_create(colname_create(ss.string(0)), c), bas.errmess_ok()


def named_column_from_smat_column(sm: arr.Smat, c: int) -> Tuple[NamedColumn, bas.Errmess]:
    return named_column_from_strings(sm.column(c))


def named_columns_from_smat(sm: arr.Smat) -> Tuple[List[NamedColumn], bas.Errmess]:
    result = []
    for ss in sm.range_columns():
        nc, em = named_column_from_strings(ss)
        if em.is_error():
            return [], em
        result.append(nc)

    return result, bas.errmess_ok()


def datset_from_string(datid_as_string: str) -> Tuple[Datset, bas.Errmess]:
    did, em = datid_from_string(datid_as_string)
    if em.is_error():
        return datset_default(), em
    else:
        return did.datset_load()


def load(datid_as_string: str) -> Datset:
    ds, em = datset_from_string(datid_as_string)

    if em.is_error():
        sys.exit(em.string())

    return ds


def datset_from_smat(sm: arr.Smat) -> Tuple[Datset, bas.Errmess]:
    ncs, em = named_columns_from_smat(sm)
    if em.is_error():
        return datset_default(), em

    ds = datset_empty(sm.num_rows() - 1)
    for nc in ncs:
        if ds.contains_colname(nc.colname()):
            return datset_default(), bas.errmess_error("datset has multiple columns with same name")
        ds.add(nc)

    return ds, bas.errmess_ok()


def datset_from_strings_csv(ss: arr.Strings) -> Tuple[Datset, bas.Errmess]:
    sm, em = csv.smat_from_strings(ss)
    if em.is_error():
        return datset_default(), em

    return datset_from_smat(sm)


def datset_from_multiline_string(s: str) -> Tuple[Datset, bas.Errmess]:
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

    assert ds.valname_from_row(1, 0).string() == '4/22/22'
    assert ds.valname_from_row(1, 2).string() == 'bob robertson'
    assert not ds.bool(2, 3)
    assert ds.num_records() == 5
    assert ds.num_cols() == 4


def smat_from_multiline_string(s: str) -> Tuple[arr.Smat, bas.Errmess]:
    ss = arr.strings_from_lines_in_string(s)
    return csv.smat_from_strings(ss)


def column_from_record(r: Record) -> Column:
    assert r.len() > 0
    a0 = r.atom(0)
    if isinstance(a0, AtomBool):
        fs = arr.bools_empty()
        for a in r.range():
            assert isinstance(a, AtomBool)
            fs.add(a.bool())
        return column_from_bools(fs)
    elif isinstance(a0, AtomFloat):
        fs = arr.floats_empty(r.len())
        for a in r.range():
            assert isinstance(a, AtomFloat)
            fs.add(a.float())
        return column_from_floats(fs)
    elif isinstance(a0, AtomCategorical):
        bas.my_error('not implemented (and will be a pain because need to ensure value consistency')
    else:
        bas.my_error('Bad column type')


class RecordArray:
    def __init__(self, li: list[Record]):
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for r in self.m_list:
            assert isinstance(r, Record)
            r.assert_ok()

    def add(self, r: Record):
        self.m_list.append(r)

    def len(self) -> int:
        return len(self.m_list)

    def record(self, i: int) -> Record:
        assert 0 <= i < self.len()
        return self.m_list[i]

    def column(self, c: int) -> Column:
        return column_from_record(self.column_of_atoms(c))

    def column_of_atoms(self, c: int) -> Record:  # using 'Record' of Atoms to represent a column. Sorry.
        result = record_empty()
        for r in self.range():
            result.add(r.atom(c))
        return result

    def range(self) -> Iterator[Record]:
        for r in self.m_list:
            yield r


def record_array_empty() -> RecordArray:
    return RecordArray([])


def colnames_from_strings(ss: arr.Strings) -> Colnames:
    assert isinstance(ss, arr.Strings)
    return colnames_from_list_of_strings(*ss.list())


def named_column_from_floats(cn: Colname, fs: arr.Floats) -> NamedColumn:
    return named_column_create(cn, column_from_floats(fs))


def datset_from_fmat(cns: Colnames, fm: arr.Fmat) -> Datset:
    assert cns.len() == fm.num_cols()
    assert isinstance(fm, arr.Fmat)
    result = datset_empty(fm.num_rows())
    for cn, fs in zip(cns.range(), fm.range_columns()):
        result.add(named_column_from_floats(cn, fs))
    return result


def record_singleton(a: Atom) -> Record:
    result = record_empty()
    result.add(a)
    return result


def record_from_floats(fs: arr.Floats) -> Record:
    r = record_empty()
    for f in fs.range2():
        r.add(atom_from_float(f))
    return r


class ColumnDescriptions:
    def __init__(self, li: List[ColumnDescription]):
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for cd in self.m_list:
            assert isinstance(cd, ColumnDescription)
            cd.assert_ok()

    def column_description(self, index: int) -> ColumnDescription:
        assert 0 <= index < self.len()
        return self.m_list[index]

    def add(self, cd: ColumnDescription):
        self.m_list.append(cd)

    def len(self) -> int:
        return len(self.m_list)

    def colnames(self) -> Colnames:
        result = colnames_empty()
        for cd in self.range():
            result.add(cd.colname())
        return result

    def range(self) -> Iterator[ColumnDescription]:
        for cd in self.m_list:
            yield cd


def column_of_random_unit_floats(n_records: int) -> Column:
    return column_from_floats(arr.floats_random_unit(n_records))


def named_column_of_random_unit_floats(name: str, n_records: int) -> NamedColumn:
    return named_column_create(colname_create(name), column_of_random_unit_floats(n_records))


def datset_of_random_unit_floats(colname: str, n_records: int) -> Datset:
    return datset_from_single_named_column(named_column_of_random_unit_floats(colname, n_records))


class LearnData:
    def __init__(self, inputs: Datset, output: NamedColumn):
        self.m_inputs = inputs
        self.m_output = output
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_inputs, Datset)
        self.m_inputs.assert_ok()
        assert isinstance(self.m_output, NamedColumn)
        self.m_output.assert_ok()

        assert not self.m_inputs.contains_colname(self.m_output.colname())
        assert self.m_inputs.num_records() == self.m_output.num_rows()

    def train_test_split(self, train_fraction: float) -> Tuple[LearnData, LearnData]:
        train_rows, test_rows = indexes_for_train_and_test(self.num_records(), train_fraction)
        return self.subset_records(train_rows), self.subset_records(test_rows)

    def num_records(self) -> int:
        return self.inputs().num_records()

    def inputs(self) -> Datset:
        return self.m_inputs

    def subset_records(self, train_rows: arr.Ints) -> LearnData:
        return learn_data_create(self.inputs().subset_records(train_rows), self.output().subset(train_rows))

    def output(self) -> NamedColumn:
        return self.m_output

    def output_column(self) -> Column:
        return self.output().column()


def learn_data_create(inputs: Datset, output: NamedColumn) -> LearnData:
    return LearnData(inputs, output)


def learn_data_from_datsets(inputs: Datset, output: Datset) -> LearnData:
    assert output.num_cols() == 1
    return learn_data_create(inputs, output.named_column(0))


def datset_of_strings(colname_as_string: str, ss: arr.Strings) -> Datset:
    return datset_from_single_column(colname_from_string(colname_as_string), column_from_strings(ss))
