import datset.ambasic as bas


def is_list_of_bool(bs) -> bool:
    if not isinstance(bs, list):
        return False

    for b in bs:
        if not isinstance(b, bool):
            return False

    return True


def is_list_of_floats(fs):
    if not isinstance(fs, list):
        return False

    for f in fs:
        if not isinstance(f, float):
            return False

    return True


class Bools:
    def __init__(self, bs: list[bool]):
        self.m_bools = bs
        self.assert_ok()

    def assert_ok(self):
        assert is_list_of_bool(self.m_bools)

    def add(self, b):
        self.m_bools.append(b)

    def value_as_string(self, r: int) -> str:
        return bas.string_from_bool(self.bool(r))

    def bool(self, r: int) -> bool:
        assert 0 <= r < self.len()
        return self.m_bools[r]

    def len(self) -> int:
        return len(self.m_bools)


class Floats:
    def __init__(self, fs: list[float]):
        self.m_floats = fs
        self.assert_ok()

    def assert_ok(self):
        assert is_list_of_floats(self.m_floats)

    def add(self, f: float):
        self.m_floats.append(f)

    def range(self):
        for f in self.m_floats:
            yield f

    def sum(self) -> float:
        result = 0.0
        for f in self.range():
            result += f
        return result

    def len(self) -> int:
        return len(self.m_floats)

    def float(self, i):
        assert 0 <= i < self.len()
        return self.m_floats[i]

    def value_as_string(self, i: int) -> str:
        return str(self.float(i))

    def extremes(self) -> bas.Interval:
        assert self.len() > 0
        v0 = self.float(0)
        result = bas.interval_create(v0, v0)
        for i in range(1, self.len()):
            result.expand_to_include(self.float(i))
        return result

    def set(self, index: int, v: float):
        assert 0 <= index < self.len()
        self.m_floats[index] = v

    def loosely_equals(self, other) -> bool:
        n = self.len()
        if n != other.len():
            return False
        for i in range(0, n):
            if not bas.loosely_equals(self.float(i), other.float(i)):
                return False
        return True

    def increment(self, i: int, delta: float):
        self.set(i, self.float(i) + delta)

    def dot_product(self, other) -> float:  # other is type Floats
        assert isinstance(other, Floats)
        assert self.len() == other.len()
        result = 0.0
        for i in range(0, self.len()):
            result += self.float(i) * other.float(i)

        return result

    def minus(self, other):  # other is type Floats and result is type Floats
        assert isinstance(other, Floats)
        assert self.len() == other.len()
        result = floats_empty()
        for i in range(0, self.len()):
            result.add(self.float(i) - other.float(i))
        return result

    def pretty_string(self) -> str:
        ss = self.strings()
        assert isinstance(ss, Strings)
        return ss.concatenate_fancy('{', ',', '}')

    def strings(self):  # returns Strings
        result = strings_empty()
        for i in range(0, self.len()):
            result.add(self.value_as_string(i))
        return result

    def deep_copy(self):  # returns Floats
        result = floats_empty()
        for i in range(0, self.len()):
            result.add(self.float(i))
        return result

    def squared(self) -> float:
        return self.dot_product(self)

    def append(self, others):  # others are of type floats
        assert isinstance(others, Floats)
        for i in range(0, others.len()):
            self.add(others.float(i))


class Namer:
    def __init__(self):
        self.m_name_to_key = {}
        self.m_key_to_name = strings_empty()
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_name_to_key, dict)
        assert isinstance(self.m_key_to_name, Strings)
        self.m_key_to_name.assert_ok()

        assert len(self.m_name_to_key) == self.m_key_to_name.len()

        for k in range(0, self.len()):
            v = self.m_key_to_name.string(k)
            assert v in self.m_name_to_key
            assert self.m_name_to_key[v] == k

    def len(self) -> int:
        return self.m_key_to_name.len()

    def contains(self, s: str) -> bool:
        return s in self.m_name_to_key

    def add(self, name: str):
        assert not self.contains(name)
        key = self.len()
        self.m_name_to_key[name] = key
        self.m_key_to_name.add(name)
        if bas.is_power_of_two(self.len()):
            self.assert_ok()

    def key_from_name(self, name: str) -> tuple[int, bool]:
        if name in self.m_name_to_key:
            return self.m_name_to_key[name], True
        else:
            return -77, False

    def name_from_key(self, k: int) -> str:
        assert 0 <= k < self.len()
        return self.m_key_to_name.string(k)


def namer_empty() -> Namer:
    return Namer()


def is_list_of_string(y: list) -> bool:
    if not isinstance(y, list):
        return False

    for x in y:
        if not isinstance(x, str):
            return False

    return True


def is_list_of_list_of_string(x: list) -> bool:
    if not isinstance(x, list):
        return False

    for y in x:
        if not is_list_of_string(y):
            return False

    return True


def concatenate_list_of_strings(ls: list[str]) -> str:
    result = ""
    for s in ls:
        result = result + s
    return result


chars_per_line = 80


def is_list_of_ints(ins: list) -> bool:
    if not isinstance(ins, list):
        return False

    for i in ins:
        if not isinstance(i, int):
            return False

    return True


class Ints:
    def __init__(self, ins: list[int]):
        self.m_ints = ins
        self.assert_ok()

    def assert_ok(self):
        assert is_list_of_ints(self.m_ints)

    def int(self, i: int) -> int:
        assert 0 <= i < self.len()
        return self.m_ints[i]

    def len(self) -> int:
        return len(self.m_ints)

    def add(self, i: int):
        self.m_ints.append(i)

    def set(self, i: int, v: int):
        assert 0 <= i < self.len()
        self.m_ints[i] = v

    def increment(self, i: int):
        assert 0 <= i < self.len()
        self.m_ints[i] += 1

    def range(self):
        for i in self.m_ints:
            yield i

    def argmax(self):
        assert self.len() > 0
        result = 0
        highest_value = self.int(0)
        for i in range(1, self.len()):
            v = self.int(i)
            if v > highest_value:
                result = i
                highest_value = v

        assert self.int(result) == highest_value
        return result

    def argmin(self):
        assert self.len() > 0
        result = 0
        lowest_value = self.int(0)
        for i in range(1, self.len()):
            v = self.int(i)
            if v < lowest_value:
                result = i
                lowest_value = v

        assert self.int(result) == lowest_value
        return result

    def max(self):
        return self.int(self.argmax())

    def histogram(self):  # Returns Ints
        result = ints_empty()
        for v in self.range():
            assert v >= 0
            while result.len() <= v:
                result.add(0)
            result.increment(v)
        return result

    def min(self) -> int:
        return self.int(self.argmin())

    def invert_index(self):  # return Ints
        assert self.sort().equals(identity_ints(self.len()))
        if self.len() > 0:
            assert self.min() >= 0
        result = ints_empty()
        for i in range(0, self.len()):
            v = self.int(i)
            assert v >= 0
            while result.len() <= v:
                result.add(-777)
            assert v < result.len()
            assert result.int(v) < 0
            result.set(v, i)

        for v in range(0, result.len()):
            i = result.int(v)
            assert 0 <= i < self.len()
            assert self.int(i) == v

        assert isinstance(result, Ints)
        return result

    def sort(self):  # return Ints
        ios = self.indexes_of_sorted()
        return self.subset(ios)

    def indexes_of_sorted(self):  # return ints
        return indexes_of_sorted(self.m_ints)

    def subset(self, indexes):  # The 'indexes' argument has type Ints. returns Ints. result[j] = self[indexes[j]]
        result = ints_empty()
        for i in range(0, indexes.len()):
            index = indexes.int(i)
            result.add(self.int(index))
        assert isinstance(result, Ints)
        return result

    def equals(self, other) -> bool:  # other is type Ints
        assert isinstance(other, Ints)
        le = self.len()
        if other.len() != le:
            return False

        for i in range(0, le):
            if self.int(i) != other.int(i):
                return False

        return True

    def pretty_string(self) -> str:
        ss = self.strings()
        assert isinstance(ss, Strings)
        return ss.concatenate_fancy('{', ',', '}')

    def strings(self):  # returns object of type Strings
        result = strings_empty()
        for i in range(0, self.len()):
            result.add(bas.string_from_int(self.int(i)))
        assert isinstance(result, Strings)
        return result

    def contains_duplicates(self) -> bool:
        dic = {}
        for i in range(0, self.len()):
            v = self.int(i)
            if v in dic:
                return True
            else:
                dic[v] = True
        return False


def indexes_of_sorted(li: list) -> Ints:
    pairs = zip(range(0, len(li)), li)
    sorted_pairs = sorted(pairs, key=lambda p: p[1])
    result = ints_empty()
    for pair in sorted_pairs:
        result.add(pair[0])
    assert isinstance(result, Ints)
    assert result.len() == len(li)
    assert result.min() == 0
    assert result.max() == len(li) - 1
    assert not result.contains_duplicates()

    for r in range(0, len(li) - 1):
        i0 = result.int(r)
        i1 = result.int(r + 1)
        assert li[i0] <= li[i1]

    return result


def identity_ints(n: int) -> Ints:
    result = ints_empty()
    for i in range(0, n):
        result.add(i)
    return result


class Strings:
    def __init__(self, ss: list[str]):
        self.m_strings = ss
        self.assert_ok()

    def assert_ok(self):
        assert is_list_of_string(self.m_strings)

    def contains_duplicates(self) -> bool:
        nm = namer_empty()
        for i in range(0, self.len()):
            s = self.string(i)
            if nm.contains(s):
                return True
            nm.add(s)

        return False

    def len(self) -> int:
        return len(self.m_strings)

    def string(self, i: int) -> str:
        assert isinstance(i, int)
        assert 0 <= i < self.len()
        return self.m_strings[i]

    def add(self, s: str):
        self.m_strings.append(s)

    def pretty_string(self) -> str:
        return self.concatenate_fancy('{', ',', '}')

    def concatenate_fancy(self, left: str, mid: str, right: str) -> str:
        result = ""
        result += left
        for i in range(0, self.len()):
            if i > 0:
                result += mid
            result += self.string(i)
        result += right
        return result

    def pretty_strings(self) -> list[str]:
        start = " {"
        finish = " }"
        result = []
        partial = start

        for i in range(0, self.len()):
            s = self.string(i)
            if (len(start) + len(s) >= chars_per_line) or (len(partial) + 1 + len(s) < chars_per_line):
                print(f'i = {i}, s = {s}, partial = [{partial}]')
                partial = partial + " " + s
            else:
                result.append(partial + '\n')
                print(f'len(start) = {len(start)}')
                partial = bas.n_spaces(len(start)) + " " + s
                print(f'partial = [{partial}]')

        result.append(partial + finish + '\n')

        return result

    def concatenate_with_padding(self, col_to_n_chars: Ints) -> str:
        result = ""
        for i in range(0, self.len()):
            s = self.string(i)
            n_pads = col_to_n_chars.int(i) - len(s)
            assert n_pads >= 0
            if i < self.len() - 1:
                n_pads += 1
            result = result + s + bas.n_spaces(n_pads)
        return result

    def concatenate(self) -> str:
        result = ""
        for i in range(0, self.len()):
            result = result + self.string(i)
        return result

    def append(self, other):  # other is type Strings
        assert isinstance(other, Strings)
        for s in other.range():
            self.add(s)

    def indexes_of_sorted(self) -> Ints:
        return indexes_of_sorted(self.m_strings)

    def subset(self, indexes: Ints):  # returns Strings
        assert isinstance(indexes, Ints)
        indexes.assert_ok()
        result = strings_empty()
        for i in range(0, indexes.len()):
            index = indexes.int(i)
            result.add(self.string(index))
        return result

    def equals(self, other) -> bool:  # other is of type strings
        assert isinstance(other, Strings)
        le = self.len()
        if le != other.len():
            return False
        for i in range(0, le):
            if self.string(i) != other.string(i):
                return False
        return True

    def is_sorted(self) -> bool:
        if self.len() <= 1:
            return True

        for i in range(0, self.len() - 1):
            if self.string(i) > self.string(i + 1):
                return False

        return True

    def with_many(self, other):  # other type Strings, result type Strings
        assert isinstance(other, Strings)
        result = self.deep_copy()
        assert isinstance(result, Strings)
        result.append(other)
        return result

    def deep_copy(self):  # return type Strings
        result = strings_empty()
        for i in range(0, self.len()):
            result.add(self.string(i))
        return result

    def range(self):
        for s in self.m_strings:
            yield s


def strings_empty() -> Strings:
    return Strings([])


def ints_empty() -> Ints:
    return Ints([])


class StringsArray:
    def __init__(self, sss: list[Strings]):
        self.m_strings_array = sss
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_strings_array, list)
        for ss in self.m_strings_array:
            assert isinstance(ss, Strings)
            ss.assert_ok()

    def len(self):
        return len(self.m_strings_array)

    def strings(self, i: int) -> Strings:
        assert 0 <= i < self.len()
        return self.m_strings_array[i]

    def add(self, ss: Strings):
        self.m_strings_array.append(ss)

    def string(self, r, c):
        return self.strings(r).string(c)

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def pretty_strings(self) -> Strings:
        pad = "  "
        col_to_n_chars = self.col_to_num_chars()
        result = strings_empty()
        for r in range(0, self.len()):
            result.add(pad + self.strings(r).concatenate_with_padding(col_to_n_chars))
        return result

    def col_to_num_chars(self) -> Ints:
        result = ints_empty()
        for ss in self.range():
            for c in range(0, ss.len()):
                le = len(ss.string(c))
                assert c <= result.len()
                if c == result.len():
                    result.add(le)
                else:
                    result.set(c, max(le, result.int(c)))
        return result

    def range(self):
        for ss in self.m_strings_array:
            yield ss


def strings_array_empty() -> StringsArray:
    return StringsArray([])


def strings_from_split(s: str, separator: bas.Character) -> Strings:
    assert isinstance(s, str)
    result = strings_empty()
    partial = ""
    for i in range(0, len(s)):
        c = bas.character_from_string(s, i)
        if c.m_byte == separator.m_byte:
            result.add(partial)
            partial = ""
        else:
            partial += c.string()

    result.add(partial)

    return result


def strings_singleton(s: str) -> Strings:
    result = strings_empty()
    result.add(s)
    return result


def strings_from_lines_in_string(s: str) -> Strings:
    return strings_from_split(s, bas.character_create('\n'))


def bools_empty() -> Bools:
    return Bools([])


def bools_singleton(b: bool) -> Bools:
    result = bools_empty()
    result.add(b)
    return result


def bools_from_strings(ss: Strings) -> tuple[Bools, bool]:
    result = bools_empty()
    for i in range(0, ss.len()):
        b, ok = bas.bool_from_string(ss.string(i))
        if ok:
            result.add(b)
        else:
            return bools_empty(), False

    return result, True


def floats_empty():
    return Floats([])


def floats_from_strings(ss: Strings) -> tuple[Floats, bool]:
    result = floats_empty()
    for i in range(0, ss.len()):
        f, ok = bas.float_from_string(ss.string(i))
        if ok:
            result.add(f)
        else:
            return floats_empty(), False

    return result, True


def test_namer_ok(n: Namer, s: str, k: int):
    k2, ok = n.key_from_name(s)
    assert ok
    assert k2 == k
    assert s == n.name_from_key(k)


def test_namer_bad(n: Namer, s: str):
    k, ok = n.key_from_name(s)
    assert not ok


def unit_test_namer():
    n = namer_empty()
    n.add("a")
    n.add("b")
    test_namer_ok(n, "a", 0)
    test_namer_ok(n, "b", 1)
    test_namer_bad(n, "c")


def strings_create(ls: list[str]) -> Strings:
    return Strings(ls)


def test_floats_ok(ls: list[str], total: float):
    ss = strings_create(ls)
    fs, ok = floats_from_strings(ss)
    assert ok
    assert bas.loosely_equals(fs.sum(), total)


def test_floats_bad(ls: list[str]):
    ss = strings_create(ls)
    fs, ok = floats_from_strings(ss)
    assert not ok


def unit_test_index_sort():
    ss = strings_from_split("i would have liked you to have been deep frozen too", bas.character_space())
    ranks = ss.indexes_of_sorted()
    ss2 = strings_from_split("been deep frozen have have i liked to too would you", bas.character_space())
    ss3 = ss.subset(ranks)
    assert ss2.equals(ss3)


def unit_test():
    assert not bas.string_contains('hello_said_peter', bas.character_space())
    assert bas.string_contains('hello_said peter', bas.character_space())
    test_floats_ok(["4", "-0.5", ".0"], 3.5)
    test_floats_ok([], 0.0)
    test_floats_bad(["5", "5", "5p"])
    test_floats_bad(["5", "5 5", "5999"])
    unit_test_namer()
    unit_test_index_sort()


def strings_without_first_n_elements(ss: Strings, n: int) -> Strings:
    result = strings_empty()
    for i in range(n, ss.len()):
        s = ss.string(i)
        result.add(s)
    return result


def is_list_of_instances_of_strings_class(ssl: list) -> bool:
    if not isinstance(ssl, list):
        return False

    for ss in ssl:
        if not isinstance(ss, Strings):
            return False

    return True


def strings_default() -> Strings:
    return strings_empty()


class RowIndexedSmat:
    def __init__(self, first_row: Strings):
        self.m_row_to_col_to_string = strings_array_empty()
        self.m_row_to_col_to_string.add(first_row)
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_row_to_col_to_string, StringsArray)
        assert self.num_rows() > 0
        assert self.num_cols() > 0

    def num_cols(self) -> int:
        assert self.num_rows() > 0
        return self.m_row_to_col_to_string.strings(0).len()

    def column(self, c: int) -> Strings:
        assert 0 <= c < self.num_cols()
        result = strings_empty()
        for r in range(0, self.num_rows()):
            result.add(self.string(r, c))
        return result

    def num_rows(self) -> int:
        return self.m_row_to_col_to_string.len()

    def add(self, ss: Strings):
        assert self.num_cols() == ss.len()
        self.m_row_to_col_to_string.add(ss)

    def string(self, r: int, c: int) -> str:
        assert 0 <= r < self.num_rows()
        assert 0 <= c < self.num_cols()
        return self.m_row_to_col_to_string.strings(r).string(c)

    def strings_from_row(self, r: int) -> Strings:
        assert 0 <= r < self.num_rows()
        return self.m_row_to_col_to_string.strings(r)

    def pretty_strings(self) -> Strings:
        return self.m_row_to_col_to_string.pretty_strings()

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')


def row_indexed_smat_transpose(ris: RowIndexedSmat) -> RowIndexedSmat:
    assert ris.num_cols() > 0

    result = row_indexed_smat_single_row(ris.column(0))

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

    def column_old(self, c: int) -> Strings:
        result = strings_empty()
        for r in range(0, self.num_rows()):
            result.add(self.string(r, c))
        return result

    def column(self, c: int) -> Strings:
        result = self.m_col_to_row_to_string.strings_from_row(c)
        if bas.expensive_assertions:
            assert result.equals(self.column_old(c))
        return result

    def string(self, r: int, c: int) -> str:
        return self.row(r).string(c)

    def row(self, r: int) -> Strings:
        assert 0 <= r < self.num_rows()
        return self.m_row_to_col_to_string.strings_from_row(r)

    def pretty_string(self) -> str:
        return self.m_row_to_col_to_string.pretty_string()

    def pretty_strings(self) -> Strings:
        return self.strings_array().pretty_strings()

    def strings_array(self) -> StringsArray:
        result = strings_array_empty()
        for i in range(0, self.num_rows()):
            result.add(self.row(i))
        return result

    def extended_with_headings(self, top_left: str, left: Strings, top: Strings):  # returns Smat
        assert self.num_rows() == left.len()
        assert self.num_cols() == top.len()
        first_row = strings_singleton(top_left).with_many(top)
        result = row_indexed_smat_single_row(first_row)

        for r in range(0, self.num_rows()):
            row = strings_singleton(left.string(r)).with_many(self.row(r))
            result.add(row)

        return smat_from_row_indexed_smat(result)

    def pretty_strings_with_headings(self, top_left: str, left: Strings, top: Strings) -> Strings:
        extended = self.extended_with_headings(top_left, left, top)
        assert isinstance(extended, Smat)
        return extended.pretty_strings()


def smat_single_row(first_row: Strings) -> Smat:
    return smat_from_row_indexed_smat(row_indexed_smat_single_row(first_row))


class Fmat:
    def __init__(self):
        self.m_row_to_col_to_value = []
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_row_to_col_to_value, list)
        if self.num_rows() > 0:
            n_cols = self.row(0).len()
            for fs in self.m_row_to_col_to_value:
                assert isinstance(fs, Floats)
                assert fs.len() == n_cols
                fs.assert_ok()

    def num_rows(self) -> int:
        return len(self.m_row_to_col_to_value)

    def row(self, r: int) -> Floats:
        assert 0 <= r < self.num_rows()
        return self.m_row_to_col_to_value[r]

    def num_cols(self) -> int:
        assert self.num_rows() > 0
        return self.row(0).len()

    def add_row(self, z: Floats):
        if self.num_rows() > 0:
            assert z.len() == self.num_cols()
        self.m_row_to_col_to_value.append(z)

    def column(self, c: int) -> Floats:
        result = floats_empty()
        for r in range(0, self.num_rows()):
            result.add(self.float(r, c))
        return result

    def float(self, r: int, c: int) -> float:
        return self.row(r).float(c)

    def loosely_equals(self, other) -> bool:
        assert isinstance(other, Fmat)
        n_rows = self.num_rows()
        if n_rows != other.num_rows():
            return False
        for i in range(0, n_rows):
            if not self.row(i).loosely_equals(other.row(i)):
                return False

        return True

    def times(self, x: Floats) -> Floats:
        assert self.num_cols() == x.len()
        result = floats_empty()
        for i in range(0, self.num_rows()):
            result.add(self.row(i).dot_product(x))
        return result

    def pretty_strings(self) -> Strings:
        return self.smat().pretty_strings()

    def row_indexed_smat(self) -> RowIndexedSmat:
        assert self.num_rows() > 0
        result = row_indexed_smat_single_row(self.row(0).strings())
        for i in range(1, self.num_rows()):
            result.add(self.row(i).strings())
        return result

    def smat(self) -> Smat:
        return smat_from_row_indexed_smat(self.row_indexed_smat())

    def pretty_strings_with_headings(self, top_left: str, left: Strings, top: Strings):
        return self.smat().pretty_strings_with_headings(top_left, left, top)


def fmat_empty():
    return Fmat()


def floats_all_constant(n, c: float) -> Floats:
    result = floats_empty()
    for i in range(0, n):
        result.add(c)
    return result


def floats_all_zero(n: int) -> Floats:
    return floats_all_constant(n, 0.0)


def floats_singleton(f: float) -> Floats:
    result = floats_empty()
    result.add(f)
    return result


def row_indexed_smat_single_row(first_row: Strings) -> RowIndexedSmat:
    return RowIndexedSmat(first_row)


def row_indexed_smat_unit(s: str) -> RowIndexedSmat:
    return row_indexed_smat_single_row(strings_singleton(s))


def row_indexed_smat_default():
    return row_indexed_smat_unit('default')


def row_indexed_smat_from_strings_array(ssa: StringsArray) -> tuple[RowIndexedSmat, bas.Errmess]:
    if not ssa:
        return row_indexed_smat_default(), bas.errmess_error(
            "Can't make a row indexed smat from an empty array of strings")

    result = row_indexed_smat_single_row(ssa.strings(0))
    for r in range(1, ssa.len()):
        ss = ssa.strings(r)
        if ss.len() != result.num_cols():
            return result, bas.errmess_error(
                f'first row of csv file has {result.num_cols()} items, but row {r} has {ss.len()} items')
        result.add(ss)

    return result, bas.errmess_ok()


def smat_from_row_indexed_smat(rsm: RowIndexedSmat) -> Smat:
    return Smat(rsm)


def smat_unit(s: str) -> Smat:
    return smat_from_row_indexed_smat(row_indexed_smat_unit(s))


def smat_default():
    return smat_unit("default")
