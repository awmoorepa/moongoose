import ambasic


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
        return ambasic.string_from_bool(self.bool(r))

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

    def sum(self) -> float:
        result = 0.0
        for i in range(0, self.len()):
            result += self.float(i)
        return result

    def len(self) -> int:
        return len(self.m_floats)

    def float(self, i):
        assert 0 <= i < self.len()
        return self.m_floats[i]

    def value_as_string(self, i: int) -> str:
        return str(self.float(i))


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
        if ambasic.is_power_of_two(self.len()):
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

    def string(self, i) -> str:
        assert 0 <= i < self.len()
        return self.m_strings[i]

    def add(self, s: str):
        self.m_strings.append(s)

    def pretty_string(self) -> str:
        return concatenate_list_of_strings(self.pretty_strings())

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
                partial = ambasic.n_spaces(len(start)) + " " + s
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
            result = result + s + ambasic.n_spaces(n_pads)
        return result

    def concatenate(self) -> str:
        result = ""
        for i in range(0, self.len()):
            result = result + self.string(i)
        return result


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
        return self.pretty_strings().concatenate()

    def pretty_strings(self) -> Strings:
        pad = "  "
        col_to_n_chars = self.col_to_num_chars()
        result = strings_empty()
        for r in range(0, self.len()):
            result.add(pad + self.strings(r).concatenate_with_padding(col_to_n_chars) + '\n')
        return result

    def col_to_num_chars(self) -> Ints:
        result = ints_empty()
        for r in range(0, self.len()):
            ss = self.strings(r)
            for c in range(0, ss.len()):
                le = len(ss.string(c))
                assert c <= result.len()
                if c == result.len():
                    result.add(le)
                else:
                    result.set(c, max(le, result.int(c)))
        return result


def strings_array_empty() -> StringsArray:
    return StringsArray([])


def strings_from_split(s: str, separator: ambasic.Character) -> Strings:
    assert isinstance(s, str)
    result = strings_empty()
    partial = ""
    for i in range(0, len(s)):
        c = ambasic.character_from_string(s, i)
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
    return strings_from_split(s, ambasic.character_create('\n'))


def bools_empty() -> Bools:
    return Bools([])


def bools_singleton(b: bool) -> Bools:
    result = bools_empty()
    result.add(b)
    return result


def bools_from_strings(ss: Strings) -> tuple[Bools, bool]:
    result = bools_empty()
    for i in range(0, ss.len()):
        b, ok = ambasic.bool_from_string(ss.string(i))
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
        f, ok = ambasic.float_from_string(ss.string(i))
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
    assert ambasic.loosely_equals(fs.sum(), total)


def test_floats_bad(ls: list[str]):
    ss = strings_create(ls)
    fs, ok = floats_from_strings(ss)
    assert not ok


def unit_test():
    test_floats_ok(["4", "-0.5", ".0"], 3.5)
    test_floats_ok([], 0.0)
    test_floats_bad(["5", "5", "5p"])
    test_floats_bad(["5", "5 5", "5999"])
    unit_test_namer()


def strings_without_first_n_elements(ss: Strings, n: int) -> Strings:
    result = strings_empty()
    for i in range(n, ss.len()):
        result.add(ss.string(i))
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
