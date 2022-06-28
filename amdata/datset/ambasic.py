from __future__ import annotations
import math
import random
from typing import Tuple, List, Iterator


def is_power_of_two(n: int) -> bool:
    assert n >= 0
    while n > 1:
        if (n & 1) == 1:
            return False
        n = n >> 1

    return True


def my_error(s: str):
    print(f'*** AWM code signals error: {s}')
    assert False


class Errmess:
    def __init__(self, is_ok: bool, s: str):
        self.m_is_ok: bool = is_ok
        self.m_string: str = s
        self.assert_ok()

    def is_error(self):
        return not self.m_is_ok

    def assert_ok(self):
        assert isinstance(self.m_is_ok, bool)
        assert isinstance(self.m_string, str)
        assert self.m_is_ok == (self.m_string == "")

    def is_ok(self):
        return self.m_is_ok

    def string(self) -> str:
        if self.is_ok():
            return 'ok'
        else:
            return self.m_string


def errmess_ok() -> Errmess:
    return Errmess(True, "")


def errmess_error(s: str) -> Errmess:
    return Errmess(False, s)


class Character:
    def __init__(self, single_character_string: str):
        if len(single_character_string) != 1:
            print(f'This string should have one character: [{single_character_string}]')
        assert len(single_character_string) == 1
        self.m_byte = ord(single_character_string)
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_byte, int)
        assert 0 <= self.m_byte < 0x7F

    def string(self) -> str:
        return chr(self.m_byte)

    def equals_string(self, single_character_string: str) -> bool:
        assert len(single_character_string) == 1
        return self.m_byte == ord(single_character_string)

    def equals(self, c) -> bool:
        assert isinstance(c, Character)
        return self.m_byte == c.m_byte


def character_create(single_character_string: str) -> Character:
    return Character(single_character_string)


def character_from_string(s: str, i: int) -> Character:
    assert isinstance(s, str)
    assert 0 <= i < len(s)
    return Character(s[i])


def float_from_string(s: str) -> Tuple[float, bool]:
    try:
        x = float(s)
        return x, True
    except ValueError:
        return -7.77e77, False


def string_denotes_true(s: str) -> bool:
    if len(s) == 0:
        return False

    c = s[0]
    if c == 't' or c == 'T':
        lo = s.lower()
        return lo == 't' or lo == "true"
    else:
        return False


def string_denotes_false(s: str) -> bool:
    if len(s) == 0:
        return False

    c = s[0]
    if c == 'f' or c == 'F':
        lo = s.lower()
        return lo == 'f' or lo == "false"
    else:
        return False


def bool_from_string(s: str) -> Tuple[bool, bool]:
    if string_denotes_true(s):
        return True, True
    elif string_denotes_false(s):
        return False, True
    else:
        return False, False


def character_space() -> Character:
    return character_create(' ')


def character_newline():
    return character_create('\n')


def loosely_equals(x: float, y: float) -> bool:
    scale = max(abs(x), abs(y), 1e-3)
    diff = abs(x - y) / scale
    return diff < 1e-5


def string_is_float(s: str) -> bool:
    f, ok = float_from_string(s)
    return ok


def string_is_bool(s: str) -> bool:
    b, ok = bool_from_string(s)
    return ok


def n_spaces(n: int) -> str:
    result = ''
    for i in range(n):
        result += ' '

    assert len(result) == n

    return result


def string_from_bool(b: bool) -> str:
    if b:
        return "true"
    else:
        return "false"


def string_from_bytes(bys: bytes) -> str:
    return bys.decode("utf-8")


def character_default() -> Character:
    return character_space()


def tidiest_with_both_between_1_and_10(a: float, b: float) -> float:
    assert 1.0 <= a
    assert a < b
    assert b < 10.0

    if a < 5.0 <= b:
        return 5.0
    else:
        largest_even_not_greater_than_b = 2.0 * math.floor(b / 2.0)
        if a < largest_even_not_greater_than_b:
            return largest_even_not_greater_than_b
        else:
            largest_int_not_greater_than_b = math.floor(b)
            if a < largest_int_not_greater_than_b:
                return largest_int_not_greater_than_b
            else:
                f = math.floor(a)
                assert f >= 1.0
                assert b < (f + 1)
                return f + 0.1 * tidiest_with_both_positive(10 * (a - f), 10 * (
                        b - f))  # Could do this with a loop, but this is much easier to read


def tidiest_with_both_positive(a: float, b: float) -> float:
    assert 0 < a < b

    n = math.floor(math.log10(a))
    ten_to_n = math.pow(10.0, n)

    if b >= 10 * ten_to_n:
        return 10 * ten_to_n
    else:
        return ten_to_n * tidiest_with_both_between_1_and_10(a / ten_to_n, min(10.0 - 1e-8, b / ten_to_n))


def tidiest(a: float, b: float) -> float:
    assert a < b

    if a <= 0 <= b:
        return 0.0
    elif a > 0:
        return tidiest_with_both_positive(a, b)
    else:
        assert b < 0
        return -tidiest_with_both_positive(-b, -a)


def test_tidiest():
    assert loosely_equals(1.001004, tidiest(1.001003, 1.001005))
    assert loosely_equals(2.0, tidiest(1.01, 3.0))
    assert loosely_equals(2.0, tidiest(1.01, 3.999))
    assert loosely_equals(200.0, tidiest(103.44, 397.4447))
    assert loosely_equals(100.0, tidiest(99.10344, 397.4447))
    assert loosely_equals(397.4446, tidiest(397.44455, 397.44478))
    assert loosely_equals(-200.0, tidiest(-397.4447, -103.44))
    assert loosely_equals(0.0, tidiest(-397.4447, 103.44))


def loosely_lte(x: float, y: float) -> bool:
    return x <= y or loosely_equals(x, y)


class Interval:
    def __init__(self, lo: float, hi: float):
        self.m_lo = lo
        self.m_hi = hi
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_lo, float)
        assert isinstance(self.m_hi, float)
        assert self.m_lo <= self.m_hi

    def expand_to_include(self, x: float):
        if x < self.lo():
            self.set_lo(x)
        elif x > self.hi():
            self.set_hi(x)

    def lo(self) -> float:
        return self.m_lo

    def set_lo(self, x: float):
        assert x <= self.hi()
        self.m_lo = x

    def hi(self) -> float:
        return self.m_hi

    def set_hi(self, x: float):
        assert self.lo() <= x
        self.m_hi = x

    def fractional_from_absolute(self, f: float) -> float:
        return (f - self.lo()) / self.width()

    def width(self) -> float:
        return self.hi() - self.lo()

    def tidiest_member(self) -> float:
        return tidiest(self.lo(), self.hi())

    def tidy_surrounder(self):
        my_width = self.width()
        lo_cap_width = 1 if loosely_equals(my_width, 0.0) else (0.25 * my_width)
        result_lo = tidiest(self.lo() - lo_cap_width, self.lo())
        min_result_width = lo_cap_width if loosely_equals(self.hi(), result_lo) else (self.hi() - result_lo)

        assert min_result_width > 0.0
        max_result_width = min_result_width * (1 + lo_cap_width)
        result_width = tidiest(min_result_width, max_result_width)
        result = interval_create(result_lo, result_lo + result_width)

        assert result.width() > 0.0
        assert result.loosely_fully_contains_interval(self)

        return result

    def loosely_fully_contains_interval(self, other) -> bool:  # other is type Interval
        assert isinstance(other, Interval)
        return loosely_lte(self.lo(), other.lo()) and loosely_lte(other.hi(), self.hi())

    def extremes(self) -> Tuple[float, float]:
        return self.lo(), self.hi()

    def deep_copy(self) -> Interval:
        return interval_create(self.lo(), self.hi())


def interval_create(lo: float, hi: float) -> Interval:
    return Interval(lo, hi)


class Maybeint:
    def __init__(self, is_defined: bool, value: int):
        self.m_is_defined = is_defined
        self.m_value = value
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_is_defined, bool)
        assert isinstance(self.m_value, int)
        if not self.m_is_defined:
            assert self.m_value < 0

    def int(self) -> Tuple[int, bool]:
        if self.m_is_defined:
            return self.m_value, True
        else:
            return -777, False

    def is_undefined(self) -> bool:
        return not self.is_defined()

    def is_defined(self) -> bool:
        return self.m_is_defined


def maybeint_defined(value: int) -> Maybeint:
    return Maybeint(True, value)


def maybeint_undefined() -> Maybeint:
    return Maybeint(False, -7777)


def string_index_of_character(s: str, target: Character) -> Tuple[int, bool]:
    assert isinstance(target, Character)
    target_as_string = target.string()
    for i, ci_as_string in enumerate(s):
        if ci_as_string == target_as_string:
            return i, True
    return -77, False


def string_contains(s: str, c: Character) -> bool:
    index, ok = string_index_of_character(s, c)
    return ok


def string_replace(s: str, old: Character, nu: Character) -> str:
    result = ""
    old_as_string = old.string()
    nu_as_string = nu.string()

    for ci_as_string in s:
        if ci_as_string == old_as_string:
            result = result + nu_as_string
        else:
            result = result + ci_as_string
    return result


log_root_two_pi = 0.5 * math.log(2 * math.pi)


def string_from_float(x: float) -> str:
    return f'{x}'


def string_from_int(x: int) -> str:
    return f'{x}'


def unit_test():
    test_tidiest()


expensive_assertions = True


def interval_unit() -> Interval:
    return interval_create(0.0, 1.0)


def int_random(n: int) -> int:
    assert isinstance(n, int)
    assert n > 0
    return random.randrange(n)


class Intervals:
    def __init__(self, li: List[Interval]):
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for iv in self.m_list:
            assert isinstance(iv, Interval)
            iv.assert_ok()

    def range(self) -> Iterator[Interval]:
        for iv in self.m_list:
            yield iv

    def add(self, iv: Interval):
        self.m_list.append(iv)

    def interval(self, i: int) -> Interval:
        assert 0 <= i < self.len()
        return self.m_list[i]

    def len(self) -> int:
        return len(self.m_list)

    def append(self, other: Intervals):
        for iv in other.range():
            self.add(iv)


def intervals_empty() -> Intervals:
    return Intervals([])


def intervals_all_constant(n: int, iv: Interval) -> Intervals:
    result = intervals_empty()
    for i in range(n):
        result.add(iv.deep_copy())
    return result


def intervals_all_unit(n: int) -> Intervals:
    return intervals_all_constant(n, interval_unit())


def intervals_singleton(iv: Interval) -> Intervals:
    result = intervals_empty()
    result.add(iv)
    return result
