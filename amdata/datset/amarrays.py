from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Tuple, List, Iterator

import numpy as np

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
    def __init__(self, bs: List[bool]):
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

    def range(self):
        for b in self.m_bools:
            yield b

    def subset(self, indexes: Ints) -> Bools:
        result = bools_empty()
        for i in indexes.range():
            result.add(self.bool(i))
        return result


def floats_numpy_from_ndarray(nda: np.ndarray) -> FloatsNumpy:
    return FloatsNumpy(len(nda), nda)


def ndarray_from_list_of_floats(li: List[float]) -> np.ndarray:
    assert isinstance(li, list)
    for x in li:
        assert isinstance(x, float)

    nda = np.empty(len(li), dtype=float)
    for i, x in enumerate(li):
        nda[i] = x

    return nda


def floats_numpy_from_list2(li: List[float]) -> FloatsNumpy:
    nda = ndarray_from_list_of_floats(li)
    return floats_numpy_from_ndarray(nda)


def floats_numpy_empty(capacity: int) -> FloatsNumpy:
    return FloatsNumpy(capacity, np.empty(0))


def ints_from_ndarray(nda: np.ndarray) -> Ints:
    result = ints_empty()
    for i in np.nditer(nda):
        print(f'i = {i}')
        print(f'type(i) = {type(i)}')
        print(f'i.shape() = {np.shape(i)}')
        print(f'i.ndim() = {np.ndim(i)}')
        j = int(i)
        assert isinstance(j, int)
        result.add(j)
    return result


class Floats(ABC):
    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def is_filled(self) -> bool:
        pass

    @abstractmethod
    def range2(self) -> Iterator[float]:
        pass

    @abstractmethod
    def sum(self) -> float:
        pass

    @abstractmethod
    def len(self) -> int:
        pass

    def extremes(self) -> bas.Interval:
        assert self.is_filled()
        return bas.interval_create(self.min(), self.max())

    @abstractmethod
    def set2(self, index: int, v: float):
        pass

    def is_loosely_zero(self) -> bool:
        epsilon = 1e-5
        return self.magnitude() < epsilon

    def loosely_equals(self, other: Floats) -> bool:
        return self.minus(other).magnitude() < max(self.magnitude(), other.magnitude(), 1.0) * 1e-5

    def increment2(self, i: int, delta: float):
        self.set2(i, self.float2(i) + delta)

    @abstractmethod
    def dot_product(self, other: Floats) -> float:
        pass

    @abstractmethod
    def minus(self, other: Floats) -> Floats:
        pass

    def pretty_string(self) -> str:
        ss = self.strings()
        assert isinstance(ss, Strings)
        return ss.concatenate_fancy('{', ',', '}')

    def strings(self) -> Strings:
        result = strings_empty()
        for x in self.range2():
            result.add(bas.string_from_float(x))
        assert isinstance(result, Strings)
        return result

    def deep_copy(self) -> Floats:
        pass

    def squared(self) -> float:
        return self.dot_product(self)

    def tidy_surrounder(self) -> bas.Interval:
        return self.extremes().tidy_surrounder()

    def min(self) -> float:
        return self.float2(self.argmin())

    @abstractmethod
    def argmin(self) -> int:
        pass

    def max(self) -> float:
        return self.float2(self.argmax())

    @abstractmethod
    def argmax(self) -> int:
        pass

    def tidy_extremes(self) -> Tuple[float, float]:
        return self.tidy_surrounder().extremes()

    @abstractmethod
    def plus(self, other: Floats) -> Floats:
        pass

    def subset(self, indexes: Ints) -> Floats:
        assert self.is_filled()
        result = floats_empty(indexes.len())
        for i in indexes.range():
            result.add(self.float2(i))
        assert result.is_filled()
        return result

    @abstractmethod
    def indexes_of_sorted(self) -> Ints:
        pass

    def without_leftmost_element(self) -> Floats:
        return self.without_n_leftmost_elements(1)

    def without_n_leftmost_elements(self, n: int) -> Floats:
        assert n <= self.len()
        result = floats_empty(self.len() - n)
        for i in range(n, self.len()):
            result.add(self.float2(i))
        return result

    @abstractmethod
    def median(self) -> float:
        pass

    def sorted(self) -> Floats:
        return self.subset(self.indexes_of_sorted())

    @abstractmethod
    def times_scalar(self, scale: float) -> Floats:
        pass

    @abstractmethod
    def map_product_with(self, other: Floats) -> Floats:
        pass

    def is_loosely_constant(self) -> bool:
        if self.len() < 2:
            return True
        return self.minus_scalar(self.float2(0)).magnitude() < max(self.magnitude(), 1.0) * 1e-5

    def distance_to(self, other: Floats) -> float:
        return math.sqrt(self.distance_squared_to(other))

    def distance_squared_to(self, other: Floats) -> float:
        return self.minus(other).sum_squares()

    def mean(self) -> float:
        assert self.len() > 0
        return self.sum() / self.len()

    @abstractmethod
    def float2(self, i: int) -> float:
        pass

    def magnitude(self) -> float:
        return math.sqrt(self.squared())

    def minus_scalar(self, x: float) -> Floats:
        return self.plus_scalar(-x)

    @abstractmethod
    def plus_scalar(self, x: float) -> Floats:
        pass

    def sum_squares(self) -> float:
        return self.dot_product(self)

    def appended_with(self, other: Floats) -> Floats:
        assert self.is_filled()
        assert other.is_filled()
        result = floats_empty(self.len() + other.len())
        for x in self.range2():
            result.add(x)
        for x in other.range2():
            result.add(x)
        return result

    @abstractmethod
    def add(self, f: float):
        pass

    def list(self) -> List[float]:
        assert self.is_filled()
        result = []
        for x in self.range2():
            result.append(x)
        return result


class FloatsNumpy(Floats):
    def __init__(self, capacity: int, nda: np.ndarray):
        n_filled = len(nda)
        assert n_filled <= capacity
        self.m_num_filled = n_filled
        self.m_capacity = capacity

        if n_filled == capacity:
            self.m_ndarray = nda
        else:
            self.m_ndarray = np.empty(shape=capacity, dtype=float)
            for i, x in enumerate(nda):
                assert isinstance(x, float)
                self.m_ndarray[i] = x
            for i in range(n_filled, capacity):
                self.m_ndarray[i] = -8e88
        self.assert_ok()

    def assert_ok(self):
        assert use_numpy
        assert isinstance(self.m_num_filled, int)
        assert isinstance(self.m_capacity, int)
        assert 0 <= self.m_num_filled <= self.m_capacity
        assert isinstance(self.m_ndarray, np.ndarray)
        if self.m_capacity > 0:
            for i, x in enumerate(np.nditer(self.m_ndarray)):
                # print(f'i = {i}')
                # print(f'x.type() = {type(x)}')
                # print(f'x.shape() = {np.shape(x)}')
                # print(f'x.isscalar() = {np.isscalar(x)}')
                # print(f'x.ndim() = {np.ndim(x)}')
                # print(f'float(x) = {float(x)}')
                # print(f'isinstance(x,numbers.Number) = {isinstance(x, numbers.Number)}')
                assert np.ndim(x) == 0
                if i < self.m_num_filled:
                    assert float(x) > -7e88
                else:
                    assert float(x) < -7e88

    def is_filled(self) -> bool:
        return self.m_num_filled == self.m_capacity

    def range2(self) -> Iterator[float]:
        assert self.is_filled()
        for x in np.nditer(self.m_ndarray):
            yield float(x)

    def sum(self) -> float:
        assert self.is_filled()
        return self.m_ndarray.sum()

    def len(self) -> int:
        assert self.is_filled()
        return self.m_capacity

    def set2(self, index: int, v: float):
        assert 0 <= index < self.len()
        self.m_ndarray[index] = v

    def increment2(self, i: int, delta: float):
        self.set2(i, self.float2(i) + delta)

    def dot_product(self, other: FloatsNumpy) -> float:
        assert self.len() == other.len()
        assert self.is_filled()
        assert other.is_filled()
        return float(np.dot(self.m_ndarray, other.m_ndarray))

    def minus(self, other: FloatsNumpy) -> Floats:
        assert isinstance(other, Floats)
        assert self.len() == other.len()
        assert self.is_filled()
        assert other.is_filled()
        return floats_numpy_from_ndarray(np.subtract(self.m_ndarray, other.m_ndarray))

    def deep_copy(self) -> Floats:
        assert self.is_filled()
        return floats_numpy_from_ndarray(np.copy(self.m_ndarray))

    def argmin(self) -> int:
        assert self.is_filled()
        return int(np.argmin(self.m_ndarray))

    def argmax(self) -> int:
        return int(np.argmax(self.m_ndarray))

    def plus(self, other: Floats) -> Floats:
        assert self.is_filled()
        assert isinstance(other, FloatsNumpy)
        return floats_numpy_from_ndarray(np.add(self.m_ndarray, other.m_ndarray))

    def indexes_of_sorted(self) -> Ints:
        assert self.is_filled()
        return ints_from_ndarray(np.argsort(self.m_ndarray))

    def median(self) -> float:
        assert self.is_filled()
        return float(np.median(self.m_ndarray))

    def times_scalar(self, scale: float) -> Floats:
        assert self.is_filled()
        return floats_numpy_from_ndarray(self.m_ndarray * scale)

    def map_product_with(self, other: FloatsNumpy) -> Floats:
        assert self.is_filled()
        assert other.is_filled()
        return floats_numpy_from_ndarray(np.multiply(self.m_ndarray, other.m_ndarray))

    def float2(self, i: int) -> float:
        return self.m_ndarray[i]

    def plus_scalar(self, x: float) -> Floats:
        assert self.is_filled()
        result = np.copy(self.m_ndarray)
        result[:] += x
        return floats_numpy_from_ndarray(result)

    def add(self, f: float):
        assert isinstance(f, float)
        assert self.m_num_filled < self.m_capacity
        self.m_ndarray[self.m_num_filled] = f
        self.m_num_filled += 1


def floats_list_empty(capacity: int) -> FloatsList:
    return FloatsList(capacity, [])


def floats_list_from_list(li: List[float]) -> FloatsList:
    return FloatsList(len(li), li)


def floats_list_all_constant(n_elements: int, value: float) -> FloatsList:
    result = floats_list_empty(n_elements)
    for i in range(n_elements):
        result.add(value)
    return result


class FloatsList(Floats):
    def __init__(self, capacity: int, li: List[float]):
        assert len(li) <= capacity
        self.m_capacity = capacity
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert not use_numpy
        assert isinstance(self.m_capacity, int)
        assert 0 <= self.num_filled() <= self.m_capacity
        assert isinstance(self.m_list, list)
        for f in self.m_list:
            assert isinstance(f, float)

    def is_filled(self) -> bool:
        return self.num_filled() == self.m_capacity

    def range2(self) -> Iterator[float]:
        assert self.is_filled()
        for x in self.m_list:
            yield x

    def sum(self) -> float:
        assert self.is_filled()
        result = 0.0
        for f in self.range2():
            result += f
        return result

    def len(self) -> int:
        assert self.is_filled()
        return self.m_capacity

    def set2(self, index: int, v: float):
        assert 0 <= index < self.len()
        self.m_list[index] = v

    def increment2(self, i: int, delta: float):
        self.set2(i, self.float2(i) + delta)

    def dot_product(self, other: Floats) -> float:
        assert self.len() == other.len()
        assert self.is_filled()
        assert other.is_filled()
        result = 0.0
        for me, ot in zip(self.range2(), other.range2()):
            result += me * ot
        return result

    def minus(self, other: Floats) -> Floats:
        assert isinstance(other, FloatsList)
        assert self.len() == other.len()
        result = floats_list_empty(self.len())
        for me, ot in zip(self.range2(), other.range2()):
            result.add(me - ot)
        assert isinstance(result, Floats)
        return result

    def pretty_string(self) -> str:
        ss = self.strings()
        assert isinstance(ss, Strings)
        return ss.concatenate_fancy('{', ',', '}')

    def strings(self) -> Strings:
        result = strings_empty()
        for x in self.range2():
            result.add(bas.string_from_float(x))
        assert isinstance(result, Strings)
        return result

    def deep_copy(self) -> Floats:
        assert self.is_filled()
        result = floats_list_empty(self.len())
        for f in self.range2():
            result.add(f)
        assert isinstance(result, Floats)
        return result

    def tidy_surrounder(self) -> bas.Interval:
        return self.extremes().tidy_surrounder()

    def min(self) -> float:
        assert self.is_filled()
        return np.min(self.m_list)

    def argmin(self) -> int:
        assert self.is_filled()
        return int(np.argmin(self.m_list))

    def max(self) -> float:
        assert self.is_filled()
        return np.max(self.m_list)

    def argmax(self) -> int:
        return int(np.argmax(self.m_list))

    def plus(self, other: Floats) -> Floats:
        assert isinstance(other, FloatsList)
        assert self.is_filled()
        le = self.len()
        assert le == other.len()
        result = floats_list_empty(le)
        for me, ot in zip(self.range2(), other.range2()):
            result.add(me + ot)
        assert isinstance(result, Floats)
        return result

    def indexes_of_sorted(self) -> Ints:
        assert self.is_filled()
        return indexes_of_sorted(self.m_list)

    def median_helper(self) -> float:
        le = self.len()
        assert le > 0
        half_length = le // 2  # integer division, rounds down
        assert isinstance(half_length, int)
        if le % 2 == 0:
            v0, v1 = self.kth_smallest_two_elements(half_length-1)
            return (v0 + v1) / 2
        else:
            return self.kth_smallest_element(half_length)

    def median_slow(self) -> float:
        s = self.sorted()
        h = self.len() // 2
        if s.len() % 2 == 0:
            return (s.float2(h-1) + s.float2(h)) / 2
        else:
            return s.float2(h)

    def median(self) -> float:
        result = self.median_helper()
        print(f' remove this slow test')
        assert bas.loosely_equals(result, self.median_slow())
        return result

    def kth_smallest_two_elements(self, k: int) -> Tuple[float, float]:
        assert 0 <= k < self.len() - 1

        lower, pivot, higher = self.split_with_random_pivot()

        if k + 1 < lower.len():
            return lower.kth_smallest_two_elements(k)
        elif k + 1 == lower.len():
            return lower.max(), pivot
        elif k == lower.len():
            return pivot, higher.min()
        else:
            return higher.kth_smallest_two_elements(k - lower.len() - 1)

    def kth_smallest_element(self, k: int) -> float:
        assert 0 <= k < self.len()

        lower, pivot, higher = self.split_with_random_pivot()

        if k < lower.len():
            return lower.kth_smallest_element(k)
        elif k == lower.len():
            return pivot
        else:
            return higher.kth_smallest_element(k - lower.len() - 1)

    def split_with_random_pivot(self) -> Tuple[FloatsList, float, FloatsList]:
        pivot_index = bas.int_random(self.len())
        pivot = self.float2(pivot_index)
        lower = []
        higher = []

        for i, f in enumerate(self.range2()):
            if i != pivot_index:
                if f < pivot:
                    lower.append(f)
                else:
                    higher.append(f)

        return floats_list_from_list(lower), pivot, floats_list_from_list(higher)

    def times_scalar(self, scale: float) -> Floats:
        assert self.is_filled()
        result = floats_list_empty(self.len())
        for f in self.range2():
            result.add(f * scale)
        assert isinstance(result, Floats)
        return result

    def map_product_with(self, other: Floats) -> Floats:
        assert isinstance(other, FloatsList)
        assert self.is_filled()
        assert other.is_filled()
        le = self.len()
        result = floats_list_empty(le)
        assert le == other.len()
        for me, ot in zip(self.range2(), other.range2()):
            result.add(me * ot)
        assert isinstance(result, Floats)
        return result

    def is_loosely_constant(self) -> bool:
        if self.len() < 2:
            return True
        return self.minus_scalar(self.float2(0)).magnitude() < max(self.magnitude(), 1.0) * 1e-5

    def distance_to(self, other: Floats) -> float:
        return math.sqrt(self.distance_squared_to(other))

    def distance_squared_to(self, other: Floats) -> float:
        return self.minus(other).sum_squares()

    def mean(self) -> float:
        assert self.len() > 0
        return self.sum() / self.len()

    def float2(self, i: int) -> float:
        return self.m_list[i]

    def plus_scalar(self, x: float) -> Floats:
        assert self.is_filled()
        result = floats_list_empty(self.len())
        for f in self.range2():
            result.add(f + x)
        assert isinstance(result, Floats)
        return result

    def add(self, f: float):
        assert isinstance(f, float)
        assert self.num_filled() < self.m_capacity
        self.m_list.append(f)

    def num_filled(self) -> int:
        return len(self.m_list)


# class Floats:
#     def __init__(self, fs: List[float]):
#         self.m_floats = fs
#         self.assert_ok()
#
#     def assert_ok(self):
#         assert is_list_of_floats(self.m_floats)
#
#     def add(self, f: float):
#         self.m_floats.append(f)
#
#     def range(self):
#         for f in self.m_floats:
#             yield f
#
#     def sum(self) -> float:
#         result = 0.0
#         for f in self.range():
#             result += f
#         return result
#
#     def len(self) -> int:
#         return len(self.m_floats)
#
#     def float(self, i):
#         assert 0 <= i < self.len()
#         return self.m_floats[i]
#
#     def value_as_string(self, i: int) -> str:
#         return str(self.float(i))
#
#     def extremes(self) -> bas.Interval:
#         assert self.len() > 0
#         v0 = self.float(0)
#         result = bas.interval_create(v0, v0)
#         for i in range(1, self.len()):
#             result.expand_to_include(self.float(i))
#         return result
#
#     def set(self, index: int, v: float):
#         assert 0 <= index < self.len()
#         self.m_floats[index] = v
#
#     def loosely_equals(self, other) -> bool:
#         assert isinstance(other, Floats)
#         n = self.len()
#         if n != other.len():
#             return False
#         for a, b in zip(self.range(), other.range()):
#             if not bas.loosely_equals(a, b):
#                 return False
#         return True
#
#     def increment(self, i: int, delta: float):
#         self.set(i, self.float(i) + delta)
#
#     def dot_product(self, other: Floats) -> float:
#         assert isinstance(other, Floats)
#         assert self.len() == other.len()
#         result = 0.0
#         for a, b in zip(self.range(), other.range()):
#             result += a * b
#
#         return result
#
#     def minus(self, other: Floats) -> Floats:
#         assert isinstance(other, Floats)
#         assert self.len() == other.len()
#         result = floats_empty()
#         for a, b in zip(self.range(), other.range()):
#             result.add(a - b)
#         return result
#
#     def pretty_string(self) -> str:
#         ss = self.strings()
#         assert isinstance(ss, Strings)
#         return ss.concatenate_fancy('{', ',', '}')
#
#     def strings(self) -> Strings:
#         result = strings_empty()
#         for x in self.range():
#             result.add(bas.string_from_float(x))
#         assert isinstance(result, Strings)
#         return result
#
#     def deep_copy(self) -> Floats:  # returns Floats
#         result = floats_empty()
#         for x in self.range():
#             result.add(x)
#         return result
#
#     def squared(self) -> float:
#         return self.dot_product(self)
#
#     def append(self, others: Floats):  # others are of type floats
#         for x in others.range():
#             self.add(x)
#
#     def tidy_surrounder(self) -> bas.Interval:
#         return self.interval().tidy_surrounder()
#
#     def interval(self) -> bas.Interval:
#         assert self.len() > 0
#         v0 = self.float(0)
#         iv = bas.interval_create(v0, v0)
#
#         for v in self.range():
#             iv.expand_to_include(v)
#
#         return iv
#
#     def list(self) -> List[float]:
#         return self.m_floats
#
#     def min(self) -> float:
#         return self.float(self.argmin())
#
#     def argmin(self) -> int:
#         return self.arg_extreme(False)
#
#     def max(self) -> float:
#         return self.float(self.argmax())
#
#     def argmax(self) -> int:
#         return self.arg_extreme(True)
#
#     def arg_extreme(self, is_max: bool) -> int:
#         assert self.len() > 0
#         extreme_index = 0
#         extreme_value = self.float(extreme_index)
#
#         for i, v in enumerate(self.range()):
#             found_better = (v > extreme_value) if is_max else (v < extreme_value)
#             if found_better:
#                 extreme_value = v
#                 extreme_index = i
#
#         return extreme_index
#
#     def tidy_extremes(self) -> Tuple[float, float]:
#         return self.interval().extremes()
#
#     def plus(self, other: Floats) -> Floats:
#         assert isinstance(other, Floats)
#         result = floats_empty()
#         for a, b in zip(self.range(), other.range()):
#             result.add(a + b)
#         assert isinstance(result, Floats)
#         return result
#
#     def subset(self, indexes: Ints) -> Floats:
#         result = floats_empty()
#         for i in indexes.range():
#             result.add(self.float(i))
#         return result
#
#     def indexes_of_sorted(self) -> Ints:
#         return indexes_of_sorted(self.m_floats)
#
#     def without_leftmost_element(self) -> Floats:
#         return self.without_n_leftmost_elements(1)
#
#     def without_n_leftmost_elements(self, n: int) -> Floats:
#         assert n <= self.len()
#         result = floats_empty()
#         for i in range(n, self.len()):
#             result.add(self.float(i))
#         return result
#
#     def sum_squares(self) -> float:
#         return self.dot_product(self)
#
#     def median_helper(self) -> float:
#         le = self.len()
#         assert le > 0
#         half_length = le // 2  # integer division, rounds down
#         assert isinstance(half_length, int)
#         if le % 2 == 0:
#             v0, v1 = self.kth_smallest_two_elements(half_length)
#             return (v0 + v1) / 2
#         else:
#             return self.kth_smallest_element(half_length)
#
#     def median_slow(self) -> float:
#         s = self.sorted()
#         h = self.len() // 2
#         if s.len() % 2 == 0:
#             return (s.float(h) + s.float(h + 1)) / 2
#         else:
#             return s.float(h)
#
#     def median(self) -> float:
#         result = self.median_helper()
#         print(f' remove this slow test')
#         assert bas.loosely_equals(result, self.median_slow())
#         return result
#
#     def kth_smallest_two_elements(self, k: int) -> Tuple[float, float]:
#         assert 0 <= k < self.len() - 1
#
#         lower, pivot, higher = self.split_with_random_pivot()
#
#         if k + 1 < lower.len():
#             return lower.kth_smallest_two_elements(k)
#         elif k + 1 == lower.len():
#             return lower.max(), pivot
#         elif k == lower.len():
#             return pivot, higher.min()
#         else:
#             return higher.kth_smallest_two_elements(k - lower.len() - 1)
#
#     def sorted(self) -> Floats:
#         return self.subset(self.indexes_of_sorted())
#
#     def kth_smallest_element(self, k: int) -> float:
#         assert 0 <= k < self.len()
#
#         lower, pivot, higher = self.split_with_random_pivot()
#
#         if k < lower.len():
#             return lower.kth_smallest_element(k)
#         elif k == lower.len():
#             return pivot
#         else:
#             return higher.kth_smallest_element(k - lower.len() - 1)
#
#     def split_with_random_pivot(self) -> Tuple[Floats, float, Floats]:
#         pivot_index = bas.int_random(self.len())
#         pivot = self.float(pivot_index)
#         lower = floats_empty()
#         higher = floats_empty()
#
#         for i, f in enumerate(self.range()):
#             if i != pivot_index:
#                 if f < pivot:
#                     lower.add(f)
#                 else:
#                     higher.add(f)
#
#         return lower, pivot, higher
#
#     def multiply_by(self, scale):
#         for i, f in enumerate(self.m_floats):
#             self.m_floats[i] *= scale
#
#     def map_product_with(self, other: Floats) -> Floats:
#         result = floats_empty()
#         for a, b in zip(self.range(), other.range()):
#             result.add(a * b)
#         return result
#
#     def is_loosely_constant(self) -> bool:
#         if self.len() < 2:
#             return True
#         first = self.float(0)
#         for f in self.range():
#             if not bas.loosely_equals(first, f):
#                 return False
#         return True
#
#     def times_scalar(self, scale: float) -> Floats:
#         result = floats_empty()
#         for x in self.range():
#             result.add(x * scale)
#         return result
#
#     def distance_to(self, other: Floats) -> float:
#         return math.sqrt(self.distance_squared_to(other))
#
#     def distance_squared_to(self, other: Floats) -> float:
#         return self.minus(other).sum_squares()
#
#     def mean(self) -> float:
#         assert self.len() > 0
#         return self.sum() / self.len()
#
#
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

        for k, v in enumerate(self.m_key_to_name.range()):
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

    def key_from_name(self, name: str) -> Tuple[int, bool]:
        if name in self.m_name_to_key:
            return self.m_name_to_key[name], True
        else:
            return -77, False

    def name_from_key(self, k: int) -> str:
        assert 0 <= k < self.len()
        return self.m_key_to_name.string(k)

    def range_keys(self) -> Iterator[str]:
        for s in self.m_key_to_name.range():
            yield s

    def names(self):  # returns Strings
        result = self.m_key_to_name
        assert isinstance(result, Strings)
        return result

    def equals(self, other: Namer) -> bool:
        return self.m_key_to_name.equals(other.m_key_to_name)


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


def concatenate_list_of_strings(ls: List[str]) -> str:
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
    def __init__(self, ins: List[int]):
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
        assert isinstance(i, int)
        self.m_ints.append(i)

    def set(self, i: int, v: int):
        assert 0 <= i < self.len()
        self.m_ints[i] = v

    def increment_by_one(self, i: int):
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
            result.increment_by_one(v)
        return result

    def min(self) -> int:
        return self.int(self.argmin())

    def invert_index(self):  # return Ints
        if bas.expensive_assertions2:
            assert self.sort().equals(identity_ints(self.len()))

        if self.len() > 0:
            assert self.min() >= 0
        result = ints_empty()
        for i, v in enumerate(self.range()):
            assert v >= 0
            while result.len() <= v:
                result.add(-777)
            assert v < result.len()
            assert result.int(v) < 0
            result.set(v, i)

        if bas.expensive_assertions2:
            for v, i in enumerate(result.range()):
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
        for index in indexes.range():
            result.add(self.int(index))
        assert isinstance(result, Ints)
        return result

    def equals(self, other) -> bool:  # other is type Ints
        assert isinstance(other, Ints)
        if other.len() != self.len():
            return False

        for a, b in zip(self.range(), other.range()):
            if a != b:
                return False

        return True

    def pretty_string(self) -> str:
        ss = self.strings()
        assert isinstance(ss, Strings)
        return ss.concatenate_fancy('{', ',', '}')

    def strings(self):  # returns object of type Strings
        result = strings_empty()
        for i in self.range():
            result.add(bas.string_from_int(i))
        assert isinstance(result, Strings)
        return result

    def contains_duplicates(self) -> bool:
        dic = {}
        for v in self.range():
            if v in dic:
                return True
            else:
                dic[v] = True
        return False

    def list(self) -> List[int]:
        return self.m_ints

    def plus(self, other):  # other is Ints and return is Ints
        assert isinstance(other, Ints)
        assert self.len() == other.len()
        result = ints_empty()
        for a, b in zip(self.range(), other.range()):
            result.add(a + b)
        assert isinstance(result, Ints)
        return result

    def shuffle(self):
        for i in range(self.len()):
            other_index = i + bas.int_random(self.len() - i)
            self.swap_elements(i, other_index)

    def swap_elements(self, index_a: int, index_b: int):
        if index_a != index_b:
            temp = self.int(index_a)
            self.set(index_a, self.int(index_b))
            self.set(index_a, temp)

    def first_n_elements(self, n_in_result: int) -> Ints:
        result = ints_empty()
        for i in range(n_in_result):
            result.add(self.int(i))
        return result

    def last_n_elements(self, n_in_result: int) -> Ints:
        result = ints_empty()
        for i in range(self.len() - n_in_result - 1, self.len()):
            result.add(self.int(i))
        return result

    def sum(self) -> int:
        return sum(self.m_ints)

    def is_weakly_increasing(self) -> bool:
        for i in range(self.len() - 1):
            if self.int(i) > self.int(i + 1):
                return False
        return True

    def last_element(self) -> int:
        assert self.len() > 0
        return self.int(self.len() - 1)

    def deep_copy(self) -> Ints:
        result = ints_empty()
        for i in self.range():
            result.add(i)
        return result

    def copied_with_one_element_added(self, i: int) -> Ints:
        result = self.deep_copy()
        result.add(i)
        return result

    def is_strictly_increasing(self) -> bool:
        if self.len() < 2:
            return True
        for i in range(self.len() - 1):
            if self.int(i) >= self.int(i + 1):
                return False
        return True


def indexes_of_sorted(li: list) -> Ints:
    pairs = zip(range(0, len(li)), li)
    sorted_pairs = sorted(pairs, key=lambda p: p[1])
    result = ints_empty()
    for pair in sorted_pairs:
        result.add(pair[0])

    if bas.expensive_assertions2:
        assert isinstance(result, Ints)
        assert result.len() == len(li)
        assert result.min() == 0
        assert result.max() == len(li) - 1
        assert not result.contains_duplicates()

        for r, i0 in enumerate(result.range()):
            if r < len(li) - 1:
                i1 = result.int(r + 1)
                assert li[i0] <= li[i1]

    return result


def identity_ints(n: int) -> Ints:
    result = ints_empty()
    for i in range(n):
        result.add(i)
    return result


class Strings:
    def __init__(self, ss: List[str]):
        self.m_strings = ss
        self.assert_ok()

    def assert_ok(self):
        assert is_list_of_string(self.m_strings)

    def contains_duplicates(self) -> bool:
        nm = namer_empty()
        for s in self.range():
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
        for i, s in enumerate(self.range()):
            if i > 0:
                result += mid
            result += s
        result += right
        return result

    def pretty_strings(self) -> List[str]:
        start = " {"
        finish = " }"
        result = []
        partial = start

        for s in self.range():
            if (len(start) + len(s) >= chars_per_line) or (len(partial) + 1 + len(s) < chars_per_line):
                partial = partial + " " + s
            else:
                result.append(partial + '\n')
                partial = bas.n_spaces(len(start)) + " " + s

        result.append(partial + finish + '\n')

        return result

    def concatenate_with_padding(self, col_to_n_chars: Ints) -> str:
        result = ""
        for i in range(self.len()):
            s = self.string(i)
            n_pads = col_to_n_chars.int(i) - len(s)
            assert n_pads >= 0
            if i < self.len() - 1:
                n_pads += 1
            result = result + s + bas.n_spaces(n_pads)
        return result

    def concatenate(self) -> str:
        result = ""
        for s in self.range():
            result = result + s
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
        for index in indexes.range():
            result.add(self.string(index))
        return result

    def equals(self, other) -> bool:  # other is of type strings
        assert isinstance(other, Strings)
        if self.len() != other.len():
            return False
        for a, b in zip(self.range(), other.range()):
            if a != b:
                return False
        return True

    def is_sorted(self) -> bool:
        if self.len() <= 1:
            return True

        for i in range(self.len() - 1):
            if self.string(i) > self.string(i + 1):
                return False

        return True

    def with_many(self, other: Strings) -> Strings:  # other type Strings, result type Strings
        assert isinstance(other, Strings)
        result = strings_empty()
        for s in self.range():
            result.add(s)
        for s in other.range():
            result.add(s)
        assert isinstance(result, Strings)
        return result

    def range(self) -> Iterator[str]:
        for s in self.m_strings:
            yield s

    def list(self) -> List[str]:
        return self.m_strings

    def decorate(self, left: str, right: str) -> Strings:
        result = strings_empty()
        for s in self.range():
            result.add(f'{left}{s}{right}')
        return result

    def deep_copy(self) -> Strings:
        result = strings_empty()
        for s in self.range():
            result.add(s)
        return result


def strings_empty() -> Strings:
    return Strings([])


def ints_empty() -> Ints:
    return Ints([])


class StringsArray:
    def __init__(self, sss: List[Strings]):
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
        for ss in self.range():
            result.add(pad + ss.concatenate_with_padding(col_to_n_chars))
        return result

    def col_to_num_chars(self) -> Ints:
        result = ints_empty()
        for ss in self.range():
            for c, s in enumerate(ss.range()):
                le = len(s)
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
    for c_as_string in s:
        c = bas.character_create(c_as_string)
        if c.equals(separator):
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


def bools_from_strings(ss: Strings) -> Tuple[Bools, bool]:
    result = bools_empty()
    for s in ss.range():
        b, ok = bas.bool_from_string(s)
        if ok:
            result.add(b)
        else:
            return bools_empty(), False

    return result, True


def floats_default():
    return floats_empty(0)


def floats_from_strings(ss: Strings) -> Tuple[Floats, bool]:
    print(f'enter floats_from_strings({ss.pretty_string()})')
    result = floats_empty(ss.len())
    for s in ss.range():
        f, ok = bas.float_from_string(s)
        if ok:
            result.add(f)
        else:
            return floats_default(), False

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


def strings_from_list(ls: List[str]) -> Strings:
    return Strings(ls)


def test_floats_ok(ls: List[str], total: float):
    ss = strings_from_list(ls)
    fs, ok = floats_from_strings(ss)
    assert ok
    assert bas.loosely_equals(fs.sum(), total)


def test_floats_bad(ls: List[str]):
    ss = strings_from_list(ls)
    fs, ok = floats_from_strings(ss)
    assert not ok


def unit_test_index_sort():
    ss = strings_from_split("i would have liked you to have been deep frozen too", bas.character_space())
    ranks = ss.indexes_of_sorted()
    ss2 = strings_from_split("been deep frozen have have i liked to too would you", bas.character_space())
    ss3 = ss.subset(ranks)
    assert ss2.equals(ss3)


def unit_test_transpose():
    s = row_indexed_smat_from_list([['a', 'b', 'c'], ['d', 'e', 'f']])
    t = row_indexed_smat_from_list([['a', 'd'], ['b', 'e'], ['c', 'f']])
    print(f's =\n{s.pretty_string()}')
    print(f't =\n{t.pretty_string()}')
    print(f's.transpose() =\n{s.transpose().pretty_string()}')
    assert s.transpose().equals(t)


def unit_test_median():
    fs = floats_varargs(3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0)
    med = fs.median()
    print(f'median = {med}')
    assert bas.loosely_equals(med, 3.5)
    assert bas.loosely_equals(fs.sorted().median(), 3.5)
    fs = fs.appended_with(floats_singleton(4.0))
    assert bas.loosely_equals(fs.median(), 4.0)
    assert bas.loosely_equals(fs.sorted().median(), 4.0)


def unit_test():
    assert not bas.string_contains('hello_said_peter', bas.character_space())
    assert bas.string_contains('hello_said peter', bas.character_space())
    test_floats_ok(["4", "-0.5", ".0"], 3.5)
    test_floats_ok([], 0.0)
    test_floats_bad(["5", "5", "5p"])
    test_floats_bad(["5", "5 5", "5999"])
    unit_test_namer()
    unit_test_index_sort()
    unit_test_transpose()
    unit_test_median()


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
    def __init__(self, n_cols: int):
        self.m_row_to_col_to_string = strings_array_empty()
        self.m_num_cols = n_cols
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_row_to_col_to_string, StringsArray)
        self.m_row_to_col_to_string.assert_ok()
        for r in self.m_row_to_col_to_string.range():
            assert r.len() == self.m_num_cols

    def num_cols(self) -> int:
        return self.m_num_cols

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

    def range_rows(self) -> Iterator[Strings]:
        for row in self.m_row_to_col_to_string.range():
            yield row

    def without_first_row(self):  # returns RowIndexedSmat
        result = row_indexed_smat_with_no_rows(self.num_cols())
        for index, row in enumerate(self.m_row_to_col_to_string.range()):
            if index > 0:
                result.add(row)
        assert isinstance(result, RowIndexedSmat)
        return result

    def transpose(self) -> RowIndexedSmat:
        result = row_indexed_smat_with_no_rows(self.num_rows())
        for c in self.range_columns_slow():
            result.add(c)
        assert isinstance(result, RowIndexedSmat)
        return result

    def equals(self, other) -> bool:  # other is also a row_indexed smat
        assert isinstance(other, RowIndexedSmat)
        if self.num_rows() != other.num_rows():
            return False

        for a, b in zip(self.range_rows(), other.range_rows()):
            if not a.equals(b):
                return False

        return True

    def range_columns_slow(self) -> Iterator[Strings]:
        for i in range(self.num_cols()):
            yield self.column_slow(i)

    def column_slow(self, c: int) -> Strings:
        result = strings_empty()
        for r in self.range_rows():
            result.add(r.string(c))
        return result


def row_indexed_smat_with_no_rows(n_cols: int) -> RowIndexedSmat:
    return RowIndexedSmat(n_cols)


def row_indexed_smat_from_list(li: List[List[str]]) -> RowIndexedSmat:
    assert isinstance(li, list)
    assert len(li) > 0
    assert isinstance(li[0], list)
    result = row_indexed_smat_with_no_rows(len(li[0]))
    for row in li:
        result.add(strings_from_list(row))
    return result


class Smat:
    def __init__(self, row_to_col_to_string: RowIndexedSmat):
        self.m_row_to_col_to_string = row_to_col_to_string
        self.m_col_to_row_to_string = row_to_col_to_string.transpose()
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_row_to_col_to_string, RowIndexedSmat)
        self.m_row_to_col_to_string.assert_ok()
        assert isinstance(self.m_row_to_col_to_string, RowIndexedSmat)
        self.m_col_to_row_to_string.assert_ok()

        assert self.m_row_to_col_to_string.equals(self.m_col_to_row_to_string.transpose())

    def num_rows(self) -> int:
        return self.m_row_to_col_to_string.num_rows()

    def num_cols(self) -> int:
        return self.m_row_to_col_to_string.num_cols()

    def column(self, c: int) -> Strings:
        return self.m_col_to_row_to_string.strings_from_row(c)

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
        for row in self.range_rows():
            result.add(row)
        return result

    def extended_with_headings(self, top_left: str, left: Strings, top: Strings):  # returns Smat
        assert self.num_rows() == left.len()
        assert self.num_cols() == top.len()
        first_row = strings_singleton(top_left).with_many(top)

        result = row_indexed_smat_single_row(first_row)

        for lf, r in zip(left.range(), self.range_rows()):
            row = strings_singleton(lf).with_many(r)
            result.add(row)

        return smat_create(result)

    def pretty_strings_with_headings(self, top_left: str, left: Strings, top: Strings) -> Strings:
        extended = self.extended_with_headings(top_left, left, top)
        assert isinstance(extended, Smat)
        return extended.pretty_strings()

    def range_rows(self) -> Iterator[Strings]:
        return self.m_row_to_col_to_string.range_rows()

    def without_first_row(self):  # returns Smat
        result = smat_create(self.m_row_to_col_to_string.without_first_row())
        assert isinstance(result, Smat)
        return result

    def range_columns(self) -> Iterator[Strings]:
        return self.m_col_to_row_to_string.range_rows()

    def equals(self, other: Smat) -> bool:
        return self.m_row_to_col_to_string.equals(other.m_row_to_col_to_string)


def smat_single_row(first_row: Strings) -> Smat:
    return smat_create(row_indexed_smat_single_row(first_row))


class FloatsArray:
    def __init__(self, li: List[Floats]):
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for ins in self.m_list:
            assert isinstance(ins, Floats)
            ins.assert_ok()

    def range(self) -> Iterator[Floats]:
        for fs in self.m_list:
            yield fs

    def add(self, fs: Floats):
        assert fs.is_filled()
        self.m_list.append(fs)

    def floats(self, i: int) -> Floats:
        assert 0 <= i < self.len()
        return self.m_list[i]

    def len(self) -> int:
        return len(self.m_list)

    def list_of_lists(self) -> List[List[float]]:
        result = []
        for fs in self.range():
            line = []
            for f in fs.range2():
                line.append(f)
            result.append(line)
        return result

    def float(self, row: int, index_in_row: int) -> float:
        return self.floats(row).float2(index_in_row)

    def loosely_equals(self, other: FloatsArray) -> bool:
        if not self.len() == other.len():
            return False

        for a, b in zip(self.range(), other.range()):
            if not a.loosely_equals(b):
                return False

        return True

    def deep_copy(self) -> FloatsArray:
        result = floats_array_empty()
        for fs in self.range():
            result.add(fs.deep_copy())
        return result

    def set(self, row: int, col: int, x: float):
        self.floats(row).set2(col, x)
        assert bas.loosely_equals(self.float(row, col), x)


# class RowIndexedFmat:
#     def __init__(self, n_cols: int):
#         self.m_num_cols = n_cols
#         self.m_row_to_col_to_value = floats_array_empty()
#         self.assert_ok()
#
#     def assert_ok(self):
#         assert isinstance(self.m_num_cols, int)
#         assert self.m_num_cols >= 0
#         assert isinstance(self.m_row_to_col_to_value, FloatsArray)
#         self.m_row_to_col_to_value.assert_ok()
#
#         for row in self.m_row_to_col_to_value.range():
#             assert row.len() == self.m_num_cols
#
#     def transpose(self):  # returns RowIndexedFmat
#         self_n_cols = self.num_cols()
#         self_n_rows = self.num_rows()
#         result_n_rows = self_n_cols
#         result_n_cols = self_n_rows
#
#         result = row_indexed_fmat_of_empties(result_n_rows, result_n_cols)
#         for self_row in self.range_rows():
#             result.add_column(self_row)
#
#         assert isinstance(result, RowIndexedFmat)
#         assert result.num_rows() == self.num_cols()
#         assert result.num_cols() == self.num_rows()
#         assert result.all_filled()
#
#         result.assert_ok()
#
#         return result
#
#     def add_column(self, col: Floats):
#         assert col.len() == self.num_rows()
#         for row, f in zip(self.range_rows(), col.range2()):
#             row.add(f)
#         self.m_num_cols += 1
#
#     def loosely_equals(self, other: RowIndexedFmat) -> bool:
#         assert isinstance(other, RowIndexedFmat)
#         if self.num_rows() != other.num_rows():
#             return False
#
#         if self.num_cols() != other.num_cols():
#             return False
#
#         for a, b in zip(self.range_rows(), other.range_rows()):
#             if not a.loosely_equals(b):
#                 return False
#         return True
#
#     def add_row(self, fs: Floats):
#         assert self.num_cols() == fs.len()
#         self.m_row_to_col_to_value.add(fs)
#
#     def range_rows(self) -> Iterator[Floats]:
#         return self.m_row_to_col_to_value.range()
#
#     def num_cols(self) -> int:
#         return self.m_num_cols
#
#     def num_rows(self) -> int:
#         return self.m_row_to_col_to_value.len()
#
#     def floats(self, r: int) -> Floats:
#         assert 0 <= r < self.num_rows()
#         return self.m_row_to_col_to_value.floats(r)
#
#     def increment(self, r: int, col: int, delta: float):
#         self.floats(r).increment2(col, delta)
#
#     def deep_copy(self) -> RowIndexedFmat:
#         result = row_indexed_fmat_with_no_rows(self.num_cols())
#         for r in self.range_rows():
#             result.add_row(r.deep_copy())
#         return result
#
#     def set(self, row: int, column: int, f: float):
#         self.m_row_to_col_to_value.floats(row).set2(column, f)
#         assert bas.loosely_equals(f, self.floats(row).float2(column))
#
#     def all_filled(self) -> bool:
#         for r in self.range_rows():
#             if not r.is_filled():
#                 return False
#         return True
#
#
# def row_indexed_fmat_with_no_rows(n_cols: int):
#     return RowIndexedFmat(n_cols)
#
#
# def row_indexed_fmat_of_empties(n_rows: int, n_cols: int) -> RowIndexedFmat:
#     result = row_indexed_fmat_with_no_rows(0)
#     for i in range(n_rows):
#         result.add_row(floats_empty(n_cols))
#     return result


class Fmat:
    def __init__(self, n_cols: int, fsa: FloatsArray):
        n_rows = fsa.len()
        transpose = floats_array_empty()
        for col in range(n_cols):
            row_of_transpose = floats_empty(n_rows)
            for r in fsa.range():
                assert r.len() == n_cols
                row_of_transpose.add(r.float2(col))
            transpose.add(row_of_transpose)

        self.m_row_to_col_to_value = fsa
        self.m_col_to_row_to_value = transpose
        self.assert_ok()

    def assert_ok(self):

        s = self.m_row_to_col_to_value
        assert isinstance(s, FloatsArray)
        s.assert_ok()
        t = self.m_col_to_row_to_value
        assert isinstance(t, FloatsArray)
        t.assert_ok()

        for row in s.range():
            assert row.len() == self.num_cols()

        for col in t.range():
            assert col.len() == self.num_rows()

        for i in range(self.num_rows()):
            for j in range(self.num_cols()):
                if not bas.loosely_equals(s.float(i, j), t.float(j, i)):
                    print(f's.float(i, j) = {s.float(i, j)}')
                    print(f't.float(j, i) = {t.float(j, i)}')
                    bas.my_error('transpose invariant failed')

    def num_rows(self) -> int:
        return self.m_row_to_col_to_value.len()

    def row(self, r: int) -> Floats:
        return self.m_row_to_col_to_value.floats(r)

    def num_cols(self) -> int:
        assert self.num_rows() > 0
        return self.row(0).len()

    def column(self, c: int) -> Floats:
        return self.m_col_to_row_to_value.floats(c)

    def float2(self, r: int, c: int) -> float:
        return self.row(r).float2(c)

    def loosely_equals(self, other) -> bool:
        assert isinstance(other, Fmat)
        return self.m_row_to_col_to_value.loosely_equals(other.m_row_to_col_to_value)

    def times(self, x: Floats) -> Floats:
        assert self.num_cols() == x.len()
        result = floats_empty(self.num_rows())
        for row in self.range_rows():
            result.add(row.dot_product(x))
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
        return smat_create(self.row_indexed_smat())

    def pretty_strings_with_headings(self, top_left: str, left: Strings, top: Strings):
        return self.smat().pretty_strings_with_headings(top_left, left, top)

    def range_rows(self) -> Iterator[Floats]:
        for fs in self.m_row_to_col_to_value.range():
            yield fs

    def range_columns(self) -> Iterator[Floats]:
        for fs in self.m_col_to_row_to_value.range():
            yield fs

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def deep_copy(self) -> Fmat:
        return fmat_from_rows(self.num_cols(), self.m_row_to_col_to_value.deep_copy())

    def increment(self, row: int, column: int, delta: float):
        self.set(row, column, self.float2(row, column) + delta)

    def set(self, row: int, column: int, f: float):
        self.m_row_to_col_to_value.set(row, column, f)
        self.m_col_to_row_to_value.set(column, row, f)


def fmat_from_rows(n_cols: int, fsa: FloatsArray) -> Fmat:
    return Fmat(n_cols, fsa)


def floats_numpy_all_constant(n: int, c: float) -> FloatsNumpy:
    nda = np.empty(n)
    nda[:] = c
    return floats_numpy_from_ndarray(nda)


def floats_all_zero(n: int) -> Floats:
    return floats_all_constant(n, 0.0)


def floats_singleton(f: float) -> Floats:
    return floats_all_constant(1, f)


def row_indexed_smat_single_row(first_row: Strings) -> RowIndexedSmat:
    result = row_indexed_smat_with_no_rows(first_row.len())
    result.add(first_row)
    return result


def row_indexed_smat_unit(s: str) -> RowIndexedSmat:
    return row_indexed_smat_single_row(strings_singleton(s))


def row_indexed_smat_default():
    return row_indexed_smat_unit('default')


def row_indexed_smat_from_strings_array(ssa: StringsArray) -> Tuple[RowIndexedSmat, bas.Errmess]:
    if not ssa:
        return row_indexed_smat_default(), bas.errmess_error(
            "Can't make a row indexed smat from an empty array of strings")

    assert ssa.len() > 0
    result = row_indexed_smat_single_row(ssa.strings(0))
    for r in range(1, ssa.len()):
        ss = ssa.strings(r)
        if ss.len() != result.num_cols():
            return result, bas.errmess_error(
                f'first row of csv file has {result.num_cols()} items, but row {r} has {ss.len()} items')
        result.add(ss)

    return result, bas.errmess_ok()


def smat_create(rsm: RowIndexedSmat) -> Smat:
    return Smat(rsm)


def smat_unit(s: str) -> Smat:
    return smat_create(row_indexed_smat_unit(s))


def smat_default():
    return smat_unit("default")


def floats_array_singleton(fs: Floats) -> FloatsArray:
    result = floats_array_empty()
    result.add(fs)
    return result


def floats_array_from_smat(sm: Smat) -> Tuple[FloatsArray, bas.Errmess]:
    result = floats_array_empty()
    for ss in sm.range_rows():
        fs, ok = floats_from_strings(ss)
        if not ok:
            return floats_array_empty(), bas.errmess_error(f"Can't convert into numbers: {ss.pretty_string()}")
        result.add(fs)

    return result, bas.errmess_ok()


def fmat_default() -> Fmat:
    return fmat_from_rows(0, floats_array_empty())


def fmat_from_smat(sm: Smat) -> Tuple[Fmat, bas.Errmess]:
    rif, err = floats_array_from_smat(sm)
    if err.is_error():
        return fmat_default(), err

    n_cols = 0 if rif.len() == 0 else rif.floats(0).len()
    return fmat_from_rows(n_cols, rif), bas.errmess_ok()


def floats_array_of_zeroes(n_rows: int, n_cols: int) -> FloatsArray:
    result = floats_array_empty()
    for r in range(n_rows):
        result.add(floats_all_zero(n_cols))
    return result


def floats_array_empty() -> FloatsArray:
    return FloatsArray([])


#  *************************************************************************

class IntsArray:
    def __init__(self, li: List[Ints]):
        self.m_list = li
        self.assert_ok()

    def range(self) -> Iterator[Ints]:
        for fs in self.m_list:
            yield fs

    def add(self, ins: Ints):
        self.m_list.append(ins)

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for ins in self.m_list:
            assert isinstance(ins, Ints)
            ins.assert_ok()

    def ints(self, i: int) -> Ints:
        assert 0 <= i < self.len()
        return self.m_list[i]

    def len(self) -> int:
        return len(self.m_list)

    def pretty_strings(self) -> Strings:
        return self.strings_array().pretty_strings()

    def strings_array(self) -> StringsArray:
        result = strings_array_empty()
        for ins in self.range():
            result.add(ins.strings())
        return result

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('{', ',', '}')


class RowIndexedImat:
    def __init__(self, n_cols: int):
        self.m_num_cols = n_cols
        self.m_row_to_col_to_value = ints_array_empty()
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_num_cols, int)
        assert self.m_num_cols >= 0
        assert isinstance(self.m_row_to_col_to_value, IntsArray)
        self.m_row_to_col_to_value.assert_ok()

        for row in self.m_row_to_col_to_value.range():
            assert row.len() == self.m_num_cols

    def transpose(self) -> RowIndexedImat:
        self_n_cols = self.num_cols()
        result_n_rows = self_n_cols

        result = row_indexed_imat_with_no_columns(result_n_rows)
        for self_row in self.range_rows():
            result.add_column(self_row)

        assert isinstance(result, RowIndexedImat)
        assert result.num_rows() == self.num_cols()
        assert result.num_cols() == self.num_rows()

        result.assert_ok()

        return result

    def add_column(self, col: Ints):
        assert col.len() == self.num_rows()
        for row, f in zip(self.range_rows(), col.range()):
            row.add(f)
        self.m_num_cols += 1

    def equals(self, other) -> bool:
        assert isinstance(other, RowIndexedImat)
        for a, b in zip(self.range_rows(), other.range_rows()):
            if not a.equals(b):
                return False
        return True

    def add_row(self, ins: Ints):
        assert self.num_cols() == ins.len()
        self.m_row_to_col_to_value.add(ins)

    def range_rows(self) -> Iterator[Ints]:
        return self.m_row_to_col_to_value.range()

    def num_cols(self) -> int:
        return self.m_num_cols

    def num_rows(self) -> int:
        return self.m_row_to_col_to_value.len()

    def ints(self, r: int) -> Ints:
        assert 0 <= r < self.num_rows()
        return self.m_row_to_col_to_value.ints(r)

    def increment_by_one(self, r: int, col: int):
        self.ints(r).increment_by_one(col)

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def pretty_strings(self) -> Strings:
        return self.m_row_to_col_to_value.pretty_strings()


def row_indexed_imat_with_no_rows(n_cols: int):
    return RowIndexedImat(n_cols)


def row_indexed_imat_with_no_columns(n_rows: int) -> RowIndexedImat:
    result = row_indexed_imat_with_no_rows(0)
    for i in range(n_rows):
        result.add_row(ints_empty())
    return result


class Imat:
    def __init__(self, rif: RowIndexedImat):
        self.m_row_to_col_to_value = rif
        self.m_col_to_row_to_value = rif.transpose()
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_row_to_col_to_value, RowIndexedImat)
        self.m_row_to_col_to_value.assert_ok()
        assert isinstance(self.m_col_to_row_to_value, RowIndexedImat)
        self.m_col_to_row_to_value.assert_ok()
        tr = self.m_col_to_row_to_value.transpose()
        assert isinstance(tr, RowIndexedImat)
        assert self.m_row_to_col_to_value.equals(tr)

    def num_rows(self) -> int:
        return self.m_row_to_col_to_value.num_rows()

    def row(self, r: int) -> Ints:
        return self.m_row_to_col_to_value.ints(r)

    def num_cols(self) -> int:
        assert self.num_rows() > 0
        return self.row(0).len()

    def column(self, c: int) -> Ints:
        return self.m_col_to_row_to_value.ints(c)

    def int(self, r: int, c: int) -> int:
        return self.row(r).int(c)

    def loosely_equals(self, other) -> bool:
        assert isinstance(other, Imat)
        return self.m_row_to_col_to_value.equals(other.m_row_to_col_to_value)

    def pretty_strings(self) -> Strings:
        return self.smat().pretty_strings()

    def row_indexed_smat(self) -> RowIndexedSmat:
        assert self.num_rows() > 0
        result = row_indexed_smat_single_row(self.row(0).strings())
        for i in range(1, self.num_rows()):
            result.add(self.row(i).strings())
        return result

    def smat(self) -> Smat:
        return smat_create(self.row_indexed_smat())

    def pretty_strings_with_headings(self, top_left: str, left: Strings, top: Strings):
        return self.smat().pretty_strings_with_headings(top_left, left, top)

    def range_rows(self) -> Iterator[Ints]:
        return self.m_row_to_col_to_value.range_rows()

    def range_columns(self) -> Iterator[Ints]:
        return self.m_col_to_row_to_value.range_rows()

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def list_of_lists(self) -> List[List[int]]:
        result = []
        for ins in self.range_rows():
            result.append(ins.list())
        return result


def imat_create(rif: RowIndexedImat) -> Imat:
    return Imat(rif)


def ints_all_constant(n, c: int) -> Ints:
    result = ints_empty()
    for i in range(n):
        result.add(c)
    return result


def ints_all_zero(n: int) -> Ints:
    return ints_all_constant(n, 0)


def ints_singleton(i: int) -> Ints:
    result = ints_empty()
    result.add(i)
    return result


def ints_array_empty() -> IntsArray:
    return IntsArray([])


def ints_array_of_empties(n_elements: int) -> IntsArray:
    result = ints_array_empty()
    for i in range(n_elements):
        result.add(ints_empty())
    return result


def row_indexed_imat_of_zeroes(n_rows: int, n_cols: int) -> RowIndexedImat:
    result = row_indexed_imat_with_no_rows(n_cols)
    for r in range(n_rows):
        result.add_row(ints_all_zero(n_cols))
    return result


def floats_from_range(lo: float, hi: float, n_elements: int) -> Floats:
    assert lo < hi
    assert n_elements > 1
    delta = (hi - lo) / (n_elements - 1)
    result = floats_empty(n_elements)
    for i in range(n_elements - 1):
        result.add(lo + delta * i)
    result.add(hi)
    return result


def ints_identity(n: int) -> Ints:
    result = ints_empty()
    for i in range(n):
        result.add(i)
    return result


def ints_random_permutation(n: int) -> Ints:
    result = ints_identity(n)
    result.shuffle()
    return result


def floats_varargs(*li: float) -> Floats:
    assert isinstance(li, tuple)
    result = floats_empty(len(li))
    for x in li:
        assert isinstance(x, float)
        result.add(x)
    return result


def strings_varargs(*li: str) -> Strings:
    assert isinstance(li, tuple)
    result = strings_empty()
    for x in li:
        assert isinstance(x, str)
        result.add(x)
    return result


def fmat_of_zeroes(n_rows: int, n_columns: int) -> Fmat:
    return fmat_from_rows(n_columns, floats_array_of_zeroes(n_rows, n_columns))


def strings_array_varargs(*li: Strings) -> StringsArray:
    assert isinstance(li, tuple)
    result = strings_array_empty()
    for ss in li:
        assert isinstance(ss, Strings)
        result.add(ss)
    return result


def floats_random(n_rows: int, iv: bas.Interval) -> Floats:
    result = floats_empty(n_rows)
    for i in range(n_rows):
        result.add(iv.random())
    return result


def floats_random_unit(n_rows: int) -> Floats:
    return floats_random(n_rows, bas.interval_unit())


def floats_array_from_list_of_lists(v2f: List[List[float]]) -> FloatsArray:
    result = floats_array_empty()
    for fs in v2f:
        result.add(floats_from_list2(fs))
    return result


use_numpy = True


def floats_empty(capacity: int) -> Floats:
    if use_numpy:
        return floats_numpy_empty(capacity)
    else:
        return floats_list_empty(capacity)


def floats_from_list2(li: List[float]) -> Floats:
    if use_numpy:
        return floats_numpy_from_list2(li)
    else:
        return floats_list_from_list(li)


def floats_all_constant(n_elements: int, value: float) -> Floats:
    if use_numpy:
        return floats_numpy_all_constant(n_elements, value)
    else:
        return floats_list_all_constant(n_elements, value)


def strings_all_constant(n_elements: int, s: str)->Strings:
    result = strings_empty()
    for i in range(n_elements):
        result.add(s)
    return result