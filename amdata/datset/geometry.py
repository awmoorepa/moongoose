from typing import List, Iterator, Tuple

import datset.ambasic as bas
import datset.amarrays as arr


class Vec:
    def __init__(self, x: float, y: float):
        self.m_x = x
        self.m_y = y
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_x, float)
        assert isinstance(self.m_y, float)

    def loosely_dominates(self, other) -> bool:  # other is type Vec
        assert isinstance(other, Vec)
        return bas.loosely_lte(other.x(), self.x()) and bas.loosely_lte(other.y(), self.y())

    def x(self) -> float:
        return self.m_x

    def y(self) -> float:
        return self.m_y


class Vecs:
    def __init__(self, vs: List[Vec]):
        self.m_list = vs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for v in self.m_list:
            assert isinstance(v, Vec)

    def add(self, v: Vec):
        self.m_list.append(v)

    def range(self) -> Iterator[Vec]:
        for v in self.m_list:
            yield v

    def unzip(self) -> Tuple[arr.Floats, arr.Floats]:
        xs = arr.floats_empty()
        ys = arr.floats_empty()
        for v in self.range():
            xs.add(v.x())
            ys.add(v.y())
        return xs, ys


def vecs_empty() -> Vecs:
    return Vecs([])


def vec_create(x: float, y: float) -> Vec:
    return Vec(x, y)


def vecs_from_floats(xs: arr.Floats, ys: arr.Floats) -> Vecs:
    assert xs.len() == ys.len()
    result = vecs_empty()
    for x, y in zip(xs.range(), ys.range()):
        result.add(vec_create(x, y))
    return result


class Rect:
    def __init__(self, lo: Vec, hi: Vec):
        self.m_lo = lo
        self.m_hi = hi
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_lo, Vec)
        self.m_lo.assert_ok()
        assert isinstance(self.m_hi, Vec)
        self.m_hi.assert_ok()
        assert self.hi().loosely_dominates(self.lo())

    def hi(self) -> Vec:
        return self.m_hi

    def lo(self) -> Vec:
        return self.m_lo

    def loosely_contains_vecs(self, vs: Vecs) -> bool:
        for v in vs.range():
            if not self.loosely_contains_vec(v):
                return False
        return True

    def loosely_contains_vec(self, v: Vec) -> bool:
        return v.loosely_dominates(self.lo()) and self.hi().loosely_dominates(v)


def rect_create(lo: Vec, hi: Vec) -> Rect:
    return Rect(lo, hi)


def rect_from_intervals(horizontal: bas.Interval, vertical: bas.Interval) -> Rect:
    lo = vec_create(horizontal.lo(), vertical.lo())
    hi = vec_create(horizontal.hi(), vertical.hi())
    return rect_create(lo, hi)
