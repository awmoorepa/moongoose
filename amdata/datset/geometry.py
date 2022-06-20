from __future__ import annotations

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


class Color:
    def __init__(self, r: float, g: float, b: float):
        self.m_r = r
        self.m_g = g
        self.m_b = b
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_r, float)
        assert 0 <= self.m_r <= 1
        assert isinstance(self.m_g, float)
        assert 0 <= self.m_g <= 1
        assert isinstance(self.m_b, float)
        assert 0 <= self.m_b <= 1

    def r(self) -> float:
        return self.m_r

    def g(self) -> float:
        return self.m_g

    def b(self) -> float:
        return self.m_b

    def darken(self) -> Color:
        return self.average_with(black())

    def lighten(self) -> Color:
        return self.average_with(white())

    def average_with(self, other) -> Color:
        return self.times(0.5).plus(other.times(0.5))

    def rgb(self) -> Tuple[float, float, float]:
        return self.m_r, self.m_g, self.m_b

    def times(self, scale: float) -> Color:
        r, g, b = self.rgb()
        return color_create(r * scale, g * scale, b * scale)

    def plus(self, other: Color) -> Color:
        r1, g1, b1 = self.rgb()
        r2, g2, b2 = other.rgb()
        return color_create(r1 + r2, g1 + g2, b1 + b2)

    def list(self) -> List[float]:
        r, g, b = self.rgb()
        return [r, g, b]


def color_create(r: float, g: float, b: float) -> Color:
    return Color(r, g, b)


def red() -> Color:
    return color_create(1.0, 0.0, 0.0)


def green() -> Color:
    return color_create(0.0, 1.0, 0.0)


def blue() -> Color:
    return color_create(0.0, 0.0, 1.0)


def black() -> Color:
    return color_create(0.0, 0.0, 0.0)


def yellow() -> Color:
    return color_create(1.0, 1.0, 0.0)


def white() -> Color:
    return color_create(1.0, 1.0, 1.0)


def orange() -> Color:
    return color_create(1.0, 0.9, 0.2)


def cyan() -> Color:
    return color_create(0.0, 1.0, 1.0)


def purple() -> Color:
    return color_create(1.0, 0.2, 1.0)


def pink() -> Color:
    return color_create(1.0, 0.5, 0.5)


def magenta() -> Color:
    return color_create(1.0, 0.0, 1.0)


class Colors:
    def __init__(self, li: List[Color]):
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for c in self.m_list:
            assert isinstance(c, Color)
            c.assert_ok()

    def color(self, i: int) -> Color:
        assert 0 <= i < self.len()
        return self.m_list[i]

    def len(self) -> int:
        return len(self.m_list)

    def range(self) -> Iterator[Color]:
        for c in self.m_list:
            yield c

    def add(self, c: Color):
        self.m_list.append(c)


def colors_empty() -> Colors:
    return Colors([])


def color_from_int(color_num: int):
    n_colors = 9
    i = color_num % n_colors
    if i == 0:
        return blue()
    elif i == 1:
        return green()
    elif i == 2:
        return red()
    elif i == 3:
        return purple()
    elif i == 4:
        return orange()
    elif i == 5:
        return magenta()
    elif i == 6:
        return cyan()
    elif i == 7:
        return white().darken()
    elif i == 8:
        return yellow().darken()
    else:
        bas.my_error("code can't reach here")


def color_cycle(n_elements: int) -> Colors:
    result = colors_empty()
    for i in range(n_elements):
        result.add(color_from_int(i))
    return result
