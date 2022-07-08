from __future__ import annotations

import math
from abc import abstractmethod, ABC
from typing import Tuple

import datset.amarrays as arr
import datset.ambasic as bas
import datset.dset as dat


def binomial_default() -> Binomial:
    return binomial_create(0.0)


def gaussian_default() -> Gaussian:
    return gaussian_create(0.0, 1.0)


def multinomial_default():
    vns = dat.valnames_from_strings(arr.strings_varargs('cat', 'dog'))
    probs = arr.floats_varargs(0.3, 0.7)
    return multinomial_create(vns, probs)


class Distribution(ABC):

    @abstractmethod
    def assert_ok(self):
        pass

    def explain(self):
        print(self.pretty_string())

    @abstractmethod
    def pretty_string(self) -> str:
        pass

    @abstractmethod
    def as_floats(self) -> arr.Floats:
        pass

    @abstractmethod
    def loglike(self, a: dat.Atom) -> float:
        pass


class Gaussian(Distribution):
    def loglike(self, a: dat.Atom) -> float:
        return self.loglike_of_sample(a.float())

    def as_floats(self) -> arr.Floats:
        return arr.floats_varargs(self.mean(), self.sdev())

    def __init__(self, mu: float, sdev: float):
        self.m_mean = mu
        self.m_sdev = sdev
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_mean, float)
        assert isinstance(self.m_sdev, float)
        assert self.m_sdev > 0

    def pretty_string(self) -> str:
        return f'Normal(mean={self.mean()},sdev={self.sdev()})'

    def mean(self) -> float:
        return self.m_mean

    def sdev(self) -> float:
        return self.m_sdev

    def loglike_of_sample(self, x: float) -> float:
        log_one_over_sigma_root_two_pi = -math.log(2 * math.pi * self.sdev())
        delta = (x - self.mean()) / (2 * self.sdev())
        return log_one_over_sigma_root_two_pi - delta * delta


class Binomial(Distribution):
    def loglike(self, a: dat.Atom) -> float:
        b = a.bool()
        p_true = self.theta()
        p_b = p_true if b else (1 - p_true)
        assert p_b > 0
        return math.log(p_b)

    def as_floats(self) -> arr.Floats:
        return arr.floats_singleton(self.theta())

    def __init__(self, theta: float):
        self.m_theta = theta
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_theta, float)
        assert 0 <= self.m_theta <= 1

    def pretty_string(self) -> str:
        return f'Binomial(p={self.theta()})'

    def theta(self) -> float:
        return self.m_theta


log_of_prob_of_zero: float = -1.0e6


class Multinomial(Distribution):
    def loglike(self, a: dat.Atom) -> float:
        value, ok = self.value_from_valname(a.valname())
        if not ok:
            return log_of_prob_of_zero
        else:
            return self.log_prob(value)

    def as_floats(self) -> arr.Floats:
        return self.floats()

    def __init__(self, vns: dat.Valnames, probs: arr.Floats):
        self.m_valnames = vns
        self.m_probs = probs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_valnames, dat.Valnames)
        self.m_valnames.assert_ok()
        assert isinstance(self.m_probs, arr.Floats)
        self.m_probs.assert_ok()
        assert bas.loosely_equals(1.0, self.m_probs.sum())
        assert self.m_valnames.len() == self.m_probs.len()

    def pretty_string(self) -> str:
        ss = self.floats().strings()
        return ss.concatenate_fancy('PDF(', ',', ')')

    def floats(self) -> arr.Floats:
        return self.m_probs

    def len(self) -> int:
        return self.floats().len()

    def log_prob(self, value: int) -> float:
        p = self.prob(value)
        assert p > 0
        if p < 1e-80:
            return log_of_prob_of_zero
        return math.log(p)

    def prob(self, value: int) -> float:
        assert 0 <= value < self.len()
        return self.floats().float(value)

    def value_from_valname(self, target: dat.Valname) -> Tuple[int, bool]:
        return self.valnames().value(target)

    def valnames(self) -> dat.Valnames:
        return self.m_valnames


def binomial_create(p: float) -> Binomial:
    return Binomial(p)


def gaussian_create(mu: float, sdev: float) -> Gaussian:
    return Gaussian(mu, sdev)


def multinomial_create(vns: dat.Valnames, probs: arr.Floats) -> Multinomial:
    return Multinomial(vns, probs)


def multinomial_from_binomial(bi: Binomial) -> Multinomial:
    p1 = bi.theta()
    p0 = 1 - p1
    probs = arr.floats_singleton(p0)
    probs.add(p1)
    vns = dat.valnames_from_strings(arr.strings_varargs('False', 'True'))
    return multinomial_create(vns, probs)
