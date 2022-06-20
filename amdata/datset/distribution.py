from __future__ import annotations

from abc import abstractmethod, ABC

import datset.amarrays as arr
import datset.ambasic as bas


def binomial_default() -> Binomial:
    return binomial_create(0.0)


def gaussian_default() -> Gaussian:
    return gaussian_create(0.0, 1.0)


def multinomial_default():
    return multinomial_create(arr.floats_all_constant(4, 0.25))


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


class Gaussian(Distribution):
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


class Binomial(Distribution):
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


class Multinomial(Distribution):
    def as_floats(self) -> arr.Floats:
        return self.floats()

    def __init__(self, probs: arr.Floats):
        self.m_probs = probs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_probs, arr.Floats)
        self.m_probs.assert_ok()
        assert bas.loosely_equals(1.0, self.m_probs.sum())

    def pretty_string(self) -> str:
        ss = self.floats().strings()
        return ss.concatenate_fancy('PDF(', ',', ')')

    def floats(self) -> arr.Floats:
        return self.m_probs

    def len(self) -> int:
        return self.floats().len()


def binomial_create(p: float) -> Binomial:
    return Binomial(p)


def gaussian_create(mu: float, sdev: float) -> Gaussian:
    return Gaussian(mu, sdev)


def multinomial_create(probs: arr.Floats) -> Multinomial:
    return Multinomial(probs)


def multinomial_from_binomial(bi: Binomial) -> Multinomial:
    p1 = bi.theta()
    p0 = 1 - p1
    probs = arr.floats_singleton(p0)
    probs.add(p1)
    return multinomial_create(probs)
