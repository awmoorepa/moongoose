import enum
import datset.ambasic as bas
import datset.amarrays as arr


class Distype(enum.Enum):
    gaussian = 0
    binomial = 1
    multinomial = 2


class Gaussian:
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


class Binomial:
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


class Multinomial:
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


class Distribution:
    def __init__(self, dt: Distype, data):
        self.m_distype = dt
        self.m_any = data
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_distype, Distype)
        dt = self.distype()
        if dt == Distype.gaussian:
            assert isinstance(self.m_any, Gaussian)
            self.m_any.assert_ok()
        elif dt == Distype.binomial:
            assert isinstance(self.m_any, Binomial)
            self.m_any.assert_ok()
        elif dt == Distype.multinomial:
            assert isinstance(self.m_any, Multinomial)
            self.m_any.assert_ok()
        else:
            bas.my_error("bad distype")

    def distype(self) -> Distype:
        return self.m_distype

    def explain(self):
        print(self.pretty_string())

    def pretty_string(self) -> str:
        dt = self.distype()
        if dt == Distype.binomial:
            return self.binomial().pretty_string()
        elif dt == Distype.gaussian:
            return self.gaussian().pretty_string()
        elif dt == Distype.multinomial:
            return self.multinomial().pretty_string()
        else:
            bas.my_error("bad distype")

    def binomial(self) -> Binomial:
        assert self.distype() == Distype.binomial
        assert isinstance(self.m_any, Binomial)
        return self.m_any

    def gaussian(self) -> Gaussian:
        assert self.distype() == Distype.gaussian
        assert isinstance(self.m_any, Gaussian)
        return self.m_any

    def multinomial(self) -> Multinomial:
        assert self.distype() == Distype.multinomial
        assert isinstance(self.m_any, Multinomial)
        return self.m_any


def distribution_from_gaussian(g: Gaussian) -> Distribution:
    return Distribution(Distype.gaussian, g)


def distribution_from_binomial(bn: Binomial) -> Distribution:
    return Distribution(Distype.binomial, bn)


def binomial_create(p: float) -> Binomial:
    return Binomial(p)


def gaussian_create(mu: float, sdev: float) -> Gaussian:
    return Gaussian(mu, sdev)


def distribution_from_multinomial(mn: Multinomial) -> Distribution:
    return Distribution(Distype.multinomial, mn)


def multinomial_create(probs: arr.Floats) -> Multinomial:
    return Multinomial(probs)
