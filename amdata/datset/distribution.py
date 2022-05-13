import enum
import datset.ambasic as bas


class Distype(enum.Enum):
    gaussian = 0
    binomial = 1


class Gaussian:
    def __init__(self, mu: float, sdev: float):
        self.m_mean = mu
        self.m_sdev = sdev
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_mean, float)
        assert isinstance(self.m_sdev, float)
        assert self.m_sdev > 0


class Binomial:
    def __init__(self, theta: float):
        self.m_theta = theta
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_theta, float)
        assert 0 <= self.m_theta <= 1


class Distribution:
    def __init__(self, dt: Distype, data):
        self.m_distype = dt
        self.m_data = data
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_distype, Distype)
        dt = self.distype()
        if dt == Distype.gaussian:
            assert isinstance(self.m_data, Gaussian)
            self.m_data.assert_ok()
        elif dt == Distype.binomial:
            assert isinstance(self.m_data, Binomial)
            self.m_data.assert_ok()
        else:
            bas.my_error("bad distype")

    def distype(self) -> Distype:
        return self.m_distype


def distribution_from_gaussian(g: Gaussian) -> Distribution:
    return Distribution(Distype.gaussian, g)
