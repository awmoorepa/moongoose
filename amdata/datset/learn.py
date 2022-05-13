import enum
import math

import datset.ambasic as bas
import datset.amarrays as arr
import datset.dset as dat
import datset.numset as noo
import datset.distribution as dis

class Learntype(enum.Enum):
    logistic = 0
    linear = 1
    multinomial = 2

    def loglike(self, beta_xs: arr.Floats, output: dat.Column) -> float:
        result = 0.0
        for r in range(0, beta_xs.len()):
            result += self.loglike_of_element(beta_xs.float(r), output.atom(r))

        return result

    def loglike_of_element(self, beta_x: float, y: dat.Atom) -> float:
        if self == Learntype.linear:
            delta = y.float() - beta_x
            return -0.5 * delta * delta - bas.log_root_two_pi
        elif self == Learntype.logistic:
            q = q_from_y(y)
            return -math.log(logistic(q * beta_x))
        else:
            bas.my_error("bad learn type for loglike")


class Learner:
    def __init__(self, lt: Learntype):
        self.m_learn_type = lt
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_learn_type, Learntype)

    def learn_type(self) -> Learntype:
        return self.m_learn_type

    def train(self, inputs: dat.Datset, output: dat.NamedColumn):  # Returns Model
        lt = self.learn_type()
        if lt == Learntype.linear:
            return train_linear(inputs, output)
        elif lt == Learntype.logistic:
            return train_logistic(inputs, output)
        elif lt == Learntype.multinomial:
            bas.my_error("multinomial not implemented")
        else:
            bas.my_error("bad LearnType")


def learner_create(lt: Learntype) -> Learner:
    return Learner(lt)


def learner_type_logistic() -> Learner:
    return learner_create(Learntype.logistic)


def logistic(z: float) -> float:
    return 1 + math.exp(z)


def q_from_y(y_k: dat.Atom) -> float:
    if y_k.bool():
        return 1.0
    else:
        return -1.0


class Weights:
    def __init__(self, fs: arr.Floats):
        self.m_weights = fs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_weights, arr.Floats)
        self.m_weights.assert_ok()

    def loglike(self, inputs: arr.Fmat, output: dat.Column, lt: Learntype) -> float:
        beta_xs = self.beta_xs(inputs)
        return lt.loglike(beta_xs, output)

    def beta_xs(self, inputs: arr.Fmat) -> arr.Floats:
        result = arr.floats_empty()
        for r in range(0, inputs.num_rows()):
            x = inputs.row(r).dot_product(self.floats())
            result.add(x)
        return result

    def loglike_derivative(self, inputs: arr.Fmat, output: dat.Column, lt: Learntype, col: int) -> float:
        result = 0.0
        for r in range(0, inputs.num_rows()):
            x_k = inputs.row(r)
            y_k = output.atom(r)
            result += self.loglike_derivative_from_row(x_k, y_k, lt, col)
        return result

    def loglike_second_derivative(self, inputs: arr.Fmat, output: dat.Column, lt: Learntype, col: int) -> float:
        result = 0.0
        for r in range(0, inputs.num_rows()):
            x_k = inputs.row(r)
            y_k = output.atom(r)
            result += self.loglike_second_derivative_from_row(x_k, y_k, lt, col)
        return result

    def increment(self, col: int, delta: float):
        self.m_weights.increment(col, delta)

    def floats(self) -> arr.Floats:
        return self.m_weights

    def loglike_derivative_from_row(self, x_k: arr.Floats, y_k: dat.Atom, lt: Learntype, j: int) -> float:
        if lt == Learntype.linear:
            return x_k.float(j) * (y_k.float() - self.times(x_k))
        elif lt == Learntype.logistic:
            q_k = q_from_y(y_k)
            return -q_k * x_k.float(j) / logistic(-q_k * self.times(x_k))
        else:
            bas.my_error("bad LearnType for loglike")

    def loglike_second_derivative_from_row(self, x_k: arr.Floats, y_k: dat.Atom, lt: Learntype, j: int) -> float:
        if lt == Learntype.linear:
            x_kj = x_k.float(j)
            return -x_kj * x_kj
        elif lt == Learntype.logistic:
            qk = q_from_y(y_k)  # note the qks get multiplied out so not really needed
            qk_xk = qk * x_k.float(j)
            eee = -qk * self.times(x_k)
            exp_eee = math.exp(eee)
            loggy = 1 + exp_eee
            return qk_xk * qk_xk * exp_eee / (loggy * loggy)
        else:
            bas.my_error("bad learn type for log like")

    def times(self, x_k: arr.Floats) -> float:
        return self.floats().dot_product(x_k)

    def pretty_strings_with_introduction(self, nns: noo.Noomnames, intro: str) -> arr.Strings:
        assert self.len() == nns.len()
        result = arr.strings_singleton(intro)
        result.add('')
        result.add('where...')
        result.add('')
        result.append(self.pretty_strings(nns))
        return result

    def pretty_string_with_introduction(self, nns: noo.Noomnames, intro: str) -> str:
        return self.pretty_strings_with_introduction(nns, intro).concatenate_fancy('', '\n', '')

    def len(self) -> int:
        return self.floats().len()

    def pretty_strings(self, nns: noo.Noomnames) -> arr.Strings:
        return self.strings_array(nns).pretty_strings()

    def strings_array(self, nns: noo.Noomnames) -> arr.StringsArray:
        assert self.len() == nns.len()
        result = arr.strings_array_empty()
        for i in range(0, self.len()):
            ss = arr.strings_empty()
            ss.add(f'w[{nns.noomname(i).string()}]')
            ss.add('=')
            ss.add(f'{self.weight(i)}')
            result.add(ss)
        return result

    def weight(self, i: int) -> float:
        return self.floats().float(i)


class Modnames:
    def __init__(self, tfs: noo.Transformers, output: dat.Colname):
        self.m_transformers = tfs
        self.m_output_colname = output
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_transformers, noo.Transformers)
        self.m_transformers.assert_ok()
        assert isinstance(self.m_output_colname, dat.Colname)
        self.m_output_colname.assert_ok()
        assert not self.noomnames().contains(noo.noomname_from_colname(self.output_colname()))

    def transformers(self) -> noo.Transformers:
        return self.m_transformers

    def noomnames(self) -> noo.Noomnames:
        return self.transformers().noomnames()

    def output_colname(self) -> dat.Colname:
        return self.m_output_colname

class ModelLogistic:
    def __init__(self, ms: Modnames, ws: Weights):
        self.m_modnames = ms
        self.m_weights = ws
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_modnames, Modnames)
        self.m_modnames.assert_ok()
        assert isinstance(self.m_weights, Weights)
        self.m_weights.assert_ok()

    def pretty_string(self) -> str:
        ms = self.modnames()
        intro = f'P({ms.output_colname().string()}|x) = sigmoid(w^T x)'
        return self.weights().pretty_string_with_introduction(ms.noomnames(), intro)

    def weights(self) -> Weights:
        return self.m_weights

    def modnames(self) -> Modnames:
        return self.m_modnames


class ModelLinear:
    def __init__(self, mns: Modnames, ws: Weights):
        self.m_modnames = mns
        self.m_weights = ws
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_modnames, Modnames)
        self.m_modnames.assert_ok()
        assert isinstance(self.m_weights, Weights)
        self.m_weights.assert_ok()

    def pretty_string(self) -> str:
        return self.pretty_strings(self.modnames()).concatenate_fancy('', '\n', '')

    def pretty_strings(self, ms: Modnames) -> arr.Strings:
        intro = f'p({ms.output_colname().string()}|x) ~ Normal(mu = w^T x)'
        return self.weights().pretty_strings_with_introduction(ms.noomnames(), intro)

    def weights(self) -> Weights:
        return self.m_weights

    def modnames(self) -> Modnames:
        return self.m_modnames

    def predict(self, row:dat.row)->dis.Gaussian:
        x = self.transformers()

    def transformers(self)->noo.Transformers:
        return self.modnames().transformers()


class Model:
    def __init__(self, le: Learner, data):
        self.m_learner = le
        self.m_data = data
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_learner, Learner)
        self.m_learner.assert_ok()

        lt = self.learntype()
        if lt == Learntype.logistic:
            assert isinstance(self.m_data, ModelLogistic)
            self.m_data.assert_ok()
        elif lt == Learntype.linear:
            assert isinstance(self.m_data, ModelLinear)
            self.m_data.assert_ok()
        elif lt == Learntype.multinomial:
            bas.my_error("multinomial not implemented")
            # assert isinstance(self.m_data, ModelMultinomial)
            # self.m_data.assert_ok()
        else:
            bas.my_error('bad LearnType')

    def learntype(self) -> Learntype:
        return self.learner().learn_type()

    def learner(self) -> Learner:
        return self.m_learner

    def explain(self):
        print(f'Model = \n{self.pretty_string()}')

    def pretty_string(self) -> str:
        lt = self.learntype()
        if lt == Learntype.linear:
            return self.linear_model().pretty_string()
        elif lt == Learntype.logistic:
            return self.logistic_model().pretty_string()
        elif lt == Learntype.multinomial:
            bas.my_error("multinomial not implemented")
        else:
            bas.my_error("bad learn type")

    def linear_model(self) -> ModelLinear:
        assert self.learntype() == Learntype.linear
        assert isinstance(self.m_data, ModelLinear)
        return self.m_data

    def logistic_model(self) -> ModelLogistic:
        assert self.learntype() == Learntype.logistic
        assert isinstance(self.m_data, ModelLogistic)
        return self.m_data

    def predict(self, row:dat.Row)->dis.Distribution:
        lt = self.learntype()
        if lt == Learntype.linear:
            return dis.distribution_from_gaussian(self.linear_model().predict(row))
        elif lt == Learntype.logistic:
            return dis.distribution_from_binomial(self.logistic_model().predict(row))
        else:
            bas.my_error('bad learntype')


# Pr(Yi=1|x1i,...xMi) = 1/(1+b^{-beta.z}) where the sigmoid function has base b (for us b = e)
#
# ll = sum_k yk log(p(xk)) + sum_k (1-yk)log(1-p(xk))
#

def model_create(le: Learner, data) -> Model:
    return Model(le, data)


def learner_type_linear() -> Learner:
    return learner_create(Learntype.linear)


def model_type_linear(ml: ModelLinear) -> Model:
    return model_create(learner_type_linear(), ml)


def linear_model_create(ms: Modnames, ws: Weights) -> ModelLinear:
    return ModelLinear(ms, ws)


def weights_create(fs: arr.Floats) -> Weights:
    return Weights(fs)


def weights_zero(n_weights: int) -> Weights:
    return weights_create(arr.floats_all_zero(n_weights))


def train_glm(inputs: arr.Fmat, output: dat.Column, lt: Learntype) -> Weights:
    ws = weights_zero(inputs.num_cols())
    start_ll = ws.loglike(inputs, output, lt)
    ll = start_ll

    while True:
        for col in range(0, inputs.num_cols()):
            bbb = ws.loglike_derivative(inputs, output, lt, col)
            ccc = ws.loglike_second_derivative(inputs, output, lt, col)
            # ll = aaa + bbb * h + 0.5 * ccc * h^2
            # h_star = -bbb/ccc
            assert math.fabs(ccc) > 1e-20
            h_star = -bbb / ccc
            ws.increment(col, h_star)
        ll_new = ws.loglike(inputs, output, lt)
        assert bas.loosely_equals(ll, ll_new) or ll_new > ll

        if math.fabs(ll_new - ll) / (math.fabs(ll_new - start_ll)) < 1e-6:
            return ws


def modnames_create(nns: noo.Noomnames, output: dat.Colname) -> Modnames:
    return Modnames(nns, output)


def train_linear_model(inputs: dat.Datset, output: dat.NamedColumn) -> ModelLinear:
    assert output.coltype() == dat.Coltype.float
    ns = noo.noomset_from_datset(inputs)
    mns = modnames_create(ns.noomnames(), output.colname())
    ws = train_glm(ns.fmat(), output.column(), Learntype.linear)
    return linear_model_create(mns, ws)


def train_linear(inputs: dat.Datset, output: dat.NamedColumn) -> Model:
    return model_type_linear(train_linear_model(inputs, output))


def logistic_model_create(mns: Modnames, ws: Weights) -> ModelLogistic:
    return ModelLogistic(mns, ws)


def train_logistic_model(inputs: dat.Datset, output: dat.NamedColumn) -> ModelLogistic:
    assert output.coltype() == dat.Coltype.bool
    ns = noo.noomset_from_datset(inputs)
    mns = modnames_create(ns.noomnames(), output.colname())
    return logistic_model_create(mns, train_glm(ns.fmat(), output.column(), Learntype.logistic))


def model_type_logistic(ml: ModelLogistic) -> Model:
    return model_create(learner_type_logistic(), ml)


def train_logistic(inputs: dat.Datset, output: dat.NamedColumn) -> Model:
    return model_type_logistic(train_logistic_model(inputs, output))
