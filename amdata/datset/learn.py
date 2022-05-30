import enum
import math

from collections.abc import Iterator

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
        for (bx, a) in zip(beta_xs.range(), output.range()):
            result += self.loglike_of_element(bx, a)

        return result

    def loglike_of_element(self, beta_x: float, y: dat.Atom) -> float:
        if self == Learntype.linear:
            delta = y.float() - beta_x
            return -0.5 * delta * delta - bas.log_root_two_pi
        elif self == Learntype.logistic:
            q = q_from_y(y)
            q_beta_x = q * beta_x
            ek_of_minus_beta = math.exp(q_beta_x)  # negative of negative is positive
            fk_of_minus_beta = 1 + ek_of_minus_beta
            return -math.log(fk_of_minus_beta)
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
            return train_multinomial(inputs, output)

        else:
            bas.my_error("bad LearnType")


def learner_create(lt: Learntype) -> Learner:
    return Learner(lt)


def q_from_y(y_k: dat.Atom) -> float:
    assert isinstance(y_k, dat.AtomBool)
    if y_k.bool():
        return -1.0
    else:
        return 1.0


penalty_parameter: float = 0.0001


def ws_penalty_second_derivative() -> float:
    return penalty_parameter * 2


class Weights:
    def __init__(self, fs: arr.Floats):
        self.m_weights = fs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_weights, arr.Floats)
        self.m_weights.assert_ok()

    def loglike(self, inputs: arr.Fmat, output: dat.Column, lt: Learntype) -> float:
        beta_xs = self.premultiplied_by(inputs)
        return lt.loglike(beta_xs, output) - self.penalty()

    def loglike_derivative(self, inputs: arr.Fmat, output: dat.Column, lt: Learntype, col: int) -> float:
        result = 0.0
        for x_k, y_k in zip(inputs.range_rows(), output.range()):
            result += self.loglike_derivative_from_row(x_k, y_k, lt, col)
        return result - self.penalty_derivative(col)

    def loglike_second_derivative(self, inputs: arr.Fmat, output: dat.Column, lt: Learntype, col: int) -> float:
        result = 0.0
        for x_k, y_k in zip(inputs.range_rows(), output.range()):
            result += self.loglike_second_derivative_from_row(x_k, y_k, lt, col)
        return result - ws_penalty_second_derivative()

    def increment(self, col: int, delta: float):
        self.m_weights.increment(col, delta)

    def floats(self) -> arr.Floats:
        return self.m_weights

    def loglike_derivative_from_row(self, x_k: arr.Floats, y_k: dat.Atom, lt: Learntype, j: int) -> float:
        if lt == Learntype.linear:
            return x_k.float(j) * (y_k.float() - self.times(x_k))
        elif lt == Learntype.logistic:
            q_k = q_from_y(y_k)
            beta_x = self.times(x_k)
            q_beta_x = q_k * beta_x
            ek_of_beta = math.exp(-q_beta_x)
            fk_of_beta = 1 + ek_of_beta
            return -q_k * x_k.float(j) / fk_of_beta
        else:
            bas.my_error("bad LearnType for loglike")

    def loglike_second_derivative_from_row(self, x_k: arr.Floats, y_k: dat.Atom, lt: Learntype, j: int) -> float:
        if lt == Learntype.linear:
            x_kj = x_k.float(j)
            return -x_kj * x_kj
        elif lt == Learntype.logistic:
            qk = q_from_y(y_k)  # note the qks get multiplied out so not really needed
            beta_x = self.times(x_k)
            q_beta_x = qk * beta_x
            ek_of_beta = math.exp(-q_beta_x)
            ek_of_minus_beta = 1 / ek_of_beta
            fk_of_beta = 1 + ek_of_beta
            fk_of_minus_beta = 1 + ek_of_minus_beta
            x_kj = x_k.float(j)
            return -x_kj * x_kj / (fk_of_beta * fk_of_minus_beta)
        else:
            bas.my_error("bad learn type for log like")

    def times(self, x_k: arr.Floats) -> float:
        return self.floats().dot_product(x_k)

    def pretty_strings_with_introduction(self, nns: noo.Noomnames, intro: str) -> arr.Strings:
        assert self.num_cols_in_x_matrix() == nns.len()+1
        result = arr.strings_singleton(intro)
        result.add('')
        result.add('where...')
        result.add('')
        result.append(self.pretty_strings(nns))
        return result

    def pretty_string_with_introduction(self, nns: noo.Noomnames, intro: str) -> str:
        return self.pretty_strings_with_introduction(nns, intro).concatenate_fancy('', '\n', '')

    def num_cols_in_x_matrix(self) -> int:
        return self.floats().len()

    def pretty_strings(self, nns: noo.Noomnames) -> arr.Strings:
        return self.strings_array(nns).pretty_strings()

    def strings_array(self, nns: noo.Noomnames) -> arr.StringsArray:
        assert self.num_cols_in_x_matrix() == nns.len()+1
        result = arr.strings_array_empty()
        nns_as_strings = arr.strings_singleton('constant').with_many(nns.strings())
        for nn_as_string, w in zip(nns_as_strings.range(), self.range()):
            ss = arr.strings_empty()
            ss.add(f'w[{nn_as_string}]')
            ss.add('=')
            ss.add(f'{w}')
            result.add(ss)
        return result

    def weight(self, i: int) -> float:
        return self.floats().float(i)

    def premultiplied_by(self, fm: arr.Fmat) -> arr.Floats:
        return fm.times(self.floats())

    def deep_copy(self):  # returns WeightsArray
        fs = self.floats().deep_copy()
        assert isinstance(fs, arr.Floats)
        return weights_create(fs)

    def penalty(self) -> float:
        return penalty_parameter * self.sum_squares()

    def penalty_derivative(self, col: int) -> float:
        return penalty_parameter * 2 * self.weight(col)

    def sum_squares(self) -> float:
        return self.times(self.floats())

    def range(self) -> Iterator[float]:
        return self.m_weights.range()


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

    def predict(self, row: dat.Row) -> dis.Binomial:
        tfs = self.transformers()
        z = tfs.transform_row(row)
        ez = math.exp(self.weights().times(z))
        p = ez / (1 + ez)
        return dis.binomial_create(p)

    def transformers(self) -> noo.Transformers:
        return self.modnames().transformers()


class ModelLinear:
    def __init__(self, mns: Modnames, ws: Weights, sdev: float):
        self.m_modnames = mns
        self.m_weights = ws
        self.m_sdev = sdev
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_modnames, Modnames)
        self.m_modnames.assert_ok()
        assert isinstance(self.m_weights, Weights)
        self.m_weights.assert_ok()
        assert isinstance(self.m_sdev, float)
        assert self.m_sdev > 0

    def pretty_string(self) -> str:
        return self.pretty_strings(self.modnames()).concatenate_fancy('', '\n', '')

    def pretty_strings(self, ms: Modnames) -> arr.Strings:
        intro = f'p({ms.output_colname().string()}|x) ~ Normal(mu = w^T x)'
        return self.weights().pretty_strings_with_introduction(ms.noomnames(), intro)

    def weights(self) -> Weights:
        return self.m_weights

    def modnames(self) -> Modnames:
        return self.m_modnames

    def predict(self, row: dat.Row) -> dis.Gaussian:
        tfs = self.transformers()
        z = tfs.transform_row(row)
        mu = self.weights().times(z)
        return dis.gaussian_create(mu, self.sdev())

    def transformers(self) -> noo.Transformers:
        return self.modnames().transformers()

    def sdev(self) -> float:
        return self.m_sdev


def wsa_penalty_second_derivative() -> float:
    return penalty_parameter * 2.0


class WeightsArray:
    def __init__(self, n_cols: int):
        self.m_num_cols = n_cols
        self.m_value_to_weights = []
        self.assert_ok()

    def add_row_of_zero_weights(self):
        self.add_weights(weights_zero(self.num_elements_in_each_weights_vector()))

    def assert_ok(self):
        assert isinstance(self.m_num_cols, int)
        assert self.m_num_cols > 0
        assert isinstance(self.m_value_to_weights, list)
        for ws in self.m_value_to_weights:
            assert isinstance(ws, Weights)
            assert self.m_num_cols == ws.num_cols_in_x_matrix()

        if len(self.m_value_to_weights) > 0:
            self.m_value_to_weights[0].loosely_equals(weights_zero(self.m_num_cols))

    def loglike_k(self, x: arr.Fmat, y: arr.Ints, k: int) -> float:
        i = y.int(k)
        z = x.row(k)
        qki = self.qki(z, i)
        sk = self.sk(z)
        return qki - math.log(sk)

    def weights(self, val: int) -> Weights:
        assert 0 <= val < self.num_values()
        return self.m_value_to_weights[val]

    def loglike(self, x: arr.Fmat, y: arr.Ints) -> float:
        result = 0.0
        for k in range(x.num_rows()):
            result += self.loglike_k(x, y, k)
        return result - self.penalty()

    def pretty_string(self, ms: Modnames, vns: dat.Valnames) -> str:
        return self.pretty_strings(ms, vns).concatenate_fancy('', '\n', '')

    def pretty_strings(self, ms: Modnames, vns: dat.Valnames) -> arr.Strings:
        return self.fmat().pretty_strings_with_headings("", vns.strings(), ms.noomnames().strings())

    def row_indexed_fmat(self) -> arr.RowIndexedFmat:
        result = arr.row_indexed_fmat_with_no_rows(self.num_elements_in_each_weights_vector())
        for ws in self.range():
            result.add_row(ws.floats())
        return result

    def fmat(self) -> arr.Fmat:
        return arr.fmat_create(self.row_indexed_fmat())

    def num_values(self) -> int:
        return len(self.m_value_to_weights)

    def predict(self, z: arr.Floats) -> dis.Multinomial:
        sk = self.sk(z)
        probs = arr.floats_empty()
        for i in range(self.num_values()):
            eki = self.eki(z, i)
            probs.add(eki / sk)
        return dis.multinomial_create(probs)

    def eki(self, z: arr.Floats, i: int) -> float:
        return math.exp(self.qki(z, i))

    def qki(self, z: arr.Floats, i: int) -> float:
        return self.weights(i).times(z)

    def increment(self, val: int, weight_index: int, delta: float):
        assert val > 0  # top row of weights must all be zero
        self.weights(val).increment(weight_index, delta)

    def sk(self, z: arr.Floats) -> float:
        result = 0
        for i in range(self.num_values()):
            result += self.eki(z, i)
        return result

    def deep_copy(self):  # returns WeightsArray
        result = weights_array_empty(self.num_elements_in_each_weights_vector())
        for ws in self.range():
            result.add_weights(ws.deep_copy())
        return result

    def add_weights(self, ws: Weights):
        assert ws.num_cols_in_x_matrix() == self.num_elements_in_each_weights_vector()
        self.m_value_to_weights.append(ws)

    def num_elements_in_each_weights_vector(self) -> int:
        return self.m_num_cols

    def penalty(self) -> float:
        return penalty_parameter * self.sum_squares()

    def penalty_derivative(self, i, j):
        return penalty_parameter * 2 * self.weights(i).weight(j)

    def sum_squares(self) -> float:
        result = 0.0
        for ws in self.range():
            result += ws.sum_squares()
        return result

    def range(self) -> Iterator[Weights]:
        pass


def weights_array_empty(n_cols: int) -> WeightsArray:
    return WeightsArray(n_cols)


def weights_array_zero(n_values: int, n_cols: int) -> WeightsArray:
    assert n_values > 1
    assert n_cols >= 1
    result = weights_array_empty(n_cols)
    for i in range(n_values):
        result.add_row_of_zero_weights()
    return result


def multinomial_train_weights(x: arr.Fmat, y: arr.Ints, ms: Modnames, vns: dat.Valnames) -> WeightsArray:
    print(f'entering multinomial_train_weights. y = {y.pretty_string()}')
    assert x.num_rows() == y.len()
    n_values = y.max() + 1
    n_cols = x.num_cols()
    wsa = weights_array_zero(n_values, n_cols)
    start_ll = wsa.loglike(x, y)
    ll = start_ll

    iteration = 0
    while True:
        ll_old = ll
        for j in range(x.num_cols()):
            for i in range(1, n_values):  # math would've redundancy which is solved by zeroing first row of weights
                print(f'iteration {iteration}, j = {j}, i = {i}, wsa =\n{wsa.pretty_string(ms, vns)}')
                aaa = 0.0
                bbb = 0.0
                ccc = 0.0
                for k in range(x.num_rows()):
                    # pik denotes P(class = i | x_k, Weights)
                    # eik = exp(Weights[i] . x_k)
                    u_to_euk = arr.floats_empty()
                    for ws in wsa.range():
                        quk = ws.times(x.row(k))
                        euk = math.exp(quk)
                        u_to_euk.add(euk)

                    sk = u_to_euk.sum()
                    pik = u_to_euk.float(i) / sk
                    prob_yk_given_xk = u_to_euk.float(y.int(k)) / sk

                    ll_k = math.log(prob_yk_given_xk)
                    aaa += ll_k
                    xkj = x.float(k, j)
                    d_ll_k_by_dw_ij = -xkj * pik
                    if i == y.int(k):
                        d_ll_k_by_dw_ij += xkj

                    d2_ll_k_by_dw2_ij = -xkj * xkj * pik * (1 - pik)

                    bbb += d_ll_k_by_dw_ij
                    ccc += d2_ll_k_by_dw2_ij

                aaa -= wsa.penalty()
                bbb -= wsa.penalty_derivative(i, j)
                ccc -= wsa_penalty_second_derivative()

                assert bas.loosely_equals(aaa, ll)

                print(f'll(W+h*I_{i}) = {aaa} + {bbb} * h + {ccc} * h^2')
                # ll = aaa + bbb * h + 0.5 * ccc * h^2
                # h_star = -bbb/ccc
                assert math.fabs(ccc) > 1e-20
                h_star = -bbb / ccc
                print(f'h_star = {h_star}')
                wsa2 = wsa.deep_copy()
                assert isinstance(wsa2, WeightsArray)
                wsa2.increment(i, j, h_star)
                ll_new2 = wsa2.loglike(x, y)
                print(f'll_new2 = {ll_new2}')
                if ll_new2 > ll:
                    print('improved!')
                    wsa = wsa2
                    ll = ll_new2
                else:
                    print('no improvement')

        if math.fabs(ll - ll_old) / (1e-5 + math.fabs(ll - start_ll)) < 1e-3:
            return wsa

        iteration += 1


class ModelMultinomial:
    def __init__(self, ms: Modnames, vns: dat.Valnames, b: WeightsArray):
        self.m_modnames = ms
        self.m_valnames = vns
        self.m_weights_array = b
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_modnames, Modnames)
        self.m_modnames.assert_ok()
        assert isinstance(self.m_valnames, dat.Valnames)
        self.m_valnames.assert_ok()
        assert isinstance(self.m_weights_array, WeightsArray)
        assert self.m_weights_array.num_values() == self.m_valnames.len()
        n_elements = self.m_weights_array.num_elements_in_each_weights_vector()
        assert n_elements == self.m_modnames.transformers().noomnames().len()

    def pretty_strings(self) -> arr.Strings:
        result = arr.strings_empty()
        ms = self.modnames()
        result.add(f'P({ms.output_colname().string()}=v|Weights,x) = exp(Weights[v] . x) / K where...')
        result.append(self.weights_array().pretty_strings(self.modnames(), self.valnames()))
        return result

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def weights(self, v: int) -> Weights:
        return self.weights_array().weights(v)

    def modnames(self) -> Modnames:
        return self.m_modnames

    def predict(self, row: dat.Row) -> dis.Multinomial:
        tfs = self.transformers()
        z = tfs.transform_row(row)
        return self.weights_array().predict(z)

    def transformers(self) -> noo.Transformers:
        return self.modnames().transformers()

    def num_values(self) -> int:
        return self.valnames().len()

    def valnames(self) -> dat.Valnames:
        return self.m_valnames

    def valname(self, val: int) -> dat.Valname:
        return self.valnames().valname(val)

    def weights_array(self) -> WeightsArray:
        return self.m_weights_array


def multinomial_model_create(ms: Modnames, vns: dat.Valnames, b: WeightsArray) -> ModelMultinomial:
    return ModelMultinomial(ms, vns, b)


def train_multinomial_model(inputs: dat.Datset, output: dat.NamedColumn) -> ModelMultinomial:
    tfs = noo.transformers_from_datset(inputs)
    x = tfs.transform_datset(inputs)
    cs = output.cats()
    y = cs.values()
    ms = modnames_create(tfs, output.colname())
    b = multinomial_train_weights(x.fmat(), y, ms, output.valnames())
    return multinomial_model_create(ms, cs.valnames(), b)


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
            assert isinstance(self.m_data, ModelMultinomial)
            self.m_data.assert_ok()
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
            return self.multinomial_model().pretty_string()
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

    def predict(self, row: dat.Row) -> dis.Distribution:
        lt = self.learntype()
        if lt == Learntype.linear:
            return dis.distribution_from_gaussian(self.linear_model().predict(row))
        elif lt == Learntype.logistic:
            return dis.distribution_from_binomial(self.logistic_model().predict(row))
        elif lt == Learntype.multinomial:
            return dis.distribution_from_multinomial(self.multinomial_model().predict(row))
        else:
            bas.my_error('bad learntype')

    def multinomial_model(self) -> ModelMultinomial:
        assert self.learntype() == Learntype.multinomial
        assert isinstance(self.m_data, ModelMultinomial)
        return self.m_data


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


def linear_model_create(ms: Modnames, ws: Weights, sdev: float) -> ModelLinear:
    return ModelLinear(ms, ws, sdev)


def weights_create(fs: arr.Floats) -> Weights:
    return Weights(fs)


def weights_zero(n_weights: int) -> Weights:
    return weights_create(arr.floats_all_zero(n_weights))


def train_glm(inputs: arr.Fmat, output: dat.Column, lt: Learntype) -> Weights:
    ws = weights_zero(inputs.num_cols())
    start_ll = ws.loglike(inputs, output, lt)
    ll = start_ll

    iteration = 0
    while True:
        ll_old = ll
        for col in range(inputs.num_cols()):
            print(f'iter {iteration}, col {col}: ll = {ll}, ws = {ws.floats().pretty_string()}')
            aaa = ws.loglike(inputs, output, lt)
            assert bas.loosely_equals(aaa, ll)
            bbb = ws.loglike_derivative(inputs, output, lt, col)
            ccc = ws.loglike_second_derivative(inputs, output, lt, col)

            print(f'll(h) = {aaa} + {bbb} * h + {ccc} * h^2')
            # ll = aaa + bbb * h + 0.5 * ccc * h^2
            # h_star = -bbb/ccc
            assert math.fabs(ccc) > 1e-20
            h_star = -bbb / ccc
            print(f'h_star = {h_star}')
            ws2 = ws.deep_copy()
            assert isinstance(ws2, Weights)
            ws2.increment(col, h_star)
            ll_new2 = ws2.loglike(inputs, output, lt)
            print(f'll_new2 = {ll_new2}')
            if ll_new2 > ll:
                print('improved!')
                ws = ws2
                ll = ll_new2
            else:
                print('no improvement')

        if math.fabs(ll - ll_old) / (math.fabs(ll - start_ll)) < 1e-6:
            return ws

        iteration += 1


def modnames_create(tfs: noo.Transformers, output: dat.Colname) -> Modnames:
    return Modnames(tfs, output)


def train_linear_model(inputs: dat.Datset, output: dat.NamedColumn) -> ModelLinear:
    tfs = noo.transformers_from_datset(inputs)
    ns = tfs.transform_datset(inputs)
    mns = modnames_create(tfs, output.colname())
    ws = train_glm(ns.fmat(), output.column(), Learntype.linear)
    predictions = ws.premultiplied_by(ns.fmat())
    errors = predictions.minus(output.floats())
    assert isinstance(errors, arr.Floats)
    sse = errors.squared()
    denominator = ns.num_rows() - ns.num_cols()
    if denominator < 1:
        denominator = 1
    mse = sse / denominator
    root_mse = math.sqrt(mse)
    return linear_model_create(mns, ws, root_mse)


def train_linear(inputs: dat.Datset, output: dat.NamedColumn) -> Model:
    return model_type_linear(train_linear_model(inputs, output))


def logistic_model_create(mns: Modnames, ws: Weights) -> ModelLogistic:
    return ModelLogistic(mns, ws)


def train_logistic_model(inputs: dat.Datset, output: dat.NamedColumn) -> ModelLogistic:
    assert isinstance(output.column(), dat.ColumnBool)
    tfs = noo.transformers_from_datset(inputs)
    ns = tfs.transform_datset(inputs)
    mns = modnames_create(tfs, output.colname())
    return logistic_model_create(mns, train_glm(ns.fmat(), output.column(), Learntype.logistic))


def model_type_logistic(ml: ModelLogistic) -> Model:
    return model_create(learner_type_logistic(), ml)


def train_logistic(inputs: dat.Datset, output: dat.NamedColumn) -> Model:
    return model_type_logistic(train_logistic_model(inputs, output))


def model_type_multinomial(m: ModelMultinomial) -> Model:
    return Model(learner_type_multinomial(), m)


def train_multinomial(inputs: dat.Datset, output: dat.NamedColumn) -> Model:
    return model_type_multinomial(train_multinomial_model(inputs, output))


def learner_type_multinomial() -> Learner:
    return learner_create(Learntype.multinomial)


def learner_type_logistic() -> Learner:
    return learner_create(Learntype.logistic)
