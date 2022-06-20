from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterator

import datset.amarrays as arr
import datset.ambasic as bas
import datset.distribution as dis
import datset.dset as dat
import datset.numset as noo


class Learner(ABC):
    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def train_from_named_column(self, inputs: dat.Datset, output: dat.NamedColumn) -> Model:
        pass

    @abstractmethod
    def name_as_string(self) -> str:
        pass

    def train(self, inputs: dat.Datset, output: dat.Datset) -> Model:
        assert output.num_cols() == 1
        assert output.num_rows() == inputs.num_rows()
        return self.train_from_named_column(inputs, output.named_column(0))

    def loglike(self, beta_xs: arr.Floats, output: dat.Column) -> float:
        result = 0.0
        for (bx, a) in zip(beta_xs.range(), output.range()):
            result += self.loglike_of_element(bx, a)

        return result

    @abstractmethod
    def loglike_derivative_from_row(self, ws: Weights, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        pass

    @abstractmethod
    def loglike_of_element(self, beta_x: float, y: dat.Atom) -> float:
        pass

    @abstractmethod
    def loglike_second_derivative_from_row(self, ws: Weights, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        pass


class LearnerLinear(Learner):
    def assert_ok(self):
        assert isinstance(self.m_name, str)

    def train_from_named_column(self, inputs: dat.Datset, output: dat.NamedColumn) -> Model:
        return train_linear_model(inputs, output)

    def name_as_string(self) -> str:
        return self.m_name

    def loglike_second_derivative_from_row(self, ws: Weights, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        x_kj = x_k.float(j)
        return -x_kj * x_kj

    def __init__(self):
        self.m_name = 'linear'

    def loglike_of_element(self, beta_x: float, y: dat.Atom) -> float:
        delta = y.float() - beta_x
        return -0.5 * delta * delta - bas.log_root_two_pi

    def loglike_derivative_from_row(self, ws: Weights, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        return x_k.float(j) * (y_k.float() - ws.times(x_k))


class LearnerLogistic(Learner):
    def assert_ok(self):
        assert isinstance(self.m_name, str)

    def train_from_named_column(self, inputs: dat.Datset, output: dat.NamedColumn) -> ModelLogistic:
        return train_logistic_model(inputs, output)

    def name_as_string(self) -> str:
        return self.m_name

    def loglike_second_derivative_from_row(self, ws: Weights, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        qk = q_from_y(y_k)  # note the qks get multiplied out so not really needed
        beta_x = ws.times(x_k)
        q_beta_x = qk * beta_x
        ek_of_beta = math.exp(-q_beta_x)
        ek_of_minus_beta = 1 / ek_of_beta
        fk_of_beta = 1 + ek_of_beta
        fk_of_minus_beta = 1 + ek_of_minus_beta
        x_kj = x_k.float(j)
        return -x_kj * x_kj / (fk_of_beta * fk_of_minus_beta)

    def __init__(self):
        self.m_name = 'logistic'

    def loglike_of_element(self, beta_x: float, y: dat.Atom) -> float:
        q = q_from_y(y)
        q_beta_x = q * beta_x
        ek_of_minus_beta = math.exp(q_beta_x)  # negative of negative is positive
        fk_of_minus_beta = 1 + ek_of_minus_beta
        return -math.log(fk_of_minus_beta)

    def loglike_derivative_from_row(self, ws: Weights, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        q_k = q_from_y(y_k)
        beta_x = ws.times(x_k)
        q_beta_x = q_k * beta_x
        ek_of_beta = math.exp(-q_beta_x)
        fk_of_beta = 1 + ek_of_beta
        return -q_k * x_k.float(j) / fk_of_beta


class LearnerMultinomial(Learner):
    def assert_ok(self):
        assert isinstance(self.m_name, str)

    def train_from_named_column(self, inputs: dat.Datset, output: dat.NamedColumn) -> Model:
        return train_multinomial_model(inputs, output)

    def name_as_string(self) -> str:
        return self.m_name

    def loglike_derivative_from_row(self, ws: Weights, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        bas.my_error("not needed")
        return -7e77

    def loglike_of_element(self, beta_x: float, y: dat.Atom) -> float:
        bas.my_error("not needed")
        return -7e77

    def loglike_second_derivative_from_row(self, ws: Weights, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        bas.my_error("not needed")
        return -7e77

    def __init__(self):
        self.m_name = 'multinomial'


class Model(ABC):

    @abstractmethod
    def assert_ok(self):
        pass

    def explain(self):
        print(self.pretty_string())

    @abstractmethod
    def pretty_string(self) -> str:
        pass

    @abstractmethod
    def predict_from_row(self, row: dat.Row) -> dis.Distribution:
        pass

    def transformers(self) -> noo.Transformers:
        return self.modnames().input_transformers()

    def batch_predict(self, inputs: dat.Datset) -> dat.Datset:
        cns = self.prediction_colnames()
        rfm = arr.row_indexed_fmat_with_no_rows(cns.len())
        for r in inputs.range_rows():
            di = self.predict_from_row(r)
            rfm.add_row(di.as_floats())
        fm = arr.fmat_create(rfm)
        return dat.datset_from_fmat(cns, fm)

    def prediction_colnames(self) -> dat.Colnames:
        return dat.colnames_from_strings(self.prediction_component_strings())

    @abstractmethod
    def prediction_component_strings(self) -> arr.Strings:
        pass

    @abstractmethod
    def modnames(self) -> Modnames:
        pass


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

    def loglike(self, le: Learner, inputs: noo.Termvecs, output: dat.Column) -> float:
        beta_xs = self.premultiplied_by(inputs)
        return le.loglike(beta_xs, output) - self.penalty()

    def loglike_derivative(self, le: Learner, inputs: noo.Termvecs, output: dat.Column, col: int) -> float:
        result = 0.0
        for x_k, y_k in zip(inputs.range(), output.range()):
            result += self.loglike_derivative_from_row(le, x_k, y_k, col)
        return result - self.penalty_derivative(col)

    def loglike_second_derivative(self, le: Learner, inputs: noo.Termvecs, output: dat.Column, col: int) -> float:
        result = 0.0
        for x_k, y_k in zip(inputs.range(), output.range()):
            result += self.loglike_second_derivative_from_row(le, x_k, y_k, col)
        return result - ws_penalty_second_derivative()

    def increment(self, col: int, delta: float):
        self.m_weights.increment(col, delta)

    def floats(self) -> arr.Floats:
        return self.m_weights

    def loglike_derivative_from_row(self, le: Learner, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        return le.loglike_derivative_from_row(self, x_k, y_k, j)

    def loglike_second_derivative_from_row(self, le: Learner, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        return le.loglike_second_derivative_from_row(self, x_k, y_k, j)

    def times(self, x_k: noo.Termvec) -> float:
        return self.floats().dot_product(x_k.floats())

    def pretty_strings_with_introduction(self, tfs: noo.Transformers, intro: str) -> arr.Strings:
        result = arr.strings_singleton(intro)
        result.add('')
        result.add('where...')
        result.add('')
        result.append(self.pretty_strings(tfs))
        return result

    def pretty_string_with_introduction(self, tfs: noo.Transformers, intro: str) -> str:
        return self.pretty_strings_with_introduction(tfs, intro).concatenate_fancy('', '\n', '')

    def num_cols_in_z_matrix(self) -> int:
        return self.floats().len()

    def pretty_strings(self, tfs: noo.Transformers) -> arr.Strings:
        return self.strings_array(tfs).pretty_strings()

    def strings_array(self, tfs: noo.Transformers) -> arr.StringsArray:
        nns = tfs.noomnames()
        assert self.num_cols_in_z_matrix() == nns.len() + 1
        result = arr.strings_array_empty()
        cws = self.correct_accounting_for_transformers(tfs)
        nns_as_strings = arr.strings_singleton('constant').with_many(nns.strings())
        for nn_as_string, w in zip(nns_as_strings.range(), cws.range()):
            ss = arr.strings_empty()
            ss.add(f'w[{nn_as_string}]')
            ss.add('=')
            ss.add(f'{w}')
            result.add(ss)
        return result

    def weight(self, i: int) -> float:
        return self.floats().float(i)

    def premultiplied_by(self, tvs: noo.Termvecs) -> arr.Floats:
        return tvs.times(self.floats())

    def deep_copy(self):  # returns WeightsArray
        fs = self.floats().deep_copy()
        assert isinstance(fs, arr.Floats)
        return weights_create(fs)

    def penalty(self) -> float:
        return penalty_parameter * self.sum_squares()

    def penalty_derivative(self, col: int) -> float:
        return penalty_parameter * 2 * self.weight(col)

    def sum_squares(self) -> float:
        return self.floats().sum_squares()

    def range(self) -> Iterator[float]:
        return self.m_weights.range()

    def correct_accounting_for_transformers(self, tfs: noo.Transformers) -> Weights:
        # predict = c_old + sum_j wold_j * (x_j - lo_j)/width_j
        #
        # predict = c_new + sum_j w_new_j x_j
        #
        # c_new = c_old + sum_j wold_j (-lo_j) / width_j
        # w_new_j = wold_j/width_j
        c_old = self.weight(0)
        c_new = c_old
        j_to_lo = arr.floats_empty()
        j_to_width = arr.floats_empty()

        for tf in tfs.range():
            for iv in tf.scaling_intervals().range():
                j_to_lo.add(iv.lo())
                j_to_width.add(iv.width())

        j_to_wold = self.floats().without_leftmost_element()
        j_to_w_new = arr.floats_empty()

        for wold, lo, width in zip(j_to_wold.range(), j_to_lo.range(), j_to_width.range()):
            j_to_w_new.add(wold / width)
            c_new -= wold * lo / width

        result = arr.floats_singleton(c_new)
        result.append(j_to_w_new)

        assert result.len() == self.num_cols_in_z_matrix()

        return weights_create(result)


class Modnames:
    def __init__(self, tfs: noo.Transformers, output: dat.Coldescribe):
        self.m_input_transformers = tfs
        self.m_output_describe = output
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_input_transformers, noo.Transformers)
        self.m_input_transformers.assert_ok()
        assert isinstance(self.m_output_describe, dat.Coldescribe)
        self.m_output_describe.assert_ok()

    def input_transformers(self) -> noo.Transformers:
        return self.m_input_transformers

    def input_noomnames(self) -> noo.Noomnames:
        return self.input_transformers().noomnames()

    def output_describe(self) -> dat.Coldescribe:
        return self.m_output_describe

    def termvec_from_row(self, row: dat.Row) -> noo.Termvec:
        return self.input_transformers().termvec_from_row(row)


class ModelLogistic(Model):
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
        intro = f'P({ms.output_describe().colname().string()}|x) = sigmoid(w^T x)'
        return self.weights().pretty_string_with_introduction(ms.input_transformers(), intro)

    def weights(self) -> Weights:
        return self.m_weights

    def modnames(self) -> Modnames:
        return self.m_modnames

    def predict_from_termvec(self, tv: noo.Termvec) -> dis.Binomial:
        ez = math.exp(self.weights().times(tv))
        p = ez / (1 + ez)
        return dis.binomial_create(p)

    def predict_from_row(self, row: dat.Row) -> dis.Binomial:
        return self.predict_from_termvec(self.transformers().termvec_from_row(row))

    def prediction_component_strings(self) -> arr.Strings:
        return arr.strings_singleton(f'prob_{self.modnames().output_describe().colname().string()}')


class ModelLinear(Model):
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
        intro = f'p({ms.output_describe().colname().string()}|x) ~ Normal(mu = w^T x)'
        return self.weights().pretty_strings_with_introduction(ms.input_transformers(), intro)

    def weights(self) -> Weights:
        return self.m_weights

    def modnames(self) -> Modnames:
        return self.m_modnames

    def predict_from_termvec(self, tv: noo.Termvec) -> dis.Gaussian:
        assert self.weights().num_cols_in_z_matrix() == tv.num_terms()
        mu = self.weights().times(tv)
        return dis.gaussian_create(mu, self.sdev())

    def predict_from_row(self, row: dat.Row) -> dis.Gaussian:
        return self.predict_from_termvec(self.transformers().termvec_from_row(row))

    def sdev(self) -> float:
        return self.m_sdev

    def prediction_component_strings(self):
        result = arr.strings_singleton(self.modnames().output_describe().colname().string())
        result.add('sdev')
        return result


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
            assert self.m_num_cols == ws.num_cols_in_z_matrix()

        if len(self.m_value_to_weights) > 0:
            self.m_value_to_weights[0].equals(weights_zero(self.m_num_cols))

    def loglike_k(self, x: noo.Termvecs, y: arr.Ints, k: int) -> float:
        i = y.int(k)
        z = x.termvec(k)
        qki = self.qki(z, i)
        sk = self.sk(z)
        return qki - math.log(sk)

    def weights(self, val: int) -> Weights:
        assert 0 <= val < self.num_values()
        return self.m_value_to_weights[val]

    def loglike(self, x: noo.Termvecs, y: arr.Ints) -> float:
        result = 0.0
        for k in range(x.num_rows()):
            result += self.loglike_k(x, y, k)
        return result - self.penalty()

    def pretty_string(self, ms: Modnames, vns: dat.Valnames) -> str:
        return self.pretty_strings(ms, vns).concatenate_fancy('', '\n', '')

    def pretty_strings(self, ms: Modnames, vns: dat.Valnames) -> arr.Strings:
        headers = arr.strings_singleton('constant').with_many(ms.input_noomnames().strings())
        return self.fmat().pretty_strings_with_headings("", vns.strings(), headers)

    def row_indexed_fmat(self) -> arr.RowIndexedFmat:
        result = arr.row_indexed_fmat_with_no_rows(self.num_elements_in_each_weights_vector())
        for ws in self.range():
            result.add_row(ws.floats())
        return result

    def fmat(self) -> arr.Fmat:
        return arr.fmat_create(self.row_indexed_fmat())

    def num_values(self) -> int:
        return len(self.m_value_to_weights)

    def predict(self, z: noo.Termvec) -> dis.Multinomial:
        sk = self.sk(z)
        probs = arr.floats_empty()
        for i in range(self.num_values()):
            eki = self.eki(z, i)
            probs.add(eki / sk)
        return dis.multinomial_create(probs)

    def eki(self, z: noo.Termvec, i: int) -> float:
        return math.exp(self.qki(z, i))

    def qki(self, tv: noo.Termvec, i: int) -> float:
        return self.weights(i).times(tv)

    def increment(self, val: int, weight_index: int, delta: float):
        assert val > 0  # top row of weights must all be zero
        self.weights(val).increment(weight_index, delta)

    def sk(self, z: noo.Termvec) -> float:
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
        assert ws.num_cols_in_z_matrix() == self.num_elements_in_each_weights_vector()
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
        for ws in self.m_value_to_weights:
            yield ws


def weights_array_empty(n_cols: int) -> WeightsArray:
    return WeightsArray(n_cols)


def weights_array_zero(n_values: int, n_cols: int) -> WeightsArray:
    assert n_values > 1
    assert n_cols >= 1
    result = weights_array_empty(n_cols)
    for i in range(n_values):
        result.add_row_of_zero_weights()
    return result


def multinomial_train_weights(x: noo.Termvecs, y: arr.Ints) -> WeightsArray:
    assert x.num_rows() == y.len()
    n_values = y.max() + 1
    n_cols = x.num_terms()
    wsa = weights_array_zero(n_values, n_cols)
    start_ll = wsa.loglike(x, y)
    ll = start_ll

    iteration = 0
    while True:
        if iteration == 0 or bas.is_power_of_two(iteration):
            print(f'Begin iteration {iteration}')
        ll_old = ll
        for j in range(x.num_terms()):
            for i in range(1, n_values):  # math would've redundancy which is solved by zeroing first row of weights
                aaa = 0.0
                bbb = 0.0
                ccc = 0.0
                for k in range(x.num_rows()):
                    # pik denotes P(class = i | x_k, Weights)
                    # eik = exp(Weights[i] . x_k)
                    u_to_euk = arr.floats_empty()
                    for ws in wsa.range():
                        quk = ws.times(x.termvec(k))
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

                # ll = aaa + bbb * h + 0.5 * ccc * h^2
                # h_star = -bbb/ccc
                assert math.fabs(ccc) > 1e-20
                h_star = -bbb / ccc
                wsa2 = wsa.deep_copy()
                assert isinstance(wsa2, WeightsArray)
                wsa2.increment(i, j, h_star)
                ll_new2 = wsa2.loglike(x, y)
                if ll_new2 > ll:
                    wsa = wsa2
                    ll = ll_new2

        if math.fabs(ll - ll_old) / (1e-5 + math.fabs(ll - start_ll)) < 1e-4:
            print(f'...finished after {iteration} iterations.')
            return wsa

        iteration += 1


class ModelMultinomial(Model):
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
        assert n_elements == self.m_modnames.input_transformers().noomnames().len() + 1

    def pretty_strings(self) -> arr.Strings:
        result = arr.strings_empty()
        ms = self.modnames()
        result.add(f'P({ms.output_describe().colname().string()}=v|Weights,x) = exp(Weights[v] . x) / K where...')
        result.append(self.weights_array().pretty_strings(self.modnames(), self.valnames()))
        return result

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def weights(self, v: int) -> Weights:
        return self.weights_array().weights(v)

    def modnames(self) -> Modnames:
        return self.m_modnames

    def predict_from_termvec(self, z: noo.Termvec) -> dis.Multinomial:
        return self.weights_array().predict(z)

    def predict_from_row(self, row: dat.Row) -> dis.Multinomial:
        return self.predict_from_termvec(self.termvec_from_row(row))

    def num_values(self) -> int:
        return self.valnames().len()

    def valnames(self) -> dat.Valnames:
        return self.m_valnames

    def valname(self, val: int) -> dat.Valname:
        return self.valnames().valname(val)

    def weights_array(self) -> WeightsArray:
        return self.m_weights_array

    def prediction_component_strings(self) -> arr.Strings:
        result = arr.strings_empty()
        for vn in self.valnames().range():
            result.add(f'prob_{vn.string()}')
        return result

    def termvec_from_row(self, row: dat.Row) -> noo.Termvec:
        return self.transformers().termvec_from_row(row)


def multinomial_model_create(ms: Modnames, vns: dat.Valnames, b: WeightsArray) -> ModelMultinomial:
    return ModelMultinomial(ms, vns, b)


def train_multinomial_model(inputs: dat.Datset, output: dat.NamedColumn) -> ModelMultinomial:
    tfs = noo.transformers_from_datset(inputs)
    x = tfs.transform_datset(inputs)
    assert isinstance(output.column(), dat.ColumnCats)
    cs = output.cats()
    y = cs.values()
    ms = modnames_create(tfs, output.coldescribe())
    b = multinomial_train_weights(x.termvecs(), y)
    return multinomial_model_create(ms, cs.valnames(), b)


# Pr(Yi=1|x1i,...xMi) = 1/(1+b^{-beta.z}) where the sigmoid function has base b (for us b = e)
#
# ll = sum_k yk log(p(xk)) + sum_k (1-yk)log(1-p(xk))
#


def learner_type_linear() -> LearnerLinear:
    return LearnerLinear()


def linear_model_create(ms: Modnames, ws: Weights, sdev: float) -> ModelLinear:
    return ModelLinear(ms, ws, sdev)


def weights_create(fs: arr.Floats) -> Weights:
    return Weights(fs)


def weights_zero(n_weights: int) -> Weights:
    return weights_create(arr.floats_all_zero(n_weights))


def train_glm(le: Learner, inputs: noo.Termvecs, output: dat.Column) -> Weights:
    ws = weights_zero(inputs.num_terms())
    start_ll = ws.loglike(le, inputs, output)
    ll = start_ll

    iteration = 0
    while True:
        if iteration == 0 or bas.is_power_of_two(iteration):
            print(f'Begin iteration {iteration}')
        ll_old = ll
        for col in range(inputs.num_terms()):
            aaa = ws.loglike(le, inputs, output)
            assert bas.loosely_equals(aaa, ll)
            bbb = ws.loglike_derivative(le, inputs, output, col)
            ccc = ws.loglike_second_derivative(le, inputs, output, col)

            # ll = aaa + bbb * h + 0.5 * ccc * h^2
            # h_star = -bbb/ccc
            assert math.fabs(ccc) > 1e-20
            h_star = -bbb / ccc
            ws2 = ws.deep_copy()
            assert isinstance(ws2, Weights)
            ws2.increment(col, h_star)
            ll_new2 = ws2.loglike(le, inputs, output)
            if ll_new2 > ll:
                ws = ws2
                ll = ll_new2

        if math.fabs(ll - ll_old) / (math.fabs(ll - start_ll)) < 1e-5:
            print(f'...finished after {iteration} iterations.')
            return ws

        iteration += 1


def modnames_create(tfs: noo.Transformers, output: dat.Coldescribe) -> Modnames:
    return Modnames(tfs, output)


def train_linear_model(inputs: dat.Datset, output: dat.NamedColumn) -> ModelLinear:
    tfs = noo.transformers_from_datset(inputs)
    ns = tfs.transform_datset(inputs)
    mns = modnames_create(tfs, output.coldescribe())
    ws = train_glm(learner_type_linear(), ns.termvecs(), output.column())
    predictions = ws.premultiplied_by(ns.termvecs())
    errors = predictions.minus(output.floats())
    assert isinstance(errors, arr.Floats)
    sse = errors.squared()
    denominator = ns.num_rows() - ns.num_cols()
    if denominator < 1:
        denominator = 1
    mse = sse / denominator
    root_mse = math.sqrt(mse)
    return linear_model_create(mns, ws, root_mse)


def logistic_model_create(mns: Modnames, ws: Weights) -> ModelLogistic:
    return ModelLogistic(mns, ws)


def train_logistic_model(inputs: dat.Datset, output: dat.NamedColumn) -> ModelLogistic:
    assert isinstance(output.column(), dat.ColumnBool)
    tfs = noo.transformers_from_datset(inputs)
    ns = tfs.transform_datset(inputs)
    mns = modnames_create(tfs, output.coldescribe())
    return logistic_model_create(mns, train_glm(learner_type_logistic(), ns.termvecs(), output.column()))


def learner_type_multinomial() -> LearnerMultinomial:
    return LearnerMultinomial()


def learner_type_logistic() -> LearnerLogistic:
    return LearnerLogistic()
