from __future__ import annotations

import math
from abc import ABC, abstractmethod

from typing import Iterator, Tuple
import datset.amarrays as arr
import datset.ambasic as bas
import datset.distribution as dis
import datset.dset as dat
import datset.numset as noo
import datset.learn as lea
from datset.learn import Model, Learner


class LearnerGlm(Learner):
    def train_from_named_column(self, inputs: dat.Datset, output: dat.NamedColumn) -> Model:
        tfs = noo.transformers_from_datset(inputs)
        tvs = tfs.transform_datset(inputs).termvecs()
        assert self.coltype() == output.coltype()
        return model_glm_from_training(self.coltype(), tvs, output.column(),
                                       lea.modnames_create(tfs, output.coldescribe()))

    def name_as_string(self) -> str:
        pass

    def loglike_derivative_from_row(self, ws: lea.Weights, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        pass

    def loglike_of_element(self, beta_x: float, y: dat.Atom) -> float:
        pass

    def loglike_second_derivative_from_row(self, ws: lea.Weights, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
        pass

    def __init__(self, ct: dat.Coltype):
        self.m_coltype = ct
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_coltype, dat.Coltype)

    def coltype(self) -> dat.Coltype:
        return self.m_coltype


def learner_type_glm(ct: dat.Coltype) -> LearnerGlm:
    return LearnerGlm(ct)


class ModelGlm(Model):
    def assert_ok(self):
        assert isinstance(self.m_gweights, GenWeights)
        self.m_gweights.assert_ok()
        assert isinstance(self.m_modnames, lea.Modnames)
        self.m_modnames.assert_ok()

    def pretty_string(self) -> str:
        return self.gweights().pretty_string(self.modnames())

    def predict_from_row(self, row: dat.Row) -> dis.Distribution:
        return self.gweights().predict_from_termvec(self.modnames().termvec_from_row(row))

    def prediction_component_strings(self) -> arr.Strings:
        return self.gweights().prediction_component_strings(self.modnames().output_describe())

    def modnames(self) -> lea.Modnames:
        return self.m_modnames

    def __init__(self, gws: GenWeights, ms: lea.Modnames):
        self.m_gweights = gws
        self.m_modnames = ms
        self.assert_ok()

    def gweights(self) -> GenWeights:
        return self.m_gweights


def model_glm_create(gws: GenWeights, ms: lea.Modnames) -> ModelGlm:
    return ModelGlm(gws, ms)


def gweights_logistic_create(weights_as_floats: arr.Floats) -> GenWeightsLogistic:
    return GenWeightsLogistic(weights_as_floats)


def gweights_logistic_all_zero2(n_terms: int) -> GenWeightsLogistic:
    return gweights_logistic_create(arr.floats_all_zero(n_terms))


def gweights_all_zero(ct: dat.Coltype, tvs: noo.Termvecs, output: dat.Column) -> GenWeights:
    assert ct == output.coltype()
    if ct == dat.Coltype.bools:
        return gweights_logistic_all_zero2(tvs.num_terms())
    elif ct == dat.Coltype.floats:
        return gweights_linear_all_zero(tvs.num_terms())
    elif ct == dat.Coltype.cats:
        return gweights_multinomial_all_zero(output.cats().num_values(), tvs.num_terms())
    else:
        bas.my_error('bad coltype')


def gweights_from_training(ct: dat.Coltype, inputs: noo.Termvecs, output: dat.Column) -> GenWeights:
    start_weights = gweights_all_zero(ct, inputs, output)
    return start_weights.train(inputs, output)


def model_glm_from_training(ct: dat.Coltype, tvs: noo.Termvecs, output: dat.Column, mns: lea.Modnames) -> ModelGlm:
    gws = gweights_from_training(ct, tvs, output)
    return model_glm_create(gws, mns)


class GenWeights(ABC):
    @abstractmethod
    def num_weights(self) -> int:
        pass

    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def predict_from_termvec(self, tv: noo.Termvec) -> dis.Distribution:
        pass

    @abstractmethod
    def weight(self, windex: int) -> float:
        pass

    def penalty(self) -> float:
        result = 0.0
        for w in self.range():
            result += w * w
        return lea.penalty_parameter * result

    def penalty_derivative(self, windex: int) -> float:
        return 2 * lea.penalty_parameter * self.weight(windex)

    def penalty_2nd_derivative(self, windex: int) -> float:
        assert isinstance(self, GenWeights)
        assert isinstance(windex, int)
        return 2 * lea.penalty_parameter

    def range(self) -> Iterator[float]:
        for i in range(self.num_weights()):
            yield self.weight(i)

    def train(self, inputs: noo.Termvecs, output: dat.Column) -> GenWeights:
        ws = self
        start_ll = self.loglike(inputs, output)
        ll = start_ll

        iteration = 0
        while True:
            if iteration == 0 or bas.is_power_of_two(iteration):
                print(f'Begin iteration {iteration}')
            ll_old = ll
            for windex in range(ws.num_weights()):
                aaa = self.loglike(inputs, output)
                assert bas.loosely_equals(aaa, ll)
                bbb = self.loglike_derivative(inputs, output, windex)
                ccc = self.loglike_2nd_derivative(inputs, output, windex)

                # ll = aaa + bbb * h + 0.5 * ccc * h^2
                # h_star = -bbb/ccc
                assert math.fabs(ccc) > 1e-20
                h_star = -bbb / ccc
                assert isinstance(ws, GenWeights)
                ws2 = ws.deep_copy()
                ws2.increment(windex, h_star)
                ll_new2 = ws2.loglike(inputs, output)
                if ll_new2 > ll:
                    ws = ws2
                    ll = ll_new2

            if math.fabs(ll - ll_old) / (math.fabs(ll - start_ll)) < 1e-5:
                print(f'...finished after {iteration} iterations.')
                return ws

            iteration += 1

    def loglike(self, inputs: noo.Termvecs, output: dat.Column) -> float:
        result = 0.0
        for k, x_k in enumerate(inputs.range()):
            result += self.loglike_from_row(x_k, output, k)
        return result - self.penalty()

    def loglike_derivative(self, inputs: noo.Termvecs, output: dat.Column, windex: int) -> float:
        result = 0.0
        for k, x_k in enumerate(inputs.range()):
            result += self.loglike_derivative_from_row(x_k, output, k, windex)
        return result - self.penalty_derivative(windex)

    def loglike_2nd_derivative(self, inputs: noo.Termvecs, output: dat.Column, windex: int) -> float:
        result = 0.0
        for k, x_k in enumerate(inputs.range()):
            result += self.loglike_2nd_derivative_from_row(x_k, output, k, windex)
        return result - self.penalty_2nd_derivative(windex)

    @abstractmethod
    def loglike_from_row(self, x_k: noo.Termvec, co: dat.Column, row: int) -> float:
        pass

    @abstractmethod
    def loglike_derivative_from_row(self, x_k: noo.Termvec, co: dat.Column, row: int, weight_index: int) -> float:
        pass

    @abstractmethod
    def loglike_2nd_derivative_from_row(self, x_k: noo.Termvec, co: dat.Column, row: int, weight_index: int) -> float:
        pass

    def pretty_string(self, mns: lea.Modnames) -> str:
        return self.pretty_strings(mns).concatenate_fancy('', '\n', '')

    @abstractmethod
    def pretty_strings(self, mns: lea.Modnames) -> arr.Strings:
        pass

    @abstractmethod
    def prediction_component_strings(self, output: dat.Coldescribe) -> arr.Strings:
        pass

    @abstractmethod
    def deep_copy(self) -> GenWeights:
        pass


def q_from_y(y_k: bool) -> float:
    return -1.0 if y_k else 1.0


def pretty_strings_from_weights(weight_names: arr.Strings, weights: arr.Floats) -> arr.Strings:
    assert weight_names.len() == weights.len()
    result = arr.strings_array_empty()
    for name, weight in zip(weight_names.range(), weights.range()):
        result.add(arr.strings_varargs(f'w[{name}]', '=', bas.string_from_float(weight)))
    return result.pretty_strings()


class GenWeightsLogistic(GenWeights):
    def deep_copy(self) -> GenWeightsLogistic:
        return gweights_logistic_create(self.floats().deep_copy())

    def num_weights(self) -> int:
        return self.m_windex_to_floats.len()

    def predict_from_termvec(self, tv: noo.Termvec) -> dis.Distribution:
        beta_x = self.floats().dot_product(tv.floats())
        ez = math.exp(beta_x)
        p = ez / (1 + ez)
        return dis.binomial_create(p)

    def weight(self, windex: int) -> float:
        return self.floats().float(windex)

    def loglike_derivative_from_row(self, x_k: noo.Termvec, co: dat.Column, row: int, weight_index: int) -> float:
        assert isinstance(co, dat.ColumnBool)
        q_k = q_from_y(co.bool(row))
        beta_x = self.floats().dot_product(x_k.floats())
        q_beta_x = q_k * beta_x
        ek_of_beta = math.exp(-q_beta_x)
        fk_of_beta = 1 + ek_of_beta
        return -q_k * x_k.float(weight_index) / fk_of_beta

    def pretty_strings(self, ms: lea.Modnames) -> arr.Strings:
        intro = f'P({ms.output_describe().colname().string()}|x) = sigmoid(w^T x)'
        result = arr.strings_varargs(intro, '...where...')
        result.append(pretty_strings_from_weights(ms.input_noomnames().strings(), self.floats()))
        return result

    def prediction_component_strings(self, output: dat.Coldescribe) -> arr.Strings:
        return arr.strings_singleton(f'p_{output.colname().string()}')

    def __init__(self, windex_to_weight: arr.Floats):
        self.m_windex_to_floats = windex_to_weight
        self.assert_ok()

    def floats(self) -> arr.Floats:
        return self.m_windex_to_floats

    def assert_ok(self):
        assert isinstance(self.m_windex_to_floats, arr.Floats)

    def loglike_2nd_derivative_from_row(self, x_k: noo.Termvec, co: dat.Column, row: int, wx: int) -> float:
        assert isinstance(co, dat.ColumnBool)
        qk = q_from_y(co.bool(row))  # note the qks get multiplied out so not really needed
        beta_x = self.floats().dot_product(x_k.floats())
        q_beta_x = qk * beta_x
        ek_of_beta = math.exp(-q_beta_x)
        ek_of_minus_beta = 1 / ek_of_beta
        fk_of_beta = 1 + ek_of_beta
        fk_of_minus_beta = 1 + ek_of_minus_beta
        x_kj = x_k.float(wx)
        return -x_kj * x_kj / (fk_of_beta * fk_of_minus_beta)

    def loglike_from_row(self, x_k: noo.Termvec, co: dat.ColumnBool, row: int) -> float:
        assert isinstance(co, dat.ColumnBool)
        q = q_from_y(co.bool(row))
        q_beta_x = q * self.floats().dot_product(x_k.floats())
        ek_of_minus_beta = math.exp(q_beta_x)  # negative of negative is positive
        fk_of_minus_beta = 1 + ek_of_minus_beta
        return -math.log(fk_of_minus_beta)


class GenWeightsLinear(GenWeights):
    def deep_copy(self) -> GenWeights:
        pass

    def num_weights(self) -> int:
        return self.m_windex_to_float.len()

    def assert_ok(self):
        assert isinstance(self.m_windex_to_float, arr.Floats)
        self.m_windex_to_float.assert_ok()
        assert isinstance(self.m_sdev, float)
        assert self.m_sdev > 0.0

    def predict_from_termvec(self, tv: noo.Termvec) -> dis.Distribution:
        beta_x = self.floats().dot_product(tv.floats())
        return dis.gaussian_create(beta_x, self.sdev())

    def weight(self, windex: int) -> float:
        return self.floats().float(windex)

    def loglike_2nd_derivative_from_row(self, x_k: noo.Termvec, co: dat.Column, row: int, weight_index: int) -> float:
        x_kj = x_k.float(weight_index)
        return -x_kj * x_kj

    def __init__(self, fs: arr.Floats, sdev: float):
        self.m_windex_to_float = fs
        self.m_sdev = sdev
        self.assert_ok()

    def floats(self) -> arr.Floats:
        return self.m_windex_to_float

    def sdev(self) -> float:
        return self.m_sdev

    def pretty_strings(self, ms: lea.Modnames) -> arr.Strings:
        intro = f'p({ms.output_describe().colname().string()}|x) ~ Normal(mu = w^T x, sdev={self.sdev()})'
        result = arr.strings_varargs(intro, "...where...")
        result.append(pretty_strings_from_weights(ms.input_noomnames().strings(), self.floats()))
        return result

    def prediction_component_strings(self, output: dat.Coldescribe) -> arr.Strings:
        result = arr.strings_singleton(output.colname().string())
        result.add('sdev')
        return result

    def loglike_from_row(self, x_k: noo.Termvec, co: dat.Column, row: int) -> float:
        assert isinstance(co, dat.ColumnFloats)
        correct = co.float(row)
        assert isinstance(correct, float)
        beta_x = self.floats().dot_product(x_k.floats())
        assert isinstance(beta_x, float)
        delta = correct - beta_x
        return -0.5 * delta * delta - bas.log_root_two_pi

    def loglike_derivative_from_row(self, x_k: noo.Termvec, co: dat.Column, row: int, windex: int) -> float:
        assert isinstance(co, dat.ColumnFloats)
        return x_k.float(windex) * (co.float(row) - self.floats().dot_product(x_k.floats()))


def gweights_linear_create(weights_as_floats: arr.Floats) -> GenWeightsLinear:
    return GenWeightsLinear(weights_as_floats, 1.0)


def gweights_linear_all_zero(n_terms: int) -> GenWeightsLinear:
    return gweights_linear_create(arr.floats_all_zero(n_terms))


class GenWeightsMultinomial(GenWeights):
    def num_weights(self) -> int:
        return self.num_terms() * (self.num_values() - 1)

    def assert_ok(self):
        fm = self.m_value_to_term_num_to_weight
        assert isinstance(fm, arr.Fmat)
        fm.assert_ok()
        assert fm.num_rows() > 0
        assert fm.row(0).loosely_equals(arr.floats_all_zero(fm.num_cols()))

    def eki(self, z: noo.Termvec, i: int) -> float:
        return math.exp(self.qki(z, i))

    def qki(self, x_k: noo.Termvec, i: int) -> float:
        return self.floats_from_value(i).dot_product(x_k.floats())

    def sk(self, z: noo.Termvec) -> float:
        result = 0
        for i in range(self.num_values()):
            result += self.eki(z, i)
        return result

    def predict_from_termvec(self, tv: noo.Termvec) -> dis.Distribution:
        sk = self.sk(tv)
        probs = arr.floats_empty()
        for i in range(self.num_values()):
            eki = self.eki(tv, i)
            probs.add(eki / sk)
        return dis.multinomial_create(probs)

    def weight(self, windex: int) -> float:
        value, term_num = self.value_and_term_num_from_windex(windex)
        return self.floats_from_value(value).float(term_num)

    def loglike_from_row(self, x_k: noo.Termvec, co: dat.Column, row: int) -> float:
        assert isinstance(co, dat.ColumnCats)
        i = co.cats().value(row)
        qki = self.qki(x_k, i)
        sk = self.sk(x_k)
        return qki - math.log(sk)

    def loglike_derivative_from_row(self, x_k: noo.Termvec, co: dat.Column, row: int, weight_index: int) -> float:
        windex_value, term_num = self.value_and_term_num_from_windex(weight_index)
        xkj = x_k.float(term_num)
        pik = self.eki(x_k, windex_value) / self.sk(x_k)
        d_ll_k_by_dw_ij = -xkj * pik
        this_rows_value = co.cats().value(row)
        if this_rows_value == windex_value:
            d_ll_k_by_dw_ij += xkj
        return d_ll_k_by_dw_ij

    def loglike_2nd_derivative_from_row(self, x_k: noo.Termvec, co: dat.Column, row: int, weight_index: int) -> float:
        windex_value, term_num = self.value_and_term_num_from_windex(weight_index)
        xkj = x_k.float(term_num)
        pik = self.eki(x_k, windex_value) / self.sk(x_k)
        d2_ll_k_by_dw2_ij = -xkj * xkj * pik * (1 - pik)
        return d2_ll_k_by_dw2_ij

    def pretty_strings(self, mns: lea.Modnames) -> arr.Strings:
        result = arr.strings_empty()
        result.add(f'P({mns.output_describe().colname().string()}=v|Weights,x) = exp(Weights[v] . x) / K where...')
        for vn, weights in zip(mns.output_describe().valnames().range(), self.fmat().range_rows()):
            weight_names = arr.strings_empty()
            for nn in mns.input_noomnames().range():
                weight_names.add(f'{vn.string()},{nn.string()}')
            result.append(pretty_strings_from_weights(weight_names, weights))
        return result

    def prediction_component_strings(self, output: dat.Coldescribe) -> arr.Strings:
        result = arr.strings_empty()
        for vn in output.valnames().range():
            result.add(f'p_{vn.string()}')
        return result

    def __init__(self, value_to_term_num_to_weight: arr.Fmat, ):
        self.m_value_to_term_num_to_weight = value_to_term_num_to_weight
        self.assert_ok()

    def num_terms(self) -> int:
        return self.m_value_to_term_num_to_weight.num_cols()

    def num_values(self) -> int:
        return self.m_value_to_term_num_to_weight.num_rows()

    def floats_from_value(self, value: int) -> arr.Floats:
        return self.fmat().row(value)

    def fmat(self) -> arr.Fmat:
        return self.m_value_to_term_num_to_weight

    def value_and_term_num_from_windex(self, windex: int) -> Tuple[int, int]:
        n_values = self.num_values()
        term_num = windex % n_values
        value = math.floor(windex / n_values) + 1
        assert isinstance(value, int)
        assert 0 <= term_num < self.num_terms()
        assert 1 <= value < self.num_values()
        assert windex == self.windex_from_value_and_term_num(value, term_num)
        return value, term_num

    def windex_from_value_and_term_num(self, value: int, term_num: int) -> int:
        assert 1 <= value < self.num_values()
        assert 0 <= term_num < self.num_terms()
        return (value - 1) * self.num_values() + term_num


def gweights_multinomial_create(weights_as_fmat: arr.Fmat) -> GenWeightsMultinomial:
    return GenWeightsMultinomial(weights_as_fmat)


def gweights_multinomial_all_zero(n_values: int, n_terms: int) -> GenWeightsMultinomial:
    return gweights_multinomial_create(arr.fmat_of_zeroes(n_values, n_terms))
