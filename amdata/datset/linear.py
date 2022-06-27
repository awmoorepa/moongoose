from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterator, Tuple

import datset.amarrays as arr
import datset.ambasic as bas
import datset.distribution as dis
import datset.dset as dat
import datset.learn as lea
import datset.numset as noo


class TermRecord:
    def __init__(self,fs:arr.Floats):
        self.m_floats = fs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_floats,arr.Floats)
        self.m_floats.assert_ok()


class TermRecords:
    def __init__(self, n_terms:int):
        self.m_num_terms = n_terms
        self.m_term_records = []
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_num_terms, int)
        assert isinstance(self.m_term_records, list)
        for tr in self.m_term_records:
            assert isinstance(tr,TermRecord)
            tr.assert_ok()
            assert tr.num_terms() == self.m_num_terms

    def add(self, tr:TermRecord):
        self.m_term_records.append(tr)


def term_records_empty(n_terms:int)->TermRecords:
    return TermRecords(n_terms)


def product_of_float_record_elements(fr:noo.FloatRecord, fr_indexes:arr.Ints)->float:
    result = 1.0
    for fr_index in fr_indexes.range():
        result *= fr.float(fr_index)
    return result


def term_record_from_floats(fs:arr.Floats)->TermRecord:
    return TermRecord(fs)


class PolynomialStructure:
    def init(self,n_fr_indexes:int):
        self.m_num_fr_indexes = n_fr_indexes
        self.m_t_index_to_fr_indexes = arr.ints_array_empty()
        self.m_fr_indexes_to_t_index = {}
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_num_fr_indexes, int)
        assert isinstance(self.m_fr_indexes_to_t_index,dict)
        for fr_indexes,t_index in self.m_fr_indexes_to_t_index:
            assert isinstance(fr_indexes,arr.Ints)
            assert isinstance(t_index,int)

        for t_index,fr_indexes in enumerate(self.m_t_index_to_fr_indexes.range()):
            assert self.contains(fr_indexes)
            assert self.t_index_from_fr_indexes(fr_indexes) == t_index
            if fr_indexes.len() > 0:
                assert fr_indexes.max() < self.m_num_fr_indexes
            assert fr_indexes.is_weakly_increasing()

    def term_records_from_float_records(self,frs:noo.FloatRecords)->TermRecords:
        assert self.num_fr_indexes() == frs.num_cols()
        result = term_records_empty(self.num_terms())
        for fr in frs.range():
            result.add(self.term_record_from_float_record(fr))
        return result

    def t_index_from_fr_indexes(self, fr_indexes:arr.Ints)->int:
        assert self.contains(fr_indexes)
        return self.m_fr_indexes_to_t_index[fr_indexes]

    def contains(self, fr_indexes:arr.Ints)->bool:
        return fr_indexes in self.m_fr_indexes_to_t_index

    def num_fr_indexes(self)->int:
        return self.m_num_fr_indexes

    def num_terms(self)->int:
        return self.m_t_index_to_fr_indexes.len()

    def term_record_from_float_record(self, fr:noo.FloatRecord)->TermRecord:
        result = arr.floats_empty()
        for fr_indexes in self.range_fr_indexes():
            result.add(product_of_float_record_elements(fr,fr_indexes))
        return term_record_from_floats(result)

    def range_fr_indexes(self)->Iterator[arr.Ints]:
        for fr_indexes in self.m_fr_indexes_to_t_index:
            yield fr_indexes


def pretty_name_from_fr_indexes(fr_indexes:arr.Ints, fr_index_to_name:arr.Strings)->str:
    pretty_name: str

    if fr_indexes.len() == 0:
        pretty_name = 'constant'
    else:
        pretty_name_components = arr.strings_empty()
        power = 0
        for i in range(fr_indexes.len()):
            fr_index = fr_indexes.int(i)
            if i == 0:
                power = 1
            else:
                previous_fr_index = fr_indexes.int(i - 1)
                if fr_index == previous_fr_index:
                    power += 1
                else:
                    base = fr_index_to_name.string(previous_fr_index)
                    pretty_name_components.add(base if power == 1 else f'{base}^{power}')
                    power = 1

        base = fr_index_to_name.string(fr_indexes.last_element())
        pretty_name_components.add(base if power == 1 else f'{base}^{power}')

        return pretty_name_components.concatenate_fancy('', ' * ', '')


class Polynomial:
    def init(self):
        self.m_poly_structure = polynomial_structure_empty()
        self.m_t_index_to_coefficient = arr.floats_empty()
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_poly_structure,PolynomialStructure)
        self.m_poly_structure.assert_ok()
        assert isinstance(self.m_t_index_to_coefficient,arr.Floats)
        self.m_t_index_to_coefficient.assert_ok()
        assert self.m_poly_structure.num_terms() == self.m_t_index_to_coefficient.len()

    def pretty_names(self, float_record_names:arr.Strings)->arr.Strings:
        result = arr.strings_empty()
        for fr_indexes in self.range_fr_indexes():
            result.add(pretty_name_from_fr_indexes(fr_indexes,float_record_names))
        return result

    def apw_helper(self,pws:arr.Floats, uws:arr.Floats, frs:arr.Ints, inc:arr.Ints, w:float):
        if frs.len()==0:
            t_index = self.t_index_from_fr_indexes(inc)
            pws.increment(t_index,w)
        else:
            fr_index,rest = frs.first_and_rest()
            iv = self.interval(fr_index)
            lo = iv.lo()
            ugly = uws.float(fr_index)
            width = iv.width()
            # pretty = lo + ugly * width
            self.apw_helper(pws,uws,rest,inc.with_this_added(fr_index),w/width)
            self.apw_helper(pws,uws,rest,inc,-lo*w/width)

    def account_for_transformer_intervals(self,fr_index_to_interval:bas.Intervals)->Polynomial:
        result = polynomial_with_no_terms()
        for coeff,fr_indexes in zip(self.range_coeffs(),self.range_fr_indexes()):
            result.add_polynomial(poly_accounting_for_intervals(coeff, fr_indexes, fr_index_to_interval))
        return result

    def t_index_from_fr_indexes(self, fr_indexes:arr.Ints)->int:
        assert fr_indexes.is_weakly_increasing()
        for t_index,fis in enumerate(self.fr_indexes().range()):
            if fis.equals(fr_indexes):
                return t_index
        bas.my_error('fr_indexes not found')

    def range_fr_indexes(self)->Iterator[arr.Ints]:
        return self.polynomial_structure().range_fr_indexes()

    def add_term(self, coeff:float, fr_indexes:arr.Ints):
        self.polynomial_structure().add_term(fr_indexes)
        self.m_t_index_to_coefficient.add(coeff)

    def polynomial_structure(self)->PolynomialStructure:
        return self.m_poly_structure


def polynomial_with_no_terms() -> Polynomial:
    return Polynomial()


def polynomial_linear(constant: float, linear_coeff: float, fr_index: int) -> Polynomial:
    result = polynomial_constant(constant)
    result.add_term(linear_coeff, arr.ints_singleton(fr_index))
    return result


def polynomial_constant(c: float) -> Polynomial:
    result = polynomial_with_no_terms()
    result.add_term(c, arr.ints_empty())
    return result


def polynomial_one() -> Polynomial:
    return polynomial_constant(1.0)


def poly_accounting_for_intervals(fr_indexes: arr.Ints, fr_index_to_interval: bas.Intervals) -> Polynomial:
    result = polynomial_one()
    for fr_index in fr_indexes.range():
        iv = fr_index_to_interval.interval(fr_index)
        # multiply result by transformed version of of this fr_index
        transformed_version = polynomial_linear(-iv.lo() / iv.width(), 1.0 / iv.width(), fr_index)
        result = result.times(transformed_version)
    return result



def floater_glm_create(p:Polynomial, ws:GenWeights)->FloaterGlm:
    return FloaterGlm(p,ws)


class FloaterClassGlm(lea.FloaterClass):
    def train_from_named_column(self, inputs: noo.NamedFloatRecords, output: dat.NamedColumn) -> FloaterGlm:
        trs = self.polynomial().term_records_from_float_records(inputs)
        ws = gweights_from_training(trs,output)
        return floater_glm_create(self.polynomial(),ws)

    def name_as_string(self) -> str:
        pass

    def loglike_derivative_from_row(self, ws: lea.Weights, x_k: noo.FloatRecord, y_k: dat.Atom, j: int) -> float:
        pass

    def loglike_of_element(self, beta_x: float, y: dat.Atom) -> float:
        pass

    def loglike_second_derivative_from_row(self, ws: lea.Weights, x_k: noo.FloatRecord, y_k: dat.Atom, j: int) -> float:
        pass

    def __init__(self, polynomial_degree:int):
        self.m_polynomial_degree = polynomial_degree
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_polynomial_degree, int)
        assert 0 <= self.m_polynomial_degree < 100

    def polynomial_degree(self) -> int:
        return self.m_polynomial_degree



def model_class_glm(polynomial_degree: int) -> FloaterClassGlm:
    return FloaterClassGlm(polynomial_degree)


class FloaterGlm(lea.Floater):
    def assert_ok(self):
        assert isinstance(self.m_gweights, GenWeights)
        self.m_gweights.assert_ok()
        assert isinstance(self.m_term_names, arr.Strings)
        self.m_term_names.assert_ok()
        assert self.m_term_names.len() == self.m_gweights.num_terms()

    def pretty_string(self,input_names: arr.Strings, normalization_intervals:bas.Intervals,output: dat.ColumnDescription) -> str:
        return self.polynomial

    def predict_from_record(self, rec: dat.Record) -> dis.Distribution:
        return self.gweights().predict_from_termvec(self.model_description().termvec_from_row(rec))

    def prediction_component_strings(self) -> arr.Strings:
        return self.gweights().prediction_component_strings(self.model_description().output())

    def model_description(self) -> lea.ModelDescription:
        return self.m_term_names

    def __init__(self, p:Polynomial):
        self.m_polynomial = p
        self.assert_ok()

    def gweights(self) -> GenWeights:
        return self.m_gweights


def gweights_logistic_create(weights_as_floats: arr.Floats) -> GenWeightsLogistic:
    return GenWeightsLogistic(weights_as_floats)


def gweights_logistic_all_zero2(n_terms: int) -> GenWeightsLogistic:
    return gweights_logistic_create(arr.floats_all_zero(n_terms))


def gweights_all_zero(ct: dat.Coltype, tvs: noo.FloatRecords, output: dat.Column) -> GenWeights:
    assert ct == output.coltype()
    if ct == dat.Coltype.bools:
        return gweights_logistic_all_zero2(tvs.num_cols())
    elif ct == dat.Coltype.floats:
        return gweights_linear_all_zero(tvs.num_cols())
    elif ct == dat.Coltype.cats:
        return gweights_multinomial_all_zero(output.cats().num_values(), tvs.num_cols())
    else:
        bas.my_error('bad coltype')


def gweights_from_training(ct: dat.Coltype, inputs: noo.FloatRecords, output: dat.Column) -> GenWeights:
    start_weights = gweights_all_zero(ct, inputs, output)
    return start_weights.train(inputs, output)


def polynomial_structure_empty()->PolynomialStructure:
    return PolynomialStructure()


class PolyDataBuilder:
    def __init__(self):
        self.m_poly_structure = polynomial_structure_empty()
        self.m_t_index_to_column = arr.floats_array_empty()
        self.assert_ok()

    def column(self, t_index: int) -> arr.Floats:
        return self.m_tindex_to_column.floats(t_index)

    def fr_indexes(self, t_index) -> arr.Ints:
        return self.polynomial().fr_indexes(t_index)

    def columns(self)->arr.FloatsArray:
        return self.m_t_index_to_column



class PolyData:
    def __init__(self,frs:noo.FloatRecords,ps:PolynomialStructure,trs:TermRecords):
        self.m_float_records = frs
        self.m_poly_structure = ps
        self.m_term_records = trs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_float_records,noo.FloatRecords)
        self.m_float_records.assert_ok()
        assert isinstance(self.m_poly_structure,PolynomialStructure)
        self.m_poly_structure.assert_ok()
        assert isinstance(self.m_term_records,TermRecords)
        self.m_term_records.assert_ok()
        assert self.m_float_records.num_cols() == self.m_poly_structure.m_num_fr_indexes()
        assert self.m_poly_structure.num_terms() == self.m_term_records.num_terms()
        for fr,tr in zip(self.m_float_records.range(),self.m_term_records.range()):
            assert tr.loosely_equals(self.m_poly_structure.term_record_from_float_record(fr))



def poly_data_builder_empty() -> PolyDataBuilder:
    return PolyDataBuilder()


def term_record_from_row(t_index_to_column, row)->TermRecords:
    pass


def term_records_from_columns(t_index_to_column:arr.FloatsArray)->TermRecords:
    result = term_records_empty(t_index_to_column.len())
    assert t_index_to_column.len() > 0
    n_rows = t_index_to_column.floats(0).len()
    for row in range(n_rows):
        result.add(term_record_from_row(t_index_to_column,row))
    return result


def poly_data_create(ps:PolynomialStructure, trs:TermRecords)->PolyData:
    return PolyData(ps,trs)


def poly_data_from_poly_data_builder(pd:PolyDataBuilder):
    trs = term_records_from_columns(pd.columns())
    return poly_data_create(pd.polynomial_structure(),trs)


def poly_data_from_float_records(frs:noo.FloatRecords, max_degree:int)->PolyData:
    pd = poly_data_builder_empty()
    pd.add(arr.ints_empty(),arr.floats_all_constant(frs.num_rows(),1.0))
    t_indexes_serving_previous_degree = arr.ints_singleton(0)
    for degree in range(1,max_degree+1):
        new_t_indexes = arr.ints_empty()
        for t_index in t_indexes_serving_previous_degree.range():
            previous_column = pd.column(t_index)
            fr_indexes = pd.fr_indexes(t_index)
            first_available_new_fr_index = 0 if fr_indexes.len==0 else fr_indexes.last_element()
            for fr_index in range(first_available_new_fr_index,frs.num_cols()):
                proposed_new_column = previous_column.products(frs.column_as_floats(t_index))
                if not proposed_new_column.is_loosely_constant():
                    new_t_index = pd.num_terms()
                    new_fr_indexes = fr_indexes.copied_with_one_element_added(fr_index)
                    assert new_fr_indexes.len()==degree
                    assert new_fr_indexes.is_weakly_increasing()
                    assert not pd.contains_fr_indexes(new_fr_indexes)
                    pd.add(new_fr_indexes, proposed_new_column)
                    new_t_indexes.add(new_t_index)
        t_indexes_serving_previous_degree = new_t_indexes

        pd.assert_ok()
        return poly_data_from_poly_data_builder(pd)


def floater_glm_from_training(frs: noo.FloatRecords, output: dat.Column,max_degree:int) -> FloaterGlm:
    pd = poly_data_from_float_records(frs,max_degree)
    gws = gweights_from_training(pd.term_records(), output)
    return floater_glm_create(pd.polynomial_structure(),gws)


class GenWeights(ABC):
    @abstractmethod
    def num_weights(self) -> int:
        pass

    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def predict_from_termvec(self, tv: noo.FloatRecord) -> dis.Distribution:
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

    def train(self, inputs: noo.FloatRecords, output: dat.Column) -> GenWeights:
        ws = self
        start_ll = ws.loglike(inputs, output)
        ll = start_ll

        iteration = 0
        while True:
            if iteration < 5 or bas.is_power_of_two(iteration):
                print(f'Begin iteration {iteration}')
            ll_old = ll
            for windex in range(ws.num_weights()):
                aaa = ws.loglike(inputs, output)
                assert bas.loosely_equals(aaa, ll)
                bbb = ws.loglike_derivative(inputs, output, windex)
                ccc = ws.loglike_2nd_derivative(inputs, output, windex)

                # ll = aaa + bbb * h + 0.5 * ccc * h^2
                # h_star = -bbb/ccc
                assert math.fabs(ccc) > 1e-20
                h_star = -bbb / ccc
                assert isinstance(ws, GenWeights)
                ws2 = ws.deep_copy()
                assert isinstance(ws2, GenWeights)
                ws2.increment(windex, h_star)
                ll_new2 = ws2.loglike(inputs, output)
                if ll_new2 > ll:
                    print(f"it's worth incrementing windex {windex} by {h_star} (delta_ll={ll_new2 - ll})")
                    ws = ws2
                    ll = ll_new2

            if math.fabs(ll - ll_old) / (math.fabs(ll - start_ll)) < 1e-5:
                print(f'...finished after {iteration} iterations.')
                return ws

            iteration += 1

    def loglike(self, inputs: noo.FloatRecords, output: dat.Column) -> float:
        result = 0.0
        for k, x_k in enumerate(inputs.range()):
            result += self.loglike_from_row(x_k, output, k)
        return result - self.penalty()

    def loglike_derivative(self, inputs: noo.FloatRecords, output: dat.Column, windex: int) -> float:
        result = 0.0
        for k, x_k in enumerate(inputs.range()):
            result += self.loglike_derivative_from_row(x_k, output, k, windex)
        return result - self.penalty_derivative(windex)

    def loglike_2nd_derivative(self, inputs: noo.FloatRecords, output: dat.Column, windex: int) -> float:
        result = 0.0
        for k, x_k in enumerate(inputs.range()):
            result += self.loglike_2nd_derivative_from_row(x_k, output, k, windex)
        return result - self.penalty_2nd_derivative(windex)

    @abstractmethod
    def loglike_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int) -> float:
        pass

    @abstractmethod
    def loglike_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int, weight_index: int) -> float:
        pass

    @abstractmethod
    def loglike_2nd_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int, weight_index: int) -> float:
        pass

    def pretty_string(self, p:Polynomial,input_names:arr.Strings,output:dat.ColumnDescription) -> str:
        return self.pretty_strings(p,input_names,output).concatenate_fancy('', '\n', '')

    @abstractmethod
    def pretty_strings(self, p:Polynomial,input_names:arr.Strings,output:dat.ColumnDescription) -> arr.Strings:
        pass

    @abstractmethod
    def prediction_component_strings(self, output: dat.ColumnDescription) -> arr.Strings:
        pass

    @abstractmethod
    def deep_copy(self) -> GenWeights:
        pass

    @abstractmethod
    def increment(self, windex: int, delta: float):
        pass

    @abstractmethod
    def num_terms(self):
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
    def increment(self, windex: int, delta: float):
        self.m_windex_to_floats.increment(windex, delta)

    def deep_copy(self) -> GenWeightsLogistic:
        return gweights_logistic_create(self.floats().deep_copy())

    def num_weights(self) -> int:
        return self.m_windex_to_floats.len()

    def predict_from_termvec(self, tv: noo.FloatRecord) -> dis.Distribution:
        beta_x = self.floats().dot_product(tv.floats())
        ez = math.exp(beta_x)
        p = ez / (1 + ez)
        return dis.binomial_create(p)

    def weight(self, windex: int) -> float:
        return self.floats().float(windex)

    def loglike_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int, weight_index: int) -> float:
        assert isinstance(co, dat.ColumnBools)
        q_k = q_from_y(co.bool(row))
        beta_x = self.floats().dot_product(x_k.floats())
        q_beta_x = q_k * beta_x
        ek_of_beta = math.exp(-q_beta_x)
        fk_of_beta = 1 + ek_of_beta
        return -q_k * x_k.float(weight_index) / fk_of_beta

    def pretty_strings(self, p:Polynomial,float_record_names:arr.Strings,output:dat.ColumnDescription) -> arr.Strings:
        intro = f'P({output.colname().string()}|x) = sigmoid(w^T x)'
        result = arr.strings_varargs(intro, '...where...')
        pretty_names,pretty_weights = self.polynomial().pretty_items(float_record_names,self.gweights())
        weight_strings = pretty_strings_from_weights(pretty_names,pretty_weights)
        result.append(weight_strings)
        return result

    def prediction_component_strings(self, output: dat.ColumnDescription) -> arr.Strings:
        return arr.strings_singleton(f'p_{output.colname().string()}')

    def __init__(self, windex_to_weight: arr.Floats):
        self.m_windex_to_floats = windex_to_weight
        self.assert_ok()

    def floats(self) -> arr.Floats:
        return self.m_windex_to_floats

    def assert_ok(self):
        assert isinstance(self.m_windex_to_floats, arr.Floats)

    def loglike_2nd_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int, wx: int) -> float:
        assert isinstance(co, dat.ColumnBools)
        qk = q_from_y(co.bool(row))  # note the qks get multiplied out so not really needed
        beta_x = self.floats().dot_product(x_k.floats())
        q_beta_x = qk * beta_x
        ek_of_beta = math.exp(-q_beta_x)
        ek_of_minus_beta = 1 / ek_of_beta
        fk_of_beta = 1 + ek_of_beta
        fk_of_minus_beta = 1 + ek_of_minus_beta
        x_kj = x_k.float(wx)
        return -x_kj * x_kj / (fk_of_beta * fk_of_minus_beta)

    def loglike_from_row(self, x_k: noo.FloatRecord, co: dat.ColumnBools, row: int) -> float:
        assert isinstance(co, dat.ColumnBools)
        q = q_from_y(co.bool(row))
        q_beta_x = q * self.floats().dot_product(x_k.floats())
        ek_of_minus_beta = math.exp(q_beta_x)  # negative of negative is positive
        fk_of_minus_beta = 1 + ek_of_minus_beta
        return -math.log(fk_of_minus_beta)

    def polynomial(self)->Polynomial:
        return self.m_polynomial


class GenWeightsLinear(GenWeights):
    def increment(self, windex: int, delta: float):
        self.m_windex_to_float.increment(windex, delta)

    def deep_copy(self) -> GenWeights:
        return gweights_linear_create(self.m_windex_to_float.deep_copy(), self.sdev())

    def num_weights(self) -> int:
        return self.m_windex_to_float.len()

    def assert_ok(self):
        assert isinstance(self.m_windex_to_float, arr.Floats)
        self.m_windex_to_float.assert_ok()
        assert isinstance(self.m_sdev, float)
        assert self.m_sdev > 0.0

    def predict_from_termvec(self, tv: noo.FloatRecord) -> dis.Distribution:
        beta_x = self.floats().dot_product(tv.floats())
        return dis.gaussian_create(beta_x, self.sdev())

    def weight(self, windex: int) -> float:
        return self.floats().float(windex)

    def loglike_2nd_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int, weight_index: int) -> float:
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

    def pretty_strings(self, p:Polynomial,float_record_names:arr.Strings,output:dat.ColumnDescription) -> arr.Strings:
        intro = f'p({output.colname().string()}|x) ~ Normal(mu = w^T x, sdev={self.sdev()})'
        result = arr.strings_varargs(intro, "...where...")
        pretty_names,pretty_weights = p.pretty_items(float_record_names,self)
        result.append(pretty_strings_from_weights(pretty_names, pretty_weights))
        return result

    def prediction_component_strings(self, output: dat.ColumnDescription) -> arr.Strings:
        result = arr.strings_singleton(output.colname().string())
        result.add('sdev')
        return result

    def loglike_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int) -> float:
        assert isinstance(co, dat.ColumnFloats)
        correct = co.float(row)
        assert isinstance(correct, float)
        beta_x = self.floats().dot_product(x_k.floats())
        assert isinstance(beta_x, float)
        delta = correct - beta_x
        return -0.5 * delta * delta - bas.log_root_two_pi

    def loglike_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int, windex: int) -> float:
        assert isinstance(co, dat.ColumnFloats)
        return x_k.float(windex) * (co.float(row) - self.floats().dot_product(x_k.floats()))

def gweights_linear_create(weights_as_floats: arr.Floats, sdev: float) -> GenWeightsLinear:
    return GenWeightsLinear(weights_as_floats, sdev)


def gweights_linear_all_zero(n_terms: int) -> GenWeightsLinear:
    return gweights_linear_create(arr.floats_all_zero(n_terms), 1.0)


class GenWeightsMultinomial(GenWeights):
    def increment(self, windex: int, delta: float):
        value, term_num = self.value_and_term_num_from_windex(windex)
        self.m_value_to_term_num_to_weight.increment(value, term_num, delta)

    def deep_copy(self) -> GenWeights:
        return gweights_multinomial_create(self.m_value_to_term_num_to_weight.deep_copy())

    def num_weights(self) -> int:
        return self.num_terms() * (self.num_values() - 1)

    def assert_ok(self):
        fm = self.m_value_to_term_num_to_weight
        assert isinstance(fm, arr.Fmat)
        fm.assert_ok()
        assert fm.num_rows() > 0
        assert fm.row(0).loosely_equals(arr.floats_all_zero(fm.num_cols()))

    def eki(self, z: noo.FloatRecord, i: int) -> float:
        return math.exp(self.qki(z, i))

    def qki(self, x_k: noo.FloatRecord, i: int) -> float:
        return self.floats_from_value(i).dot_product(x_k.floats())

    def sk(self, z: noo.FloatRecord) -> float:
        result = 0
        for i in range(self.num_values()):
            result += self.eki(z, i)
        return result

    def predict_from_termvec(self, tv: noo.FloatRecord) -> dis.Distribution:
        sk = self.sk(tv)
        probs = arr.floats_empty()
        for i in range(self.num_values()):
            eki = self.eki(tv, i)
            probs.add(eki / sk)
        return dis.multinomial_create(probs)

    def weight(self, windex: int) -> float:
        value, term_num = self.value_and_term_num_from_windex(windex)
        return self.floats_from_value(value).float(term_num)

    def loglike_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int) -> float:
        assert isinstance(co, dat.ColumnCats)
        i = co.cats().value(row)
        qki = self.qki(x_k, i)
        sk = self.sk(x_k)
        return qki - math.log(sk)

    def loglike_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int, weight_index: int) -> float:
        windex_value, term_num = self.value_and_term_num_from_windex(weight_index)
        xkj = x_k.float(term_num)
        pik = self.eki(x_k, windex_value) / self.sk(x_k)
        d_ll_k_by_dw_ij = -xkj * pik
        this_rows_value = co.cats().value(row)
        if this_rows_value == windex_value:
            d_ll_k_by_dw_ij += xkj
        return d_ll_k_by_dw_ij

    def loglike_2nd_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int, weight_index: int) -> float:
        windex_value, term_num = self.value_and_term_num_from_windex(weight_index)
        xkj = x_k.float(term_num)
        pik = self.eki(x_k, windex_value) / self.sk(x_k)
        d2_ll_k_by_dw2_ij = -xkj * xkj * pik * (1 - pik)
        return d2_ll_k_by_dw2_ij

    def pretty_strings(self, float_record_names:arr.Strings,output:dat.ColumnDescription) -> arr.Strings:
        result = arr.strings_empty()
        result.add(f'P({output.colname().string()}=v|Weights,x) = exp(Weights[v] . x) / K where...')
        for vn, weights in zip(output.valnames().range(), self.fmat().range_rows()):
            pretty_names, pretty_weights = self.polynomial().pretty_items(float_record_names, self.gweights())
            final_names = arr.strings_empty()
            for pn in pretty_names.range():
                final_names.add(f'{vn.string()},{pn.string()}')
            result.append(pretty_strings_from_weights(final_names, pretty_weights))
        return result

    def prediction_component_strings(self, output: dat.ColumnDescription) -> arr.Strings:
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
        term_num = windex % self.num_terms()
        value = math.floor(windex / self.num_terms()) + 1
        assert isinstance(value, int)
        assert 0 <= term_num < self.num_terms()
        assert 1 <= value < self.num_values()
        assert windex == self.windex_from_value_and_term_num(value, term_num)
        return value, term_num

    def windex_from_value_and_term_num(self, value: int, term_num: int) -> int:
        assert 1 <= value < self.num_values()
        assert 0 <= term_num < self.num_terms()
        return (value - 1) * self.num_terms() + term_num


def gweights_multinomial_create(weights_as_fmat: arr.Fmat) -> GenWeightsMultinomial:
    return GenWeightsMultinomial(weights_as_fmat)


def gweights_multinomial_all_zero(n_values: int, n_terms: int) -> GenWeightsMultinomial:
    return gweights_multinomial_create(arr.fmat_of_zeroes(n_values, n_terms))
