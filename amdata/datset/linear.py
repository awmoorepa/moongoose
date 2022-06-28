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
    def __init__(self, fs: arr.Floats):
        self.m_floats = fs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_floats, arr.Floats)
        self.m_floats.assert_ok()

    def num_terms(self) -> int:
        return self.floats().len()

    def floats(self) -> arr.Floats:
        return self.m_floats


class TermRecords:
    def __init__(self, n_terms: int):
        self.m_num_terms = n_terms
        self.m_term_records = []
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_num_terms, int)
        assert isinstance(self.m_term_records, list)
        for tr in self.m_term_records:
            assert isinstance(tr, TermRecord)
            tr.assert_ok()
            assert tr.num_terms() == self.m_num_terms

    def add(self, tr: TermRecord):
        self.m_term_records.append(tr)

    def num_terms(self) -> int:
        return self.m_num_terms

    def range(self) -> Iterator[TermRecord]:
        for tr in self.m_term_records:
            yield tr


def term_records_empty(n_terms: int) -> TermRecords:
    return TermRecords(n_terms)


def product_of_float_record_elements(fr: noo.FloatRecord, fr_indexes: arr.Ints) -> float:
    result = 1.0
    for fr_index in fr_indexes.range():
        result *= fr.float(fr_index)
    return result


def term_record_from_floats(fs: arr.Floats) -> TermRecord:
    return TermRecord(fs)


class PolynomialStructure:
    def __init__(self, n_fr_indexes: int):
        self.m_num_fr_indexes = n_fr_indexes
        self.m_t_index_to_fr_indexes = arr.ints_array_empty()
        self.m_fr_indexes_to_t_index = {}
        self.assert_ok()

    def pretty_weight_names(self, float_record_names: arr.Strings) -> arr.Strings:
        result = arr.strings_empty()
        for fr_indexes in self.range_fr_indexes():
            result.add(pretty_from_fr_indexes(fr_indexes, float_record_names))
        return result

    def assert_ok(self):
        assert isinstance(self.m_num_fr_indexes, int)
        assert isinstance(self.m_fr_indexes_to_t_index, dict)
        for fr_indexes, t_index in self.m_fr_indexes_to_t_index:
            assert isinstance(fr_indexes, arr.Ints)
            assert isinstance(t_index, int)

        for t_index, fr_indexes in enumerate(self.m_t_index_to_fr_indexes.range()):
            assert self.contains(fr_indexes)
            assert self.t_index_from_fr_indexes(fr_indexes) == t_index
            if fr_indexes.len() > 0:
                assert fr_indexes.max() < self.m_num_fr_indexes
            assert fr_indexes.is_weakly_increasing()

    def term_records_from_float_records(self, frs: noo.FloatRecords) -> TermRecords:
        assert self.num_fr_indexes() == frs.num_cols()
        result = term_records_empty(self.num_terms())
        for fr in frs.range():
            result.add(self.term_record_from_float_record(fr))
        return result

    def t_index_from_fr_indexes(self, fr_indexes: arr.Ints) -> int:
        assert self.contains(fr_indexes)
        return self.m_fr_indexes_to_t_index[fr_indexes]

    def contains(self, fr_indexes: arr.Ints) -> bool:
        return fr_indexes in self.m_fr_indexes_to_t_index

    def num_fr_indexes(self) -> int:
        return self.m_num_fr_indexes

    def num_terms(self) -> int:
        return self.m_t_index_to_fr_indexes.len()

    def term_record_from_float_record(self, fr: noo.FloatRecord) -> TermRecord:
        result = arr.floats_empty()
        for fr_indexes in self.range_fr_indexes():
            result.add(product_of_float_record_elements(fr, fr_indexes))
        return term_record_from_floats(result)

    def range_fr_indexes(self) -> Iterator[arr.Ints]:
        for fr_indexes in self.m_fr_indexes_to_t_index:
            yield fr_indexes

    def add_term(self, fr_indexes: arr.Ints):
        assert 0 <= fr_indexes.min() <= fr_indexes.max() < self.num_fr_indexes()
        t_index = self.num_terms()
        c = fr_indexes.deep_copy()
        self.m_fr_indexes_to_t_index[c] = t_index
        self.m_t_index_to_fr_indexes.add(c)
        if bas.expensive_assertions:
            self.assert_ok()

    def fr_indexes(self, t_index: int) -> arr.Ints:
        assert 0 <= t_index < self.num_terms()
        return self.m_t_index_to_fr_indexes.ints(t_index)


def pretty_from_fr_indexes(fr_indexes: arr.Ints, fr_index_to_name: arr.Strings) -> str:
    pretty_name: str

    if fr_indexes.len() == 0:
        return 'constant'
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


def pretty_from_coefficient(coefficient: float, fr_indexes: arr.Ints, fr_index_to_name: arr.Strings) -> arr.Strings:
    weight_name = f'w[{pretty_from_fr_indexes(fr_indexes, fr_index_to_name)}]'
    return arr.strings_varargs(weight_name, bas.string_from_float(coefficient))


class Polynomial:
    def __init__(self, ps: PolynomialStructure, coefficients: arr.Floats):
        self.m_poly_structure = ps
        self.m_t_index_to_coefficient = coefficients
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_poly_structure, PolynomialStructure)
        self.m_poly_structure.assert_ok()
        assert isinstance(self.m_t_index_to_coefficient, arr.Floats)
        self.m_t_index_to_coefficient.assert_ok()
        assert self.m_poly_structure.num_terms() == self.m_t_index_to_coefficient.len()

    def account_for_transformer_intervals(self, fr_index_to_interval: bas.Intervals) -> Polynomial:
        result = polynomial_with_no_terms(fr_index_to_interval.len())
        for coefficient, fr_indexes in zip(self.range_coefficients(), self.range_fr_indexes()):
            p = poly_accounting_for_intervals(fr_indexes, fr_index_to_interval)
            p.multiply_by(coefficient)
            result.increment(p)
        return result

    def t_index_from_fr_indexes(self, fr_indexes: arr.Ints) -> int:
        assert fr_indexes.is_weakly_increasing()
        for t_index, fis in enumerate(self.range_fr_indexes()):
            if fis.equals(fr_indexes):
                return t_index
        bas.my_error('fr_indexes not found')

    def range_fr_indexes(self) -> Iterator[arr.Ints]:
        return self.polynomial_structure().range_fr_indexes()

    def add_term(self, coefficient: float, fr_indexes: arr.Ints):
        self.polynomial_structure().add_term(fr_indexes)
        self.m_t_index_to_coefficient.add(coefficient)

    def polynomial_structure(self) -> PolynomialStructure:
        return self.m_poly_structure

    def range_coefficients(self) -> Iterator[float]:
        return self.coefficients().range()

    def coefficients(self) -> arr.Floats:
        return self.m_t_index_to_coefficient

    def increment(self, other: Polynomial):
        for fr_indexes, coefficient in zip(other.range_fr_indexes(), other.range_coefficients()):
            t_index = self.t_index_from_fr_indexes(fr_indexes)
            self.m_t_index_to_coefficient.increment(t_index, coefficient)

    def multiply_by(self, scale: float):
        self.m_t_index_to_coefficient.multiply_by(scale)


def polynomial_with_no_terms(n_fr_indexes: int) -> Polynomial:
    return polynomial_create(polynomial_structure_empty(n_fr_indexes), arr.floats_empty())


def polynomial_linear(n_fr_indexes: int, constant: float, slope: float, fr_index: int) -> Polynomial:
    result = polynomial_constant(n_fr_indexes, constant)
    result.add_term(slope, arr.ints_singleton(fr_index))
    return result


def polynomial_constant(n_fr_indexes, c: float) -> Polynomial:
    result = polynomial_with_no_terms(n_fr_indexes)
    result.add_term(c, arr.ints_empty())
    return result


def polynomial_one(n_fr_indexes: int) -> Polynomial:
    return polynomial_constant(n_fr_indexes, 1.0)


def poly_accounting_for_intervals(fr_indexes: arr.Ints, fr_index_to_interval: bas.Intervals) -> Polynomial:
    n_fr_indexes = fr_index_to_interval.len()
    result = polynomial_one(n_fr_indexes)
    for fr_index in fr_indexes.range():
        iv = fr_index_to_interval.interval(fr_index)
        # multiply result by transformed version of this fr_index
        transformed_version = polynomial_linear(n_fr_indexes, -iv.lo() / iv.width(), 1.0 / iv.width(), fr_index)
        result = result.times(transformed_version)
    return result


def floater_glm_create(ps: PolynomialStructure, ws: GenWeights) -> FloaterGlm:
    return FloaterGlm(ps, ws)


class FloaterClassGlm(lea.FloaterClass):
    def train(self, inputs: noo.FloatRecords, output: dat.Column) -> FloaterGlm:
        pd = poly_data_from_float_records(inputs, self.polynomial_degree())
        ws = gen_weights_from_training(pd.term_records(), output)
        return floater_glm_create(pd.polynomial_structure(), ws)

    def name_as_string(self) -> str:
        pass

    def __init__(self, polynomial_degree: int):
        self.m_polynomial_degree = polynomial_degree
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_polynomial_degree, int)
        assert 0 <= self.m_polynomial_degree < 100

    def polynomial_degree(self) -> int:
        return self.m_polynomial_degree


def model_class_glm(polynomial_degree: int) -> FloaterClassGlm:
    return FloaterClassGlm(polynomial_degree)


def polynomial_create(ps: PolynomialStructure, coefficients: arr.Floats) -> Polynomial:
    return Polynomial(ps, coefficients)


class FloaterGlm(lea.Floater):

    def assert_ok(self):
        assert isinstance(self.m_polynomial_structure, PolynomialStructure)
        self.m_polynomial_structure.assert_ok()
        assert isinstance(self.m_gen_weights, GenWeights)
        self.m_gen_weights.assert_ok()
        assert self.m_polynomial_structure.num_terms() == self.m_gen_weights.num_weight_indexes()

    def pretty_strings(self, td: noo.TransformerDescription) -> arr.Strings:
        intro = self.gen_weights().pretty_strings_intro(td.output_description())
        undecorated = self.gen_weights().pretty_weight_names(self.polynomial_structure(), td)
        decorated = undecorated.decorate('w[', ']')
        weight_values = self.gen_weights().weight_values(self.polynomial_structure(), td.input_intervals())
        many_rows = arr.strings_array_empty()
        for d, w in zip(decorated.range(), weight_values.range()):
            ss = arr.strings_varargs(d, '=', bas.string_from_float(w))
            many_rows.add(ss)

        return intro.with_many(many_rows.pretty_strings())

    def pretty_string(self, td: noo.TransformerDescription) -> str:
        return self.pretty_strings(td).concatenate_fancy('', '\n', '\n')

    def predict_from_float_record(self, fr: noo.FloatRecord) -> dis.Distribution:
        return self.predict_from_term_record(self.term_record_from_float_record(fr))

    def prediction_component_strings(self, output: dat.ColumnDescription) -> arr.Strings:
        return self.gen_weights().prediction_component_strings(output)

    def __init__(self, ps: PolynomialStructure, ws: GenWeights):
        self.m_polynomial_structure = ps
        self.m_gen_weights = ws
        self.assert_ok()

    def gen_weights(self) -> GenWeights:
        return self.m_gen_weights

    def predict_from_term_record(self, tr: TermRecord) -> dis.Distribution:
        return self.gen_weights().predict_from_term_record(tr)

    def polynomial_structure(self) -> PolynomialStructure:
        return self.m_polynomial_structure

    def term_record_from_float_record(self, fr: noo.FloatRecord) -> TermRecord:
        return self.polynomial_structure().term_record_from_float_record(fr)


def gen_weights_logistic_create(weights_as_floats: arr.Floats) -> GenWeightsLogistic:
    return GenWeightsLogistic(weights_as_floats)


def gen_weights_logistic_all_zero(n_terms: int) -> GenWeightsLogistic:
    return gen_weights_logistic_create(arr.floats_all_zero(n_terms))


def gen_weights_all_zero(tvs: TermRecords, output: dat.Column) -> GenWeights:
    ct = output.coltype()
    if ct == dat.Coltype.bools:
        return gen_weights_logistic_all_zero(tvs.num_terms())
    elif ct == dat.Coltype.floats:
        return gen_weights_linear_all_zero(tvs.num_terms())
    elif ct == dat.Coltype.cats:
        return gen_weights_multinomial_all_zero(output.cats().num_values(), tvs.num_terms())
    else:
        bas.my_error('bad coltype')


def gen_weights_from_training(inputs: TermRecords, output: dat.Column) -> GenWeights:
    start_weights = gen_weights_all_zero(inputs, output)
    return start_weights.train(inputs, output)


def polynomial_structure_empty(n_fr_indexes: int) -> PolynomialStructure:
    return PolynomialStructure(n_fr_indexes)


class PolyDataBuilder:
    def __init__(self, n_fr_indexes: int):
        self.m_poly_structure = polynomial_structure_empty(n_fr_indexes)
        self.m_t_index_to_column = arr.floats_array_empty()
        self.assert_ok()

    def column(self, t_index: int) -> arr.Floats:
        return self.m_t_index_to_column.floats(t_index)

    def fr_indexes(self, t_index) -> arr.Ints:
        return self.polynomial_structure().fr_indexes(t_index)

    def columns(self) -> arr.FloatsArray:
        return self.m_t_index_to_column

    def polynomial_structure(self) -> PolynomialStructure:
        return self.m_poly_structure

    def add_term(self, column: arr.Floats, fr_indexes: arr.Ints):
        self.m_poly_structure.add_term(fr_indexes)
        self.m_t_index_to_column.add(column)

    def num_terms(self) -> int:
        return self.m_t_index_to_column.len()

    def contains_fr_indexes(self, fr_indexes: arr.Ints) -> bool:
        return self.polynomial_structure().contains(fr_indexes)

    def assert_ok(self):
        pass


class PolyData:
    def __init__(self, ps: PolynomialStructure, trs: TermRecords):
        self.m_polynomial_structure = ps
        self.m_term_records = trs
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_polynomial_structure, PolynomialStructure)
        self.m_polynomial_structure.assert_ok()
        assert isinstance(self.m_term_records, TermRecords)
        self.m_term_records.assert_ok()
        assert self.m_polynomial_structure.num_terms() == self.m_term_records.num_terms()

    def term_records(self) -> TermRecords:
        return self.m_term_records

    def polynomial_structure(self) -> PolynomialStructure:
        return self.m_polynomial_structure


def poly_data_builder_empty(n_fr_indexes: int) -> PolyDataBuilder:
    return PolyDataBuilder(n_fr_indexes)


def term_records_from_columns(t_index_to_column: arr.FloatsArray) -> TermRecords:
    result = term_records_empty(t_index_to_column.len())
    assert t_index_to_column.len() > 0
    n_rows = t_index_to_column.floats(0).len()
    for row in range(n_rows):
        fs = arr.floats_empty()
        for col in t_index_to_column.range():
            fs.add(col.float(row))
        result.add(term_record_from_floats(fs))
    return result


def poly_data_create(ps: PolynomialStructure, trs: TermRecords) -> PolyData:
    return PolyData(ps, trs)


def poly_data_from_poly_data_builder(pd: PolyDataBuilder) -> PolyData:
    trs = term_records_from_columns(pd.columns())
    return poly_data_create(pd.polynomial_structure(), trs)


def poly_data_from_float_records(frs: noo.FloatRecords, max_degree: int) -> PolyData:
    pdb = poly_data_builder_empty(frs.num_cols())
    pdb.add_term(arr.floats_all_constant(frs.num_rows(), 1.0), arr.ints_empty())
    prev_t_indexes = arr.ints_singleton(0)
    for degree in range(1, max_degree + 1):
        new_t_indexes = arr.ints_empty()
        assert isinstance(prev_t_indexes, arr.Ints)
        for t_index in prev_t_indexes.range():
            previous_column = pdb.column(t_index)
            fr_indexes = pdb.fr_indexes(t_index)
            first_available_new_fr_index = 0 if fr_indexes.len == 0 else fr_indexes.last_element()
            for fr_index in range(first_available_new_fr_index, frs.num_cols()):
                proposed_new_column = previous_column.map_product_with(frs.column_as_floats(t_index))
                if not proposed_new_column.is_loosely_constant():
                    new_t_index = pdb.num_terms()
                    new_fr_indexes = fr_indexes.copied_with_one_element_added(fr_index)
                    assert new_fr_indexes.len() == degree
                    assert new_fr_indexes.is_weakly_increasing()
                    assert not pdb.contains_fr_indexes(new_fr_indexes)
                    pdb.add_term(proposed_new_column, new_fr_indexes)
                    new_t_indexes.add(new_t_index)
        prev_t_indexes = new_t_indexes
        assert isinstance(prev_t_indexes, arr.Ints)

        pdb.assert_ok()
        return poly_data_from_poly_data_builder(pdb)


def floater_glm_from_training(frs: noo.FloatRecords, output: dat.Column, max_degree: int) -> FloaterGlm:
    pd = poly_data_from_float_records(frs, max_degree)
    gws = gen_weights_from_training(pd.term_records(), output)
    return floater_glm_create(pd.polynomial_structure(), gws)


class GenWeights(ABC):
    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def predict_from_term_record(self, tv: TermRecord) -> dis.Distribution:
        pass

    @abstractmethod
    def weight(self, weight_index: int) -> float:
        pass

    def penalty(self) -> float:
        result = 0.0
        for w in self.range():
            result += w * w
        return lea.penalty_parameter * result

    def penalty_derivative(self, weight_index: int) -> float:
        return 2 * lea.penalty_parameter * self.weight(weight_index)

    def penalty_2nd_derivative(self, weight_index: int) -> float:
        assert isinstance(self, GenWeights)
        assert isinstance(weight_index, int)
        return 2 * lea.penalty_parameter

    def range(self) -> Iterator[float]:
        for i in range(self.num_weight_indexes()):
            yield self.weight(i)

    def train(self, inputs: TermRecords, output: dat.Column) -> GenWeights:
        ws = self
        start_ll = ws.loglike(inputs, output)
        ll = start_ll

        iteration = 0
        while True:
            if iteration < 5 or bas.is_power_of_two(iteration):
                print(f'Begin iteration {iteration}')
            ll_old = ll
            for weight_index in range(ws.num_weight_indexes()):
                aaa = ws.loglike(inputs, output)
                assert bas.loosely_equals(aaa, ll)
                bbb = ws.loglike_derivative(inputs, output, weight_index)
                ccc = ws.loglike_2nd_derivative(inputs, output, weight_index)

                # ll = aaa + bbb * h + 0.5 * ccc * h^2
                # h_star = -bbb/ccc
                assert math.fabs(ccc) > 1e-20
                h_star = -bbb / ccc
                assert isinstance(ws, GenWeights)
                ws2 = ws.deep_copy()
                assert isinstance(ws2, GenWeights)
                ws2.increment(weight_index, h_star)
                ll_new2 = ws2.loglike(inputs, output)
                if ll_new2 > ll:
                    print(f"it's worth incrementing weight_index {weight_index} by {h_star} (delta_ll={ll_new2 - ll})")
                    ws = ws2
                    ll = ll_new2

            if math.fabs(ll - ll_old) / (math.fabs(ll - start_ll)) < 1e-5:
                print(f'...finished after {iteration} iterations.')
                return ws

            iteration += 1

    def loglike(self, inputs: TermRecords, output: dat.Column) -> float:
        result = 0.0
        for k, x_k in enumerate(inputs.range()):
            result += self.loglike_from_row(x_k, output, k)
        return result - self.penalty()

    def loglike_derivative(self, inputs: TermRecords, output: dat.Column, weight_index: int) -> float:
        result = 0.0
        for k, x_k in enumerate(inputs.range()):
            result += self.loglike_derivative_from_row(x_k, output, k, weight_index)
        return result - self.penalty_derivative(weight_index)

    def loglike_2nd_derivative(self, inputs: TermRecords, output: dat.Column, weight_index: int) -> float:
        result = 0.0
        for k, x_k in enumerate(inputs.range()):
            result += self.loglike_2nd_derivative_from_row(x_k, output, k, weight_index)
        return result - self.penalty_2nd_derivative(weight_index)

    @abstractmethod
    def loglike_from_row(self, x_k: TermRecord, co: dat.Column, row: int) -> float:
        pass

    @abstractmethod
    def loglike_derivative_from_row(self, x_k: TermRecord, co: dat.Column, row: int, weight_index: int) -> float:
        pass

    @abstractmethod
    def loglike_2nd_derivative_from_row(self, x_k: TermRecord, co: dat.Column, row: int, weight_index: int) -> float:
        pass

    @abstractmethod
    def pretty_strings_intro(self, output: dat.ColumnDescription) -> arr.Strings:
        pass

    @abstractmethod
    def prediction_component_strings(self, output: dat.ColumnDescription) -> arr.Strings:
        pass

    @abstractmethod
    def deep_copy(self) -> GenWeights:
        pass

    @abstractmethod
    def increment(self, weight_index: int, delta: float):
        pass

    @abstractmethod
    def num_weight_indexes(self):
        pass

    @abstractmethod
    def pretty_weight_names(self, ps: PolynomialStructure, td: noo.TransformerDescription) -> arr.Strings:
        pass

    @abstractmethod
    def weight_values(self, param, param1):
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
    def weight_values(self, ps: PolynomialStructure, scaling_intervals: bas.Intervals) -> arr.Floats:
        p = polynomial_create(ps, self.floats())
        return p.account_for_transformer_intervals(scaling_intervals).coefficients()

    def num_weight_indexes(self):
        return self.floats().len()

    def pretty_weight_names(self, ps: PolynomialStructure, tfs: noo.Transformers) -> arr.Strings:
        pass

    def increment(self, weight_index: int, delta: float):
        self.m_weight_index_to_floats.increment(weight_index, delta)

    def deep_copy(self) -> GenWeightsLogistic:
        return gen_weights_logistic_create(self.floats().deep_copy())

    def num_weights(self) -> int:
        return self.m_weight_index_to_floats.len()

    def predict_from_term_record(self, tv: noo.FloatRecord) -> dis.Distribution:
        beta_x = self.floats().dot_product(tv.floats())
        ez = math.exp(beta_x)
        p = ez / (1 + ez)
        return dis.binomial_create(p)

    def weight(self, weight_index: int) -> float:
        return self.floats().float(weight_index)

    def loglike_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int, weight_index: int) -> float:
        assert isinstance(co, dat.ColumnBools)
        q_k = q_from_y(co.bool(row))
        beta_x = self.floats().dot_product(x_k.floats())
        q_beta_x = q_k * beta_x
        ek_of_beta = math.exp(-q_beta_x)
        fk_of_beta = 1 + ek_of_beta
        return -q_k * x_k.float(weight_index) / fk_of_beta

    def pretty_strings_intro(self, output: dat.ColumnDescription) -> arr.Strings:
        intro = f'P({output.colname().string()}|x) = sigmoid(w^T x)'
        result = arr.strings_varargs(intro, '...where...')
        return result

    def prediction_component_strings(self, output: dat.ColumnDescription) -> arr.Strings:
        return arr.strings_singleton(f'p_{output.colname().string()}')

    def __init__(self, weight_index_to_weight: arr.Floats):
        self.m_weight_index_to_floats = weight_index_to_weight
        self.assert_ok()

    def floats(self) -> arr.Floats:
        return self.m_weight_index_to_floats

    def assert_ok(self):
        assert isinstance(self.m_weight_index_to_floats, arr.Floats)

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


class GenWeightsLinear(GenWeights):
    def weight_values(self, ps: PolynomialStructure, scaling_intervals: bas.Intervals) -> arr.Floats:
        p = polynomial_create(ps, self.floats())
        return p.account_for_transformer_intervals(scaling_intervals).coefficients()

    def num_weight_indexes(self):
        return self.floats().len()

    def pretty_weight_names(self, ps: PolynomialStructure, td: noo.TransformerDescription) -> arr.Strings:
        return ps.pretty_weight_names(td.float_record_names())

    def increment(self, weight_index: int, delta: float):
        self.m_weight_index_to_float.increment(weight_index, delta)

    def deep_copy(self) -> GenWeights:
        return gen_weights_linear_create(self.m_weight_index_to_float.deep_copy(), self.sdev())

    def num_weights(self) -> int:
        return self.m_weight_index_to_float.len()

    def assert_ok(self):
        assert isinstance(self.m_weight_index_to_float, arr.Floats)
        self.m_weight_index_to_float.assert_ok()
        assert isinstance(self.m_sdev, float)
        assert self.m_sdev > 0.0

    def predict_from_term_record(self, tv: noo.FloatRecord) -> dis.Distribution:
        beta_x = self.floats().dot_product(tv.floats())
        return dis.gaussian_create(beta_x, self.sdev())

    def weight(self, weight_index: int) -> float:
        return self.floats().float(weight_index)

    def loglike_2nd_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int,
                                        weight_index: int) -> float:
        x_kj = x_k.float(weight_index)
        return -x_kj * x_kj

    def __init__(self, fs: arr.Floats, sdev: float):
        self.m_weight_index_to_float = fs
        self.m_sdev = sdev
        self.assert_ok()

    def floats(self) -> arr.Floats:
        return self.m_weight_index_to_float

    def sdev(self) -> float:
        return self.m_sdev

    def pretty_strings_intro(self, output: dat.ColumnDescription) -> str:
        return f'p({output.colname().string()}|x) ~ Normal(mu = w^T x, sdev={self.sdev()})'

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

    def loglike_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int, weight_index: int) -> float:
        assert isinstance(co, dat.ColumnFloats)
        return x_k.float(weight_index) * (co.float(row) - self.floats().dot_product(x_k.floats()))


def gen_weights_linear_create(weights_as_floats: arr.Floats, sdev: float) -> GenWeightsLinear:
    return GenWeightsLinear(weights_as_floats, sdev)


def gen_weights_linear_all_zero(n_terms: int) -> GenWeightsLinear:
    return gen_weights_linear_create(arr.floats_all_zero(n_terms), 1.0)


class GenWeightsMultinomial(GenWeights):
    def weight_values(self, ps: PolynomialStructure, scaling_intervals: bas.Intervals) -> arr.Floats:
        result = arr.floats_empty()
        for weights in self.fmat().range_rows():
            p = polynomial_create(ps, weights)
            result.append(p.account_for_transformer_intervals(scaling_intervals).coefficients())
        return result

    def pretty_weight_names(self, ps: PolynomialStructure, td: noo.TransformerDescription) -> arr.Strings:
        fr_names = td.float_record_names()
        result = arr.strings_empty()
        for valname, weights in zip(td.output_description().valnames().range(), self.fmat().range_rows()):
            undecorated = ps.pretty_weight_names(fr_names)
            left = f'{valname},'
            decorated = undecorated.decorate(left, '')
            result.append(decorated)
        return result

    def increment(self, weight_index: int, delta: float):
        value, term_num = self.value_and_term_num_from_weight_index(weight_index)
        self.m_value_to_term_num_to_weight.increment(value, term_num, delta)

    def deep_copy(self) -> GenWeights:
        return gen_weights_multinomial_create(self.m_value_to_term_num_to_weight.deep_copy())

    def num_weight_indexes(self) -> int:
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

    def predict_from_term_record(self, tv: noo.FloatRecord) -> dis.Distribution:
        sk = self.sk(tv)
        probs = arr.floats_empty()
        for i in range(self.num_values()):
            eki = self.eki(tv, i)
            probs.add(eki / sk)
        return dis.multinomial_create(probs)

    def weight(self, weight_index: int) -> float:
        value, term_num = self.value_and_term_num_from_weight_index(weight_index)
        return self.floats_from_value(value).float(term_num)

    def loglike_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int) -> float:
        assert isinstance(co, dat.ColumnCats)
        i = co.cats().value(row)
        qki = self.qki(x_k, i)
        sk = self.sk(x_k)
        return qki - math.log(sk)

    def loglike_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int, weight_index: int) -> float:
        weight_index_value, term_num = self.value_and_term_num_from_weight_index(weight_index)
        xkj = x_k.float(term_num)
        pik = self.eki(x_k, weight_index_value) / self.sk(x_k)
        d_ll_k_by_dw_ij = -xkj * pik
        this_rows_value = co.cats().value(row)
        if this_rows_value == weight_index_value:
            d_ll_k_by_dw_ij += xkj
        return d_ll_k_by_dw_ij

    def loglike_2nd_derivative_from_row(self, x_k: noo.FloatRecord, co: dat.Column, row: int,
                                        weight_index: int) -> float:
        weight_index_value, term_num = self.value_and_term_num_from_weight_index(weight_index)
        xkj = x_k.float(term_num)
        pik = self.eki(x_k, weight_index_value) / self.sk(x_k)
        d2_ll_k_by_dw2_ij = -xkj * xkj * pik * (1 - pik)
        return d2_ll_k_by_dw2_ij

    def pretty_strings_intro(self, output: dat.ColumnDescription) -> str:
        return f'P({output.colname().string()}=v|Weights,x) = exp(Weights[v] . x) / K'

    def prediction_component_strings(self, output: dat.ColumnDescription) -> arr.Strings:
        result = arr.strings_empty()
        for vn in output.valnames().range():
            result.add(f'p_{vn.string()}')
        return result

    def __init__(self, value_to_term_num_to_weight: arr.Fmat, ):
        self.m_value_to_term_num_to_weight = value_to_term_num_to_weight
        self.assert_ok()

    def num_values(self) -> int:
        return self.m_value_to_term_num_to_weight.num_rows()

    def floats_from_value(self, value: int) -> arr.Floats:
        return self.fmat().row(value)

    def fmat(self) -> arr.Fmat:
        return self.m_value_to_term_num_to_weight

    def value_and_term_num_from_weight_index(self, weight_index: int) -> Tuple[int, int]:
        term_num = weight_index % self.num_terms()
        value = math.floor(weight_index / self.num_terms()) + 1
        assert isinstance(value, int)
        assert 0 <= term_num < self.num_weight_indexes()
        assert 1 <= value < self.num_values()
        assert weight_index == self.weight_index_from_value_and_term_num(value, term_num)
        return value, term_num

    def weight_index_from_value_and_term_num(self, value: int, term_num: int) -> int:
        assert 1 <= value < self.num_values()
        assert 0 <= term_num < self.num_terms()
        return (value - 1) * self.num_terms() + term_num

    def num_terms(self) -> int:
        return self.fmat().num_cols()


def gen_weights_multinomial_create(weights_as_fmat: arr.Fmat) -> GenWeightsMultinomial:
    return GenWeightsMultinomial(weights_as_fmat)


def gen_weights_multinomial_all_zero(n_values: int, n_terms: int) -> GenWeightsMultinomial:
    return gen_weights_multinomial_create(arr.fmat_of_zeroes(n_values, n_terms))


def unit_test():
    s = """
    a,b,c,y
    0,0,0,50\n
    10,1,0.1,51\n
    0,1,0,50\n
    0,0,0.1,50"""
    ds, ok = dat.datset_from_multiline_string(s)
    assert ok
    output = ds.subset('y')
    inputs = ds.without_column(output.named_column(0))
    td = noo.transformer_description_from_datset(inputs, output.named_column(0))
    nfs = td.input_transformers().named_float_records_from_datset(inputs)
    assert nfs.num_fr_indexes()
