from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Iterator, Tuple, List

import datset.amarrays as arr
import datset.ambasic as bas
import datset.distribution as dis
import datset.dset as dat
import datset.learn as lea
import datset.numset as noo
import datset.piclass as pic
from datset.learn import Floater


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

    def float(self, t_index: int) -> float:
        return self.floats().float2(t_index)


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


def term_record_from_floats(fs: arr.Floats) -> TermRecord:
    return TermRecord(fs)


class Factor:
    def __init__(self, cov_id: int, power: int):
        self.m_cov_id = cov_id
        self.m_power = power
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_cov_id, int)
        assert isinstance(self.m_power, int)
        assert 0 <= self.m_cov_id < 1000000
        assert 0 <= self.m_power < 100

    def cov_id(self) -> int:
        return self.m_cov_id

    def power(self) -> int:
        return self.m_power

    def deep_copy(self) -> Factor:
        return factor_create(self.cov_id(), self.power())

    def evaluate(self, fr: noo.FloatRecord) -> float:
        x = fr.float(self.cov_id())
        result = 1.0
        for i in range(self.power()):
            result *= x
        return result

    def pretty_string(self, covariate_names: arr.Strings) -> str:
        s = covariate_names.string(self.cov_id())
        return s if self.power() == 1 else f'{s}^{self.power()}'

    def string(self) -> str:
        s = f'x{self.cov_id()}'
        return s if self.power() == 1 else f'{s}^{self.power()}'

    def equals(self, other: Factor) -> bool:
        return self.cov_id() == other.cov_id() and self.power() == other.power()


def term_empty() -> Term:
    return Term([])


class Term:
    def __init__(self, ps: List[Factor]):
        self.m_list = ps
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for p in self.m_list:
            assert isinstance(p, Factor)
            p.assert_ok()
        assert self.cov_ids().is_strictly_increasing()

    def cov_ids(self) -> arr.Ints:
        result = arr.ints_empty()
        for p in self.range():
            result.add(p.cov_id())
        return result

    def len(self) -> int:
        return len(self.m_list)

    def max_cov_id(self) -> int:
        assert self.len() > 0
        return self.last_factor().cov_id()

    def last_factor(self) -> Factor:
        assert self.len() > 0
        return self.factor(self.len() - 1)

    def factor(self, index: int) -> Factor:
        assert 0 <= index < self.len()
        return self.m_list[index]

    def deep_copy(self) -> Term:
        result = term_empty()
        for f in self.range():
            result.add(f.deep_copy())
        return result

    def range(self) -> Iterator[Factor]:
        for p in self.m_list:
            yield p

    def evaluate(self, fr: noo.FloatRecord) -> float:
        result = 1.0
        for fct in self.range():
            result *= fct.evaluate(fr)
        return result

    def pretty_string(self, covariate_names: arr.Strings) -> str:
        if self.len() == 0:
            return 'constant'
        else:
            result = arr.strings_empty()
            for p in self.range():
                result.add(p.pretty_string(covariate_names))
            return result.concatenate_fancy('', ' * ', '')

    def poly_accounting_for_intervals(self, cov_id_to_interval: bas.Intervals) -> Polynomial:
        n_covariates = cov_id_to_interval.len()
        result = polynomial_one(n_covariates)
        for fct in self.range():
            iv = cov_id_to_interval.interval(fct.cov_id())
            transformed_version = polynomial_linear(n_covariates, -iv.lo() / iv.width(), 1.0 / iv.width(), fct.cov_id())
            for i in range(fct.power()):
                result = result.times_polynomial(transformed_version)
        return result

    def add(self, p: Factor):
        if self.len() > 0:
            assert self.max_cov_id() < p.cov_id()
        self.m_list.append(p)

    def times_term(self, other: Term) -> Term:
        result = term_empty()
        my_index = 0
        other_index = 0
        while True:
            if my_index == self.len():
                for i in range(other_index, other.len()):
                    result.add(other.factor(i))
                return result
            elif other_index == other.len():
                for i in range(my_index, self.len()):
                    result.add(self.factor(i))
                return result
            else:
                my_factor = self.factor(my_index)
                other_factor = other.factor(other_index)
                if my_factor.cov_id() < other_factor.cov_id():
                    result.add(my_factor)
                    my_index += 1
                elif other_factor.cov_id() < my_factor.cov_id():
                    result.add(other_factor)
                    other_index += 1
                else:
                    new_power = my_factor.power() + other_factor.power()
                    new_pow = factor_create(my_factor.cov_id(), new_power)
                    result.add(new_pow)
                    my_index += 1
                    other_index += 1

    def last_cov_id(self) -> int:
        return self.last_factor().cov_id()

    def times_cov_id(self, cov_id: int) -> Term:
        result = term_empty()
        already_added = False
        for f in self.range():
            if already_added or f.cov_id() < cov_id:
                result.add(f)
            elif f.cov_id() == cov_id:
                result.add(factor_create(cov_id, f.power() + 1))
                already_added = True
            else:
                result.add(factor_create(cov_id, 1))
                result.add(f)
                already_added = True

        if not already_added:
            result.add(factor_create(cov_id, 1))
        result.assert_ok()
        assert result.degree() == self.degree() + 1
        return result

    def degree(self) -> int:
        result = 0
        for fct in self.range():
            result += fct.power()
        return result

    def string(self) -> str:
        if self.len() == 0:
            return '1'
        return self.strings().concatenate_fancy('', ' * ', '')

    def strings(self) -> arr.Strings:
        result = arr.strings_empty()
        for fct in self.range():
            result.add(fct.string())
        return result

    def equals(self, other: Term) -> bool:
        if self.len() != other.len():
            return False

        for f_me, f_other in zip(self.range(), other.range()):
            if not f_me.equals(f_other):
                return False

        return True


class Terms:
    def __init__(self, li: List[Term]):
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for t in self.m_list:
            assert isinstance(t, Term)
            t.assert_ok()

    def num_terms(self) -> int:
        return len(self.m_list)

    def range(self) -> Iterator[Term]:
        for t in self.m_list:
            yield t

    def add(self, tm: Term):
        self.m_list.append(tm)

    def term(self, t_index: int) -> Term:
        assert 0 <= t_index < self.num_terms()
        return self.m_list[t_index]


def terms_empty() -> Terms:
    return Terms([])


class PolynomialStructure:
    def __init__(self, n_covariates: int):
        self.m_num_covariates = n_covariates
        self.m_t_index_to_term = terms_empty()
        self.assert_ok()

    def pretty_weight_names(self, covariate_names: arr.Strings) -> arr.Strings:
        result = arr.strings_empty()
        for tm in self.range_terms():
            result.add(tm.pretty_string(covariate_names))
        return result

    def assert_ok(self):
        assert isinstance(self.m_num_covariates, int)

        for t_index, t in enumerate(self.m_t_index_to_term.range()):
            if t.len() > 0:
                assert t.max_cov_id() < self.m_num_covariates
            for t_index_prime in range(t_index + 1, self.m_t_index_to_term.num_terms()):
                assert not t.equals(self.term(t_index_prime))

    def term_records_from_float_records(self, frs: noo.FloatRecords) -> TermRecords:
        assert self.num_covariates() == frs.num_cols()
        result = term_records_empty(self.num_terms())
        for fr in frs.range():
            result.add(self.term_record_from_float_record(fr))
        return result

    def t_index_from_term(self, target: Term) -> Tuple[int, bool]:
        for t_index, tm in enumerate(self.range_terms()):
            if tm.equals(target):
                assert self.term(t_index).equals(target)
                return t_index, True
        return -777, False

    def num_covariates(self) -> int:
        return self.m_num_covariates

    def num_terms(self) -> int:
        return self.m_t_index_to_term.num_terms()

    def term_record_from_float_record(self, fr: noo.FloatRecord) -> TermRecord:
        result = arr.floats_empty(self.num_terms())
        for tm in self.range_terms():
            result.add(tm.evaluate(fr))
        return term_record_from_floats(result)

    def range_terms(self) -> Iterator[Term]:
        return self.m_t_index_to_term.range()

    def add_term(self, tm: Term):
        if tm.len() > 0:
            assert tm.max_cov_id() < self.num_covariates()
        self.m_t_index_to_term.add(tm)

    def term(self, t_index: int) -> Term:
        return self.m_t_index_to_term.term(t_index)

    def deep_copy(self) -> PolynomialStructure:
        result = polynomial_structure_empty(self.num_covariates())
        for tm in self.range_terms():
            result.add_term(tm.deep_copy())
        return result

    def contains(self, tm: Term) -> bool:
        t_index, ok = self.t_index_from_term(tm)
        return ok

    def equals(self, other: PolynomialStructure) -> bool:
        for t in self.range_terms():
            if not other.contains(t):
                return False

        for t in other.range_terms():
            if not self.contains(t):
                return False

        return True


def pretty_from_coefficient(coefficient: float, tm: Term, cov_id_to_name: arr.Strings) -> arr.Strings:
    weight_name = f'w[{tm.pretty_string(cov_id_to_name)}]'
    return arr.strings_varargs(weight_name, '=', bas.string_from_float(coefficient))


class CoefficientTerm:
    def __init__(self, coefficient: float, tm: Term):
        self.m_coefficient = coefficient
        self.m_term = tm
        self.assert_ok()

    def term(self) -> Term:
        return self.m_term

    def assert_ok(self):
        assert isinstance(self.m_coefficient, float)
        assert isinstance(self.m_term, Term)
        self.m_term.assert_ok()

    def coefficient(self) -> float:
        return self.m_coefficient

    def increment_coefficient(self, delta: float):
        self.m_coefficient += delta

    def times_scalar(self, scale: float) -> CoefficientTerm:
        return coefficient_term_create(self.coefficient() * scale, self.term())

    def deep_copy(self) -> CoefficientTerm:
        return coefficient_term_create(self.coefficient(), self.term().deep_copy())


class CoefficientTerms:
    def __init__(self, li: List[CoefficientTerm]):
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for t_index, ct in enumerate(self.m_list):
            assert isinstance(ct, CoefficientTerm)
            ct.assert_ok()
            for ti2 in range(t_index + 1, self.len()):
                assert not ct.term().equals(self.term(ti2))

    def len(self) -> int:
        return len(self.m_list)

    def range(self) -> Iterator[CoefficientTerm]:
        for ct in self.m_list:
            yield ct

    def add(self, ct: CoefficientTerm):
        self.m_list.append(ct)

    def term(self, t_index: int) -> Term:
        return self.coefficient_term(t_index).term()

    def coefficient_term(self, t_index: int) -> CoefficientTerm:
        assert 0 <= t_index < self.len()
        return self.m_list[t_index]

    def increment_coefficient(self, t_index: int, delta: float):
        self.coefficient_term(t_index).increment_coefficient(delta)

    def deep_copy(self) -> CoefficientTerms:
        result = coefficient_terms_empty()
        for ct in self.range():
            result.add(ct.deep_copy())
        return result


def coefficient_term_create(coefficient: float, tm: Term) -> CoefficientTerm:
    return CoefficientTerm(coefficient, tm)


class Polynomial:
    def __init__(self, n_covariates: int, cts: CoefficientTerms):
        self.m_num_covariates = n_covariates
        self.m_t_index_to_coefficient_term = cts
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_t_index_to_coefficient_term, CoefficientTerms)
        self.m_t_index_to_coefficient_term.assert_ok()
        assert isinstance(self.m_num_covariates, int)
        for tm in self.range_terms():
            assert tm.len() == 0 or tm.max_cov_id() < self.num_covariates()

    def account_for_transformer_intervals(self, cov_id_to_interval: bas.Intervals) -> Polynomial:
        result = polynomial_with_no_terms(cov_id_to_interval.len())
        for coefficient, tm in zip(self.range_coefficients(), self.range_terms()):
            p = tm.poly_accounting_for_intervals(cov_id_to_interval)
            result = result.plus(p.times_scalar(coefficient))
        result.assert_ok()
        return result

    def t_index_from_term(self, target: Term) -> Tuple[int, bool]:
        for t_index, tm in enumerate(self.range_terms()):
            if tm.equals(target):
                return t_index, True
        return -777, False

    def range_terms(self) -> Iterator[Term]:
        for ct in self.range():
            yield ct.term()

    def add_term(self, coefficient: float, tm: Term):
        self.m_t_index_to_coefficient_term.add(coefficient_term_create(coefficient, tm))

    def polynomial_structure_slow(self) -> PolynomialStructure:
        result = polynomial_structure_empty(self.num_covariates())
        for t in self.range_terms():
            result.add_term(t)
        return result

    def range_coefficients(self) -> Iterator[float]:
        for ct in self.range():
            yield ct.coefficient()

    def coefficients_slow(self) -> arr.Floats:
        result = arr.floats_empty(self.num_terms())
        for ct in self.range():
            result.add(ct.coefficient())
        return result

    def plus(self, other: Polynomial) -> Polynomial:
        result = self.deep_copy()
        for c, t in zip(other.range_coefficients(), other.range_terms()):
            result_t_index, ok = result.t_index_from_term(t)
            if ok:
                result.increment_coefficient(result_t_index, c)
            else:
                result.add_term(c, t)
        return result

    def times_scalar(self, scale: float) -> Polynomial:
        result = polynomial_with_no_terms(self.num_covariates())
        for ct in self.range():
            result.add(ct.times_scalar(scale))
        return result

    def increment_coefficient(self, t_index: int, delta: float):
        self.m_t_index_to_coefficient_term.increment_coefficient(t_index, delta)

    def times_polynomial(self, other: Polynomial) -> Polynomial:
        result = polynomial_with_no_terms(self.num_covariates())
        for my_coefficient, my_term in zip(self.range_coefficients(), self.range_terms()):
            result = result.plus(other.times_term(my_term).times_scalar(my_coefficient))
        result.assert_ok()
        return result

    def deep_copy(self) -> Polynomial:
        return polynomial_create(self.num_covariates(), self.coefficient_terms().deep_copy())

    def times_term(self, tm: Term) -> Polynomial:
        result = polynomial_with_no_terms(self.num_covariates())
        for c, t in zip(self.range_coefficients(), self.range_terms()):
            result.add_term(c, tm.times_term(t))
        return result

    def num_covariates(self) -> int:
        return self.m_num_covariates

    def pretty_string(self, covariate_names: arr.Strings) -> str:
        return self.pretty_strings(covariate_names).concatenate_fancy('', '\n', '')

    def pretty_strings(self, covariate_names: arr.Strings) -> arr.Strings:
        return self.pretty_strings_array(covariate_names).pretty_strings()

    def pretty_strings_array(self, covariate_names: arr.Strings) -> arr.StringsArray:
        coefficient_strings = self.coefficients_slow().strings()
        term_strings = self.polynomial_structure_slow().pretty_weight_names(covariate_names)
        assert coefficient_strings.len() == term_strings.len()
        result = arr.strings_array_empty()
        for t_index in range(coefficient_strings.len()):
            c = coefficient_strings.string(t_index)
            t = term_strings.string(t_index)

            possible_times = '*' if self.term(t_index).len() > 0 else ''
            possible_term = t if self.term(t_index).len() > 0 else ''
            possible_plus = '+' if t_index < self.num_terms() - 1 else ''
            result.add(arr.strings_varargs(c, possible_times, possible_term, possible_plus))
        return result

    def string(self) -> str:
        return self.strings().concatenate_fancy('', ' + ', '')

    def strings(self) -> arr.Strings:
        result = arr.strings_empty()
        for c, t in zip(self.range_coefficients(), self.range_terms()):
            if math.fabs(c) > 1e-8:
                s = bas.string_from_float(c) if t.len() == 0 else f'{c} * {t.string()}'
                result.add(s)
        if result.len() == 0:
            result.add('0')
        return result

    def num_terms(self) -> int:
        return self.coefficient_terms().len()

    def range(self) -> Iterator[CoefficientTerm]:
        return self.m_t_index_to_coefficient_term.range()

    def add(self, ct: CoefficientTerm):
        self.m_t_index_to_coefficient_term.add(ct)

    def term(self, t_index: int) -> Term:
        return self.coefficient_term(t_index).term()

    def coefficient_term(self, t_index: int) -> CoefficientTerm:
        return self.m_t_index_to_coefficient_term.coefficient_term(t_index)

    def coefficient_terms(self) -> CoefficientTerms:
        return self.m_t_index_to_coefficient_term


def coefficient_terms_create(li: List[CoefficientTerm]) -> CoefficientTerms:
    return CoefficientTerms(li)


def coefficient_terms_empty() -> CoefficientTerms:
    return coefficient_terms_create([])


def polynomial_with_no_terms(n_covariates: int) -> Polynomial:
    return polynomial_create(n_covariates, coefficient_terms_empty())


def term_with_single_power(p: Factor) -> Term:
    result = term_empty()
    result.add(p)
    return result


def factor_create(cov_id: int, power: int) -> Factor:
    return Factor(cov_id, power)


def term_with_single_linear_factor(cov_id: int) -> Term:
    return term_with_single_power(factor_create(cov_id, 1))


def polynomial_linear(n_covariates: int, constant_coefficient: float, slope: float, cov_id: int) -> Polynomial:
    result = polynomial_constant(n_covariates, constant_coefficient)
    result.add_term(slope, term_with_single_linear_factor(cov_id))
    return result


def polynomial_constant(n_covariates, c: float) -> Polynomial:
    result = polynomial_with_no_terms(n_covariates)
    result.add_term(c, term_empty())
    return result


def polynomial_one(n_covariates: int) -> Polynomial:
    return polynomial_constant(n_covariates, 1.0)


def floater_glm_create(ps: PolynomialStructure, ws: GenWeights) -> FloaterGlm:
    ws.assert_ok()
    return FloaterGlm(ps, ws)


class FloaterClassGlm(lea.FloaterClass):
    def pretty_strings(self) -> arr.Strings:
        result = arr.strings_singleton(f'polynomial GLM of degree {self.polynomial_degree()}')
        return result

    def train(self, inputs: noo.FloatRecords, output: dat.Column) -> FloaterGlm:
        pd = poly_data_from_float_records(inputs, self.polynomial_degree())
        ws = gen_weights_from_training(pd.term_records(), output)
        ws.assert_ok()
        return floater_glm_create(pd.polynomial_structure(), ws)

    def name_as_string(self) -> str:
        return f'poly({self.polynomial_degree()})'

    def __init__(self, polynomial_degree: int):
        self.m_polynomial_degree = polynomial_degree
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_polynomial_degree, int)
        assert 0 <= self.m_polynomial_degree < 100

    def polynomial_degree(self) -> int:
        return self.m_polynomial_degree


def floater_class_glm(polynomial_degree: int) -> FloaterClassGlm:
    return FloaterClassGlm(polynomial_degree)


def model_class_glm(polynomial_degree: int) -> lea.ModelClassFloater:
    return lea.model_class_from_floater_class(floater_class_glm(polynomial_degree))


def polynomial_create(n_covariates: int, cts: CoefficientTerms) -> Polynomial:
    return Polynomial(n_covariates, cts)


class FloaterGlm(lea.Floater):

    def distribution_description(self):
        return self.gen_weights().distribution_description()

    def loosely_equals(self, other: Floater) -> bool:
        if not isinstance(other, FloaterGlm):
            return False

        assert isinstance(other, FloaterGlm)

        if not self.polynomial_structure().equals(other.polynomial_structure()):
            return False

        if not self.gen_weights().loosely_equals(other.gen_weights()):
            return False

        return True

    def assert_ok(self):
        assert isinstance(self.m_polynomial_structure, PolynomialStructure)
        self.m_polynomial_structure.assert_ok()
        assert isinstance(self.m_gen_weights, GenWeights)
        self.m_gen_weights.assert_ok()

    def pretty_strings(self, td: noo.TransformerDescription) -> arr.Strings:
        intro = self.gen_weights().pretty_strings_intro(td.output_description())
        assert isinstance(intro, arr.Strings)
        undecorated = self.gen_weights().pretty_weight_names(self.polynomial_structure(), td)
        assert isinstance(undecorated, arr.Strings)
        decorated = undecorated.decorate('w[', ']')
        weight_values = self.gen_weights().weight_values(self.polynomial_structure(), td.input_intervals())
        many_rows = arr.strings_array_empty()
        for d, w in zip(decorated.range(), weight_values.range2()):
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
        return gen_weights_multinomial_all_zero(output.cats().valnames(), tvs.num_terms())
    else:
        bas.my_error('bad coltype')


def gen_weights_from_training(inputs: TermRecords, output: dat.Column) -> GenWeights:
    start_weights = gen_weights_all_zero(inputs, output)
    return start_weights.train(inputs, output)


def polynomial_structure_empty(n_covariates: int) -> PolynomialStructure:
    return PolynomialStructure(n_covariates)


class PolyDataBuilder:
    def __init__(self, n_covariates: int):
        self.m_poly_structure = polynomial_structure_empty(n_covariates)
        self.m_t_index_to_column = arr.floats_array_empty()
        self.assert_ok()

    def column(self, t_index: int) -> arr.Floats:
        return self.m_t_index_to_column.floats(t_index)

    def term(self, t_index) -> Term:
        return self.polynomial_structure().term(t_index)

    def columns(self) -> arr.FloatsArray:
        return self.m_t_index_to_column

    def polynomial_structure(self) -> PolynomialStructure:
        return self.m_poly_structure

    def add_term(self, column: arr.Floats, tm: Term):
        self.m_poly_structure.add_term(tm)
        self.m_t_index_to_column.add(column)

    def num_terms(self) -> int:
        return self.m_t_index_to_column.len()

    def contains_term(self, tm: Term) -> bool:
        return self.polynomial_structure().contains(tm)

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


def poly_data_builder_empty(n_covariates: int) -> PolyDataBuilder:
    return PolyDataBuilder(n_covariates)


def term_records_from_columns(t_index_to_column: arr.FloatsArray) -> TermRecords:
    n_terms = t_index_to_column.len()
    result = term_records_empty(n_terms)
    assert n_terms > 0
    n_rows = t_index_to_column.floats(0).len()
    for row in range(n_rows):
        fs = arr.floats_empty(n_terms)
        for col in t_index_to_column.range():
            fs.add(col.float2(row))
        result.add(term_record_from_floats(fs))
    return result


def poly_data_create(ps: PolynomialStructure, trs: TermRecords) -> PolyData:
    return PolyData(ps, trs)


def poly_data_from_poly_data_builder(pd: PolyDataBuilder) -> PolyData:
    trs = term_records_from_columns(pd.columns())
    return poly_data_create(pd.polynomial_structure(), trs)


def poly_data_from_float_records(frs: noo.FloatRecords, max_degree: int) -> PolyData:
    assert isinstance(frs, noo.FloatRecords)
    pdb = poly_data_builder_empty(frs.num_cols())
    pdb.add_term(arr.floats_all_constant(frs.num_rows(), 1.0), term_empty())
    prev_t_indexes = arr.ints_singleton(0)
    for degree in range(1, max_degree + 1):
        new_t_indexes = arr.ints_empty()
        assert isinstance(prev_t_indexes, arr.Ints)
        for t_index in prev_t_indexes.range():
            previous_column = pdb.column(t_index)
            tm = pdb.term(t_index)
            first_available_new_cov_id = 0 if tm.len() == 0 else tm.last_cov_id()
            for cov_id in range(first_available_new_cov_id, frs.num_cols()):
                proposed_new_column = previous_column.map_product_with(frs.column_as_floats(cov_id))
                if not proposed_new_column.is_loosely_constant():
                    new_t_index = pdb.num_terms()
                    new_term = tm.times_cov_id(cov_id)
                    assert new_term.degree() == degree
                    assert not pdb.contains_term(new_term)
                    pdb.add_term(proposed_new_column, new_term)
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
                    ws = ws2
                    ll = ll_new2

            if math.fabs(ll - ll_old) <= 1e-20 + (math.fabs(ll - start_ll)) * 1e-5:
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
            assert isinstance(x_k, TermRecord)
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
    def weight_values(self, ps: PolynomialStructure, inputs_intervals: bas.Intervals) -> arr.Floats:
        pass

    @abstractmethod
    def loosely_equals(self, other) -> bool:
        pass

    @abstractmethod
    def distribution_description(self) -> dis.DistributionDescription:
        pass


def q_from_y(y_k: bool) -> float:
    return -1.0 if y_k else 1.0


def pretty_strings_from_weights(weight_names: arr.Strings, weights: arr.Floats) -> arr.Strings:
    assert weight_names.len() == weights.len()
    result = arr.strings_array_empty()
    for name, weight in zip(weight_names.range(), weights.range2()):
        result.add(arr.strings_varargs(f'w[{name}]', '=', bas.string_from_float(weight)))
    return result.pretty_strings()


def coefficient_terms_from_polynomial_structure(ps: PolynomialStructure, coefficients: arr.Floats) -> CoefficientTerms:
    result = coefficient_terms_empty()
    for t, coefficient in zip(ps.range_terms(), coefficients.range2()):
        ct = coefficient_term_create(coefficient, t)
        result.add(ct)
    return result


class GenWeightsLogistic(GenWeights):
    def distribution_description(self) -> dis.DistributionDescription:
        return dis.distribution_description_binomial()

    def loosely_equals(self, other: GenWeights) -> bool:
        if not isinstance(other, GenWeightsLogistic):
            return False

        assert isinstance(other, GenWeightsLogistic)
        return self.m_weight_index_to_floats.loosely_equals(other.m_weight_index_to_floats)

    def weight_values(self, ps: PolynomialStructure, scaling_intervals: bas.Intervals) -> arr.Floats:
        cts = coefficient_terms_from_polynomial_structure(ps, self.floats())
        p = polynomial_create(ps.num_covariates(), cts)
        return p.account_for_transformer_intervals(scaling_intervals).coefficients_slow()

    def num_weight_indexes(self):
        return self.floats().len()

    def pretty_weight_names(self, ps: PolynomialStructure, td: noo.TransformerDescription) -> arr.Strings:
        return ps.pretty_weight_names(td.covariate_names())

    def increment(self, weight_index: int, delta: float):
        self.m_weight_index_to_floats.increment2(weight_index, delta)

    def deep_copy(self) -> GenWeightsLogistic:
        return gen_weights_logistic_create(self.floats().deep_copy())

    def num_weights(self) -> int:
        return self.m_weight_index_to_floats.len()

    def predict_from_term_record(self, tr: TermRecord) -> dis.Distribution:
        beta_x = self.floats().dot_product(tr.floats())
        ez = math.exp(beta_x)
        p = ez / (1 + ez)
        return dis.binomial_create(p)

    def weight(self, weight_index: int) -> float:
        return self.floats().float2(weight_index)

    def loglike_derivative_from_row(self, x_k: TermRecord, co: dat.Column, row: int, weight_index: int) -> float:
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

    def loglike_2nd_derivative_from_row(self, x_k: TermRecord, co: dat.Column, row: int, wx: int) -> float:
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

    def loglike_from_row(self, x_k: TermRecord, co: dat.ColumnBools, row: int) -> float:
        assert isinstance(co, dat.ColumnBools)
        q = q_from_y(co.bool(row))
        q_beta_x = q * self.floats().dot_product(x_k.floats())
        ek_of_minus_beta = math.exp(q_beta_x)  # negative of negative is positive
        fk_of_minus_beta = 1 + ek_of_minus_beta
        return -math.log(fk_of_minus_beta)


def polynomial_from_polynomial_structure(ps: PolynomialStructure, coefficients: arr.Floats) -> Polynomial:
    cts = coefficient_terms_from_polynomial_structure(ps, coefficients)
    return polynomial_create(ps.num_covariates(), cts)


class GenWeightsLinear(GenWeights):
    def distribution_description(self) -> dis.DistributionDescription:
        return dis.distribution_description_gaussian()

    def loosely_equals(self, other: GenWeights) -> bool:
        if not isinstance(other, GenWeightsLinear):
            return False
        assert isinstance(other, GenWeightsLinear)
        return bas.loosely_equals(self.m_sdev, other.m_sdev) and self.m_weight_index_to_float.loosely_equals(
            other.m_weight_index_to_float)

    def weight_values(self, ps: PolynomialStructure, scaling_intervals: bas.Intervals) -> arr.Floats:
        p = polynomial_from_polynomial_structure(ps, self.floats())
        return p.account_for_transformer_intervals(scaling_intervals).coefficients_slow()

    def num_weight_indexes(self):
        return self.floats().len()

    def pretty_weight_names(self, ps: PolynomialStructure, td: noo.TransformerDescription) -> arr.Strings:
        return ps.pretty_weight_names(td.covariate_names())

    def increment(self, weight_index: int, delta: float):
        self.m_weight_index_to_float.increment2(weight_index, delta)

    def deep_copy(self) -> GenWeights:
        return gen_weights_linear_create(self.m_weight_index_to_float.deep_copy(), self.sdev())

    def num_weights(self) -> int:
        return self.m_weight_index_to_float.len()

    def assert_ok(self):
        assert isinstance(self.m_weight_index_to_float, arr.Floats)
        self.m_weight_index_to_float.assert_ok()
        assert isinstance(self.m_sdev, float)
        assert self.m_sdev > 0.0

    def predict_from_term_record(self, tv: TermRecord) -> dis.Distribution:
        beta_x = self.floats().dot_product(tv.floats())
        return dis.gaussian_create(beta_x, self.sdev())

    def weight(self, weight_index: int) -> float:
        return self.floats().float2(weight_index)

    def loglike_2nd_derivative_from_row(self, x_k: TermRecord, co: dat.Column, row: int,
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

    def pretty_strings_intro(self, output: dat.ColumnDescription) -> arr.Strings:
        s = f'p({output.colname().string()}|x) ~ Normal(mu = w^T x, sdev={self.sdev()})'
        return arr.strings_singleton(s)

    def prediction_component_strings(self, output: dat.ColumnDescription) -> arr.Strings:
        result = arr.strings_singleton(output.colname().string())
        result.add('sdev')
        return result

    def loglike_from_row(self, x_k: TermRecord, co: dat.Column, row: int) -> float:
        assert isinstance(co, dat.ColumnFloats)
        correct = co.float(row)
        assert isinstance(correct, float)
        beta_x = self.floats().dot_product(x_k.floats())
        assert isinstance(beta_x, float)
        delta = correct - beta_x
        return -0.5 * delta * delta - bas.log_root_two_pi

    def loglike_derivative_from_row(self, x_k: TermRecord, co: dat.Column, row: int, weight_index: int) -> float:
        assert isinstance(co, dat.ColumnFloats)
        return x_k.float(weight_index) * (co.float(row) - self.floats().dot_product(x_k.floats()))


def gen_weights_linear_create(weights_as_floats: arr.Floats, sdev: float) -> GenWeightsLinear:
    return GenWeightsLinear(weights_as_floats, sdev)


def gen_weights_linear_all_zero(n_terms: int) -> GenWeightsLinear:
    return gen_weights_linear_create(arr.floats_all_zero(n_terms), 1.0)


class GenWeightsMultinomial(GenWeights):
    def distribution_description(self) -> dis.DistributionDescription:
        return dis.distribution_description_multinomial(self.valnames())

    def loosely_equals(self, other: GenWeights) -> bool:
        if not isinstance(other, GenWeightsMultinomial):
            return False
        assert isinstance(other, GenWeightsMultinomial)

        if not self.m_value_to_term_num_to_weight.loosely_equals(other.m_value_to_term_num_to_weight):
            return False

        if not self.m_valnames.equals(other.valnames()):
            return False

        return True

    def weight_values(self, ps: PolynomialStructure, scaling_intervals: bas.Intervals) -> arr.Floats:
        result = arr.floats_empty(self.num_values() * self.num_terms())
        for weights in self.fmat().range_rows():
            p = polynomial_from_polynomial_structure(ps, weights)
            coefficients = p.account_for_transformer_intervals(scaling_intervals).coefficients_slow()
            for c in coefficients.range2():
                result.add(c)
        assert result.is_filled()
        return result

    def pretty_weight_names(self, ps: PolynomialStructure, td: noo.TransformerDescription) -> arr.Strings:
        fr_names = td.covariate_names()
        result = arr.strings_empty()
        for valname, weights in zip(td.output_description().valnames().range(), self.fmat().range_rows()):
            undecorated = ps.pretty_weight_names(fr_names)
            left = f'{valname.string()},'
            decorated = undecorated.decorate(left, '')
            result.append(decorated)
        return result

    def increment(self, weight_index: int, delta: float):
        value, term_num = self.value_and_term_num_from_weight_index(weight_index)
        self.m_value_to_term_num_to_weight.increment(value, term_num, delta)

    def deep_copy(self) -> GenWeights:
        return gen_weights_multinomial_create(self.m_valnames.deep_copy(),
                                              self.m_value_to_term_num_to_weight.deep_copy())

    def num_weight_indexes(self) -> int:
        return self.num_terms() * (self.num_values() - 1)

    def assert_ok(self):
        assert isinstance(self.m_valnames, dat.Valnames)
        self.m_valnames.assert_ok()
        fm = self.m_value_to_term_num_to_weight
        assert isinstance(fm, arr.Fmat)
        fm.assert_ok()
        assert fm.num_rows() > 0
        assert fm.row(0).loosely_equals(arr.floats_all_zero(fm.num_cols()))
        assert self.m_valnames.len() == fm.num_rows()

    def eki(self, z: TermRecord, i: int) -> float:
        return math.exp(self.qki(z, i))

    def qki(self, x_k: TermRecord, i: int) -> float:
        return self.floats_from_value(i).dot_product(x_k.floats())

    def sk(self, z: TermRecord) -> float:
        result = 0
        for i in range(self.num_values()):
            result += self.eki(z, i)
        return result

    def predict_from_term_record(self, tv: TermRecord) -> dis.Distribution:
        sk = self.sk(tv)
        n_values = self.num_values()
        probs = arr.floats_empty(n_values)
        for i in range(n_values):
            eki = self.eki(tv, i)
            probs.add(eki / sk)
        return dis.multinomial_create(self.valnames(), probs)

    def weight(self, weight_index: int) -> float:
        value, term_num = self.value_and_term_num_from_weight_index(weight_index)
        return self.floats_from_value(value).float2(term_num)

    def loglike_from_row(self, x_k: TermRecord, co: dat.Column, row: int) -> float:
        assert isinstance(co, dat.ColumnCats)
        i = co.cats().value(row)
        qki = self.qki(x_k, i)
        sk = self.sk(x_k)
        return qki - math.log(sk)

    def loglike_derivative_from_row(self, x_k: TermRecord, co: dat.Column, row: int, weight_index: int) -> float:
        weight_index_value, term_num = self.value_and_term_num_from_weight_index(weight_index)
        xkj = x_k.float(term_num)
        pik = self.eki(x_k, weight_index_value) / self.sk(x_k)
        d_ll_k_by_dw_ij = -xkj * pik
        this_rows_value = co.cats().value(row)
        if this_rows_value == weight_index_value:
            d_ll_k_by_dw_ij += xkj
        return d_ll_k_by_dw_ij

    def loglike_2nd_derivative_from_row(self, x_k: TermRecord, co: dat.Column, row: int,
                                        weight_index: int) -> float:
        weight_index_value, term_num = self.value_and_term_num_from_weight_index(weight_index)
        xkj = x_k.float(term_num)
        pik = self.eki(x_k, weight_index_value) / self.sk(x_k)
        d2_ll_k_by_dw2_ij = -xkj * xkj * pik * (1 - pik)
        return d2_ll_k_by_dw2_ij

    def pretty_strings_intro(self, output: dat.ColumnDescription) -> arr.Strings:
        s = f'P({output.colname().string()}=v|Weights,x) = exp(Weights[v] . x) / K'
        return arr.strings_singleton(s)

    def prediction_component_strings(self, output: dat.ColumnDescription) -> arr.Strings:
        result = arr.strings_empty()
        for vn in output.valnames().range():
            result.add(f'p_{vn.string()}')
        return result

    def __init__(self, vns: dat.Valnames, value_to_term_num_to_weight: arr.Fmat, ):
        value_to_term_num_to_weight.assert_ok()
        self.m_valnames = vns
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

    def valnames(self) -> dat.Valnames:
        return self.m_valnames


def gen_weights_multinomial_create(vns: dat.Valnames, weights_as_fmat: arr.Fmat) -> GenWeightsMultinomial:
    return GenWeightsMultinomial(vns, weights_as_fmat)


def gen_weights_multinomial_all_zero(vns: dat.Valnames, n_terms: int) -> GenWeightsMultinomial:
    return gen_weights_multinomial_create(vns, arr.fmat_of_zeroes(vns.len(), n_terms))


def unit_test():
    s = """
    a,b,c,y
    0,0,0,50\n
    10,1,0,51\n
    0,1,0,50\n
    0,0,0,50"""
    ds, ok = dat.datset_from_multiline_string(s)
    assert ok
    output = ds.subset('y')
    inputs = ds.without_column(output.named_column(0))
    td = noo.transformer_description_from_datsets(inputs, output)
    nfs = td.input_transformers().named_float_records_from_datset(inputs)
    assert nfs.num_covariates() == inputs.num_cols()
    fgc = floater_class_glm(1)
    tfs = noo.transformers_from_datset(inputs)
    inputs_as_named_float_records = tfs.named_float_records_from_datset(inputs)
    fg = fgc.train(inputs_as_named_float_records.float_records(), output.column(0))
    gw = fg.gen_weights()
    assert isinstance(gw, GenWeightsLinear)
    p = polynomial_from_polynomial_structure(fg.polynomial_structure(), gw.floats())
    p2 = p.account_for_transformer_intervals(td.input_intervals())
    p2.assert_ok()
    true_weights = arr.floats_varargs(50.0, 0.1, 0.0)
    assert true_weights.distance_to(p2.coefficients_slow()) < 0.5


class PiClassGlm(pic.PiClass):
    def __init__(self, name: str):
        self.m_name = name
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_name, str)

    def choose(self, train: dat.LearnData, test: dat.LearnData) -> pic.TestResults:
        trs = pic.test_results_empty()
        max_degree = 2
        for degree in range(max_degree + 1):
            trs.add_experiment(model_class_glm(degree), train, test)
        return trs


def piclass_glm() -> PiClassGlm:
    return PiClassGlm('glm')
