from __future__ import annotations

from abc import ABC, abstractmethod

import datset.amarrays as arr
import datset.distribution as dis
import datset.dset as dat
import datset.numset as noo


class ModelClass(ABC):
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
        assert output.num_records() == inputs.num_records()
        return self.train_from_named_column(inputs, output.named_column(0))


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
    def predict_from_record(self, rec: dat.Record) -> dis.Distribution:
        pass

    def batch_predict(self, inputs: dat.Datset) -> dat.Datset:
        cns = self.prediction_colnames()
        rfm = arr.row_indexed_fmat_with_no_rows(cns.len())
        for r in inputs.range_records():
            di = self.predict_from_record(r)
            rfm.add_row(di.as_floats())
        fm = arr.fmat_create(rfm)
        return dat.datset_from_fmat(cns, fm)

    def prediction_colnames(self) -> dat.Colnames:
        return dat.colnames_from_strings(self.prediction_component_strings())

    @abstractmethod
    def prediction_component_strings(self) -> arr.Strings:
        pass


def q_from_y(y_k: dat.Atom) -> float:
    assert isinstance(y_k, dat.AtomBool)
    if y_k.bool():
        return -1.0
    else:
        return 1.0


penalty_parameter: float = 0.0001


# def ws_penalty_second_derivative() -> float:
#     return penalty_parameter * 2
#
#
# class Weights:
#     def __init__(self, fs: arr.Floats):
#         self.m_weights = fs
#         self.assert_ok()
#
#     def assert_ok(self):
#         assert isinstance(self.m_weights, arr.Floats)
#         self.m_weights.assert_ok()
#
#     def loglike(self, le: ModelClass, inputs: noo.Termvecs, output: dat.Column) -> float:
#         beta_xs = self.premultiplied_by(inputs)
#         return le.loglike(beta_xs, output) - self.penalty()
#
#     def loglike_derivative(self, le: ModelClass, inputs: noo.Termvecs, output: dat.Column, col: int) -> float:
#         result = 0.0
#         for x_k, y_k in zip(inputs.range(), output.range()):
#             result += self.loglike_derivative_from_row(le, x_k, y_k, col)
#         return result - self.penalty_derivative(col)
#
#     def loglike_second_derivative(self, le: ModelClass, inputs: noo.Termvecs, output: dat.Column, col: int) -> float:
#         result = 0.0
#         for x_k, y_k in zip(inputs.range(), output.range()):
#             result += self.loglike_second_derivative_from_row(le, x_k, y_k, col)
#         return result - ws_penalty_second_derivative()
#
#     def increment(self, col: int, delta: float):
#         self.m_weights.increment(col, delta)
#
#     def floats(self) -> arr.Floats:
#         return self.m_weights
#
#     def loglike_derivative_from_row(self, le: ModelClass, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
#         return le.loglike_derivative_from_row(self, x_k, y_k, j)
#
#     def loglike_second_derivative_from_row(self, le: ModelClass, x_k: noo.Termvec, y_k: dat.Atom, j: int) -> float:
#         return le.loglike_second_derivative_from_row(self, x_k, y_k, j)
#
#     def times(self, x_k: noo.Termvec) -> float:
#         return self.floats().dot_product(x_k.floats())
#
#     def pretty_strings_with_introduction(self, tfs: noo.Transformers, intro: str) -> arr.Strings:
#         result = arr.strings_singleton(intro)
#         result.add('')
#         result.add('where...')
#         result.add('')
#         result.append(self.pretty_strings(tfs))
#         return result
#
#     def pretty_string_with_introduction(self, tfs: noo.Transformers, intro: str) -> str:
#         return self.pretty_strings_with_introduction(tfs, intro).concatenate_fancy('', '\n', '')
#
#     def num_cols_in_z_matrix(self) -> int:
#         return self.floats().len()
#
#     def pretty_strings(self, tfs: noo.Transformers) -> arr.Strings:
#         return self.strings_array(tfs).pretty_strings()
#
#     def strings_array(self, tfs: noo.Transformers) -> arr.StringsArray:
#         nns = tfs.noomnames()
#         assert self.num_cols_in_z_matrix() == nns.len() + 1
#         result = arr.strings_array_empty()
#         cws = self.correct_accounting_for_transformers(tfs)
#         nns_as_strings = arr.strings_singleton('constant').with_many(nns.strings())
#         for nn_as_string, w in zip(nns_as_strings.range(), cws.range()):
#             ss = arr.strings_empty()
#             ss.add(f'w[{nn_as_string}]')
#             ss.add('=')
#             ss.add(f'{w}')
#             result.add(ss)
#         return result
#
#     def weight(self, i: int) -> float:
#         return self.floats().float(i)
#
#     def premultiplied_by(self, tvs: noo.Termvecs) -> arr.Floats:
#         return tvs.times(self.floats())
#
#     def deep_copy(self):  # returns WeightsArray
#         fs = self.floats().deep_copy()
#         assert isinstance(fs, arr.Floats)
#         return weights_create(fs)
#
#     def penalty(self) -> float:
#         return penalty_parameter * self.sum_squares()
#
#     def penalty_derivative(self, col: int) -> float:
#         return penalty_parameter * 2 * self.weight(col)
#
#     def sum_squares(self) -> float:
#         return self.floats().sum_squares()
#
#     def range(self) -> Iterator[float]:
#         return self.m_weights.range()
#
#     def correct_accounting_for_transformers(self, tfs: noo.Transformers) -> Weights:
#         # predict = c_old + sum_j wold_j * (x_j - lo_j)/width_j
#         #
#         # predict = c_new + sum_j w_new_j x_j
#         #
#         # c_new = c_old + sum_j wold_j (-lo_j) / width_j
#         # w_new_j = wold_j/width_j
#         c_old = self.weight(0)
#         c_new = c_old
#         j_to_lo = arr.floats_empty()
#         j_to_width = arr.floats_empty()
#
#         for tf in tfs.range():
#             for iv in tf.scaling_intervals().range():
#                 j_to_lo.add(iv.lo())
#                 j_to_width.add(iv.width())
#
#         j_to_wold = self.floats().without_leftmost_element()
#         j_to_w_new = arr.floats_empty()
#
#         for wold, lo, width in zip(j_to_wold.range(), j_to_lo.range(), j_to_width.range()):
#             j_to_w_new.add(wold / width)
#             c_new -= wold * lo / width
#
#         result = arr.floats_singleton(c_new)
#         result.append(j_to_w_new)
#
#         assert result.len() == self.num_cols_in_z_matrix()
#
#         return weights_create(result)
#

class FloaterClass(ABC):
    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def train_from_named_column(self, inputs: noo.NamedFloatRecords, output: dat.NamedColumn) -> Floater:
        pass

    @abstractmethod
    def name_as_string(self) -> str:
        pass


class ModelFloater(Model):
    def assert_ok(self):
        assert isinstance(self.m_transformers, noo.Transformers)
        assert isinstance(self.m_output_description, dat.ColumnDescription)
        self.m_output_description.assert_ok()
        assert isinstance(self.m_floater, Floater)
        self.m_floater.assert_ok()

    def pretty_string(self) -> str:
        return self.floater().pretty_string(self.transformers().names(), self.output_description())

    def predict_from_record(self, rec: dat.Record) -> dis.Distribution:
        fr = self.transformers().float_record_from_record(rec)
        return self.floater().predict(fr)

    def prediction_component_strings(self) -> arr.Strings:
        return self.floater().prediction_component_strings()

    def __init__(self, tfs: noo.Transformers, output: dat.ColumnDescription, fl: Floater):
        self.m_transformers = tfs
        self.m_output_description = output
        self.m_floater = fl
        self.assert_ok()

    def floater(self) -> Floater:
        return self.m_floater

    def transformers(self) -> noo.Transformers:
        return self.m_transformers

    def output_description(self) -> dat.ColumnDescription:
        return self.m_output_description


class Floater(ABC):
    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def pretty_string(self, input_names: arr.Strings, output: dat.ColumnDescription):
        pass

    @abstractmethod
    def predict(self, fr: noo.FloatRecord) -> dis.Distribution:
        pass

    @abstractmethod
    def prediction_component_strings(self) -> arr.Strings:
        pass


def model_from_floater(tfs: noo.Transformers, output: dat.ColumnDescription, fl: Floater) -> ModelFloater:
    return ModelFloater(tfs, output, fl)


class ModelClassFloatsOnly(ModelClass):
    def assert_ok(self):
        assert isinstance(self.m_floater_class, FloaterClass)
        self.m_floater_class.assert_ok()
        assert isinstance(self.m_transformers, noo.Transformers)
        self.m_transformers.assert_ok()

    def train_from_named_column(self, inputs: dat.Datset, output: dat.NamedColumn) -> Model:
        tfs = self.transformers()
        nfs = tfs.named_float_records_from_datset(inputs)
        return model_from_floater(tfs, output.column_description(),
                                  self.floater_class().train_from_named_column(nfs, output))

    def name_as_string(self) -> str:
        return self.floater_class().name_as_string()

    def __init__(self, fc: FloaterClass, tfs: noo.Transformers):
        self.m_floater_class = fc
        self.m_transformers = tfs
        self.assert_ok()

    def transformers(self) -> noo.Transformers:
        return self.m_transformers

    def floater_class(self) -> FloaterClass:
        return self.m_floater_class
