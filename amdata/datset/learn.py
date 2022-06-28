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


class FloaterClass(ABC):
    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def train(self, inputs: noo.FloatRecords, output: dat.Column) -> Floater:
        pass

    @abstractmethod
    def name_as_string(self) -> str:
        pass


class ModelFloater(Model):
    def assert_ok(self):
        assert isinstance(self.m_transformer_description, noo.Transformers)
        self.m_transformer_description.assert_ok()
        assert isinstance(self.m_floater, Floater)
        self.m_floater.assert_ok()

    def pretty_string(self) -> str:
        return self.floater().pretty_string(self.transformer_description())

    def predict_from_record(self, rec: dat.Record) -> dis.Distribution:
        fr = self.input_transformers().float_record_from_record(rec)
        return self.floater().predict_from_float_record(fr)

    def prediction_component_strings(self) -> arr.Strings:
        return self.floater().prediction_component_strings(self.output_description())

    def __init__(self, td: noo.TransformerDescription, fl: Floater):
        self.m_transformer_description = td
        self.m_floater = fl
        self.assert_ok()

    def floater(self) -> Floater:
        return self.m_floater

    def input_transformers(self) -> noo.Transformers:
        return self.transformer_description().input_transformers()

    def output_description(self) -> dat.ColumnDescription:
        return self.transformer_description().output_description()

    def transformer_description(self) -> noo.TransformerDescription:
        return self.m_transformer_description


class Floater(ABC):
    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def pretty_string(self, td: noo.TransformerDescription):
        pass

    @abstractmethod
    def predict_from_float_record(self, fr: noo.FloatRecord) -> dis.Distribution:
        pass

    @abstractmethod
    def prediction_component_strings(self, output: dat.ColumnDescription) -> arr.Strings:
        pass


def model_from_floater(td: noo.TransformerDescription, fl: Floater) -> ModelFloater:
    return ModelFloater(td, fl)


class ModelClassFloatsOnly(ModelClass):
    def assert_ok(self):
        assert isinstance(self.m_floater_class, FloaterClass)
        self.m_floater_class.assert_ok()
        assert isinstance(self.m_transformers, noo.Transformers)
        self.m_transformers.assert_ok()

    def train_from_named_column(self, inputs: dat.Datset, output: dat.NamedColumn) -> Model:
        td = noo.transformer_description_from_datset(inputs, output)
        nfs = td.input_transformers().named_float_records_from_datset(inputs)
        fc = self.floater_class()
        fl = fc.train(nfs.float_records(), output.column())
        return model_from_floater(td, fl)

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
