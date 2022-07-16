from __future__ import annotations

from abc import ABC, abstractmethod

import datset.amarrays as arr
import datset.distribution as dis
import datset.dset as dat
import datset.numset as noo


class ModelClass(ABC):
    """
    A ModelClass is a definition of the model, including hyperparameters. You need a model class in
    order to do machine learning. At the moment the available model classes are only polynomial GLMs (General
    Linear Models) which do a linear regression, logistic regression or multinomial regression depending on
    whether the output is float, bool or categorical respectively.

    To create a GLM Model Class call

    import datset.learn as lea
    mc = lea.model_class_glm(2)

    The 2 is a hyperparameter: polynomial degree. Use 1 for linear, 2 for quadratic etc.

    Things you can do with a model class:
    train() : create a model from an input and output training set
    explain() : send to stdout a short string explaining the model class
    """

    @abstractmethod
    def assert_ok(self):
        pass

    @abstractmethod
    def train_from_learn_data(self, ld: dat.LearnData) -> Model:
        pass

    @abstractmethod
    def name_as_string(self) -> str:
        pass

    def train(self, inputs: dat.Datset, output: dat.Datset) -> Model:
        """
        Finds a good fitting model from the model class which hopefully explains the data accurately
        :param inputs: A Datset with zero or more named columns
        :param output: A Datset with exactly one named column
        :return: a Model, which can be used for predictions
        """
        return self.train_from_learn_data(dat.learn_data_from_datsets(inputs, output))

    def explain(self):
        """
        Send a short self-description to stdout
        :return: Nothing
        """
        print(self.pretty_string())

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    @abstractmethod
    def pretty_strings(self) -> arr.Strings:
        pass


class Model(ABC):
    """
    A mapping from a Record (one row of a Datset) to an output distribution, specifying the
    distribution of the output value, conditioned on the observed value in the input record.

    A model is obtained from a model class, and an input and output Datset

    To create a model do

    import datset.learn as lea
    mod = mc.train(inputs,output)

    Useful methods on models:
    mod.explain() self-description to stdout
    mod.predict_from_record() returns the predicted probability distribution
    mod.batch_predict() takes a whole set of records (in the form of a datset) and outputs a new datset
                        in which the i'th output record explains the probability distribution of the
                        prediction for the i'th input record
    mod.loglike(input,output) computes the log of the probability (or pdf) of the output value.
    """

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
        fsa = arr.floats_array_empty()
        for r in inputs.range_records():
            di = self.predict_from_record(r)
            assert di.distribution_description().equals(self.distribution_description())

            fsa.add(di.as_floats())
        fm = arr.fmat_from_rows(cns.len(), fsa)
        return dat.datset_from_fmat(cns, fm)

    def prediction_colnames(self) -> dat.Colnames:
        return dat.colnames_from_strings(self.prediction_component_strings())

    @abstractmethod
    def prediction_component_strings(self) -> arr.Strings:
        pass

    def loglike_batch(self, test_in: dat.Datset, test_out: dat.Datset) -> float:
        out_col = test_out.column(0)
        result = 0.0
        for input_record, output_atom in zip(test_in.range_records(), out_col.range()):
            delta = self.loglike(input_record, output_atom)
            assert isinstance(delta, float)
            result += delta

        return result

    def loglike_from_learn_data(self, test: dat.LearnData) -> float:
        out_col = test.output_column()
        result = 0.0
        for input_record, output_atom in zip(test.inputs().range_records(), out_col.range()):
            delta = self.loglike(input_record, output_atom)
            assert isinstance(delta, float)
            result += delta

        return result

    def loglike(self, input_record: dat.Record, output_atom: dat.Atom) -> float:
        dst = self.predict_from_record(input_record)
        result = dst.loglike(output_atom)
        assert isinstance(result, float)
        return result

    @abstractmethod
    def loosely_equals(self, other: Model) -> bool:
        pass

    @abstractmethod
    def distribution_description(self) -> dis.DistributionDescription:
        pass


def q_from_y(y_k: dat.Atom) -> float:
    assert isinstance(y_k, dat.AtomBool)
    if y_k.bool():
        return -1.0
    else:
        return 1.0


penalty_parameter: float = 0.002


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

    @abstractmethod
    def pretty_strings(self) -> arr.Strings:
        pass


class ModelFloater(Model):
    def distribution_description(self) -> dis.DistributionDescription:
        return self.floater().distribution_description()

    def loosely_equals(self, other: Model) -> bool:
        if not isinstance(other, ModelFloater):
            return False

        assert isinstance(other, ModelFloater)
        if not self.m_floater.loosely_equals(other.m_floater):
            return False

        if not self.m_transformer_description.loosely_equals(other.m_transformer_description):
            return False

        return True

    def assert_ok(self):
        assert isinstance(self.m_transformer_description, noo.TransformerDescription)
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

    @abstractmethod
    def loosely_equals(self, other: Floater) -> bool:
        pass

    @abstractmethod
    def distribution_description(self):
        pass


def model_from_floater(td: noo.TransformerDescription, fl: Floater) -> ModelFloater:
    return ModelFloater(td, fl)


class ModelClassFloater(ModelClass):
    def pretty_strings(self) -> arr.Strings:
        result = arr.strings_empty()
        result.add('This is a floater model which means it begins by transforming all columns to floats')
        result.add('then using hot coding of categoricals and linear reduction of floats to unit interval')
        result.add('and then the underlying algorithm is:')
        result.append(self.floater_class().pretty_strings())
        return result

    def assert_ok(self):
        assert isinstance(self.m_floater_class, FloaterClass)
        self.m_floater_class.assert_ok()

    def train_from_learn_data(self, ld: dat.LearnData) -> Model:
        td = noo.transformer_description_from_learn_data(ld)
        nfs = td.input_transformers().named_float_records_from_datset(ld.inputs())
        fc = self.floater_class()
        fl = fc.train(nfs.float_records(), ld.output().column())
        return model_from_floater(td, fl)

    def name_as_string(self) -> str:
        return self.floater_class().name_as_string()

    def __init__(self, fc: FloaterClass):
        self.m_floater_class = fc
        self.assert_ok()

    def floater_class(self) -> FloaterClass:
        return self.m_floater_class


def model_class_from_floater_class(fc: FloaterClass) -> ModelClassFloater:
    return ModelClassFloater(fc)
