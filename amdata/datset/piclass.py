from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, List, Iterator

import datset.dset as dat
import datset.amarrays as arr
import datset.learn as lea


class PiClass(ABC):
    """
    A PiClass is a Partially-Instantiated Model Class. This is a representation of a set of ModelClass.
    The example that currently exists is

    import datset.piclass as pic
    pc = pic.piclass_glm()

    which represents the set of ModelClass that are polynomials.

    From a PiClass you can give an input Datset and output Datset and learn the best ModelClass in the set
    of models specified by the PiClass and also the actual Model obtained from that ModelClass on this
    data.

    mc, mod = pc.train(inputs,output)

    """
    @abstractmethod
    def assert_ok(self):
        pass

    def train(self, inputs: dat.Datset, output: dat.Datset) -> Tuple[lea.ModelClass, lea.Model]:
        return self.train_from_learn_data(dat.learn_data_from_datsets(inputs, output))

    def train_from_learn_data(self, ld: dat.LearnData) -> Tuple[lea.ModelClass, lea.Model]:
        best_mc = self.choose_model_class(ld)
        return best_mc, best_mc.train_from_learn_data(ld)

    def choose_model_class(self, sld: dat.LearnData) -> lea.ModelClass:
        train, test = sld.train_test_split(0.67)
        trs = self.choose(train, test)
        trs.explain()
        return trs.best_model_class()

    @abstractmethod
    def choose(self, train: dat.LearnData, test: dat.LearnData) -> TestResults:
        pass


class Evaluation:
    def __init__(self, train_loglike: float, test_loglike: float):
        self.m_train_loglike = train_loglike
        self.m_test_loglike = test_loglike
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_train_loglike, float)
        assert isinstance(self.m_test_loglike, float)

    def test_loglike(self) -> float:
        return self.m_test_loglike

    def train_loglike(self) -> float:
        return self.m_train_loglike


class TestResult:
    def __init__(self, mc: lea.ModelClass, ev: Evaluation):
        self.m_model_class = mc
        self.m_evaluation = ev
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_model_class, lea.ModelClass)
        self.m_model_class.assert_ok()
        assert isinstance(self.m_evaluation, Evaluation)
        self.m_evaluation.assert_ok()

    def test_eval(self) -> float:
        return self.evaluation().test_loglike()

    def train_eval(self) -> float:
        return self.evaluation().train_loglike()

    def evaluation(self) -> Evaluation:
        return self.m_evaluation

    def model_class(self) -> lea.ModelClass:
        return self.m_model_class

    def explain_strings(self) -> arr.Strings:
        return arr.strings_varargs(self.model_name(), f'{self.train_eval()}', f'{self.test_eval()}')

    def model_name(self) -> str:
        return self.model_class().name_as_string()


def evaluation_create(train_loglike: float, test_loglike: float) -> Evaluation:
    return Evaluation(train_loglike, test_loglike)


def test_result_create(mc: lea.ModelClass, ev: Evaluation) -> TestResult:
    return TestResult(mc, ev)


def evaluation_from_learn_data(mc: lea.ModelClass, train: dat.LearnData, test: dat.LearnData) -> Evaluation:
    mod = mc.train_from_learn_data(train)
    return evaluation_create(mod.loglike_from_learn_data(train), mod.loglike_from_learn_data(test))


def test_result_from_learn_data(mc: lea.ModelClass, train: dat.LearnData, test: dat.LearnData) -> TestResult:
    return test_result_create(mc, evaluation_from_learn_data(mc, train, test))


class TestResults:
    def __init__(self, li: List[TestResult]):
        self.m_list = li
        self.assert_ok()

    def assert_ok(self):
        assert isinstance(self.m_list, list)
        for tr in self.m_list:
            assert isinstance(tr, TestResult)
            tr.assert_ok()

    def best_model_class_index(self) -> int:
        assert self.len() > 0
        best = 0
        for i in range(1, self.len()):
            if self.test_eval(i) > self.test_eval(best):
                best = i
        return best

    def best_model_class(self) -> lea.ModelClass:
        return self.model_class(self.best_model_class_index())

    def len(self) -> int:
        return len(self.m_list)

    def test_result(self, index: int) -> TestResult:
        assert 0 <= index < self.len()
        return self.m_list[index]

    def explain(self):
        print(self.pretty_string())

    def pretty_string(self) -> str:
        return self.pretty_strings().concatenate_fancy('', '\n', '')

    def pretty_strings(self) -> arr.Strings:
        first_line = arr.strings_varargs("", "Model", "TrainLoglike", "TestLoglike")
        result = arr.strings_array_empty()
        result.add(first_line)

        best_index = self.best_model_class_index()
        for index, tr in enumerate(self.range()):
            s = "*" if index == best_index else ""
            result.add(arr.strings_singleton(s).with_many(tr.explain_strings()))

        return result.pretty_strings()

    def test_eval(self, index: int) -> float:
        return self.test_result(index).test_eval()

    def train_eval(self, index: int) -> float:
        return self.test_result(index).train_eval()

    def range(self) -> Iterator[TestResult]:
        for tr in self.m_list:
            yield tr

    def model_class(self, index: int) -> lea.ModelClass:
        return self.test_result(index).model_class()

    def add_experiment(self, mc: lea.ModelClass, train: dat.LearnData, test: dat.LearnData):
        self.add(test_result_from_learn_data(mc, train, test))

    def add(self, tr: TestResult):
        self.m_list.append(tr)


def test_results_empty() -> TestResults:
    return TestResults([])
