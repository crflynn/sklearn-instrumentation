import logging

from sklearn.base import BaseEstimator
from sklearn.datasets import load_iris
from sklearn.dummy import DummyClassifier

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.testing import SklearnInstrumentionAsserter


def test_instrumentation(classification_model, simple_decorator, full):
    instrumentor = SklearnInstrumentor(instrument=simple_decorator)
    instrumentor.instrument_instance(classification_model)
    asserter = SklearnInstrumentionAsserter(instrumentor=instrumentor)
    asserter.assert_instrumented_estimator(classification_model)
    instrumentor.uninstrument_instance(classification_model, full=full)
    asserter.assert_uninstrumented_estimator(classification_model, full=full)


def test_package_instrumentation(simple_decorator, full):
    instrumentor = SklearnInstrumentor(instrument=simple_decorator)
    instrumentor.instrument_packages(["sklearn"])
    asserter = SklearnInstrumentionAsserter(instrumentor=instrumentor)
    asserter.assert_instrumented_packages(["sklearn"])
    instrumentor.uninstrument_packages(["sklearn"], full=full)
    asserter.assert_uninstrumented_packages(["sklearn"], full=full)


def test_class_instrumentation(classification_model, simple_decorator, full):
    instrumentor = SklearnInstrumentor(instrument=simple_decorator)
    instrumentor.instrument_instance_classes(classification_model)
    asserter = SklearnInstrumentionAsserter(instrumentor=instrumentor)
    classes = instrumentor._get_instance_classes(classification_model)
    asserter.assert_instrumented_classes(estimators=classes)
    instrumentor.uninstrument_instance_classes(classification_model, full=full)
    asserter.assert_uninstrumented_classes(classes, full=full)


def test_no_duplicate_instrumentation(caplog, simple_decorator):
    caplog.set_level(level=logging.INFO)

    class DoubleAttributeEstimator(BaseEstimator):
        def __init__(self, estimator):
            self.first_estimator = estimator
            self.second_estimator = estimator

        def fit(self, X, y):
            self.first_estimator.fit(X, y)
            return self

    dummy = DummyClassifier()
    meta_estimator = DoubleAttributeEstimator(estimator=dummy)

    X, y = load_iris(return_X_y=True)

    instrumentor = SklearnInstrumentor(instrument=simple_decorator)
    instrumentor.instrument_instance(meta_estimator)

    meta_estimator.fit(X, y)
    # assert once for meta_estimator + once for dummy
    # even though dummy is referenced in two attributes of meta_estimator
    assert caplog.text.count("hello world") == 2
