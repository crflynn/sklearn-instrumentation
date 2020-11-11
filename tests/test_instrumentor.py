from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.testing import SklearnInstrumentionAsserter


def test_instrumentation(classification_model, simple_decorator, full):
    instrumentor = SklearnInstrumentor(decorator=simple_decorator)
    instrumentor.instrument_estimator(classification_model)
    asserter = SklearnInstrumentionAsserter(instrumentor=instrumentor)
    asserter.assert_instrumented_estimator(classification_model)
    instrumentor.uninstrument_estimator(classification_model, full=full)
    asserter.assert_uninstrumented_estimator(classification_model, full=full)


def test_package_instrumentation(simple_decorator, full):
    instrumentor = SklearnInstrumentor(decorator=simple_decorator)
    instrumentor.instrument_packages(["sklearn"])
    asserter = SklearnInstrumentionAsserter(instrumentor=instrumentor)
    asserter.assert_instrumented_packages(["sklearn"])
    instrumentor.uninstrument_packages(["sklearn"], full=full)
    asserter.assert_uninstrumented_packages(["sklearn"], full=full)
