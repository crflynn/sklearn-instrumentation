import logging

import pandas as pd

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.logging import ColumnNameLogger
from sklearn_instrumentation.instruments.logging import GetSizeOfLogger
from sklearn_instrumentation.instruments.logging import ShapeLogger
from sklearn_instrumentation.instruments.logging import TimeElapsedLogger
from sklearn_instrumentation.testing import SklearnInstrumentionAsserter


def test_time_elapsed_logger(classification_model, iris, caplog):
    caplog.set_level(logging.INFO)
    instrumentor = SklearnInstrumentor(instrument=TimeElapsedLogger())
    classification_model.fit(iris.X_train, iris.y_train)
    assert "elapsed time" not in caplog.text

    instrumentor.instrument_instance(classification_model)
    classification_model.predict(iris.X_test)
    expected_message_parts = [
        "Pipeline.predict starting",
        "FeatureUnion.transform starting",
        "StandardScaler.transform starting",
        "StandardScaler.transform elapsed time",
        "PCA.transform (_BasePCA.transform) starting",
        "PCA.transform (_BasePCA.transform) elapsed time",
        "FeatureUnion.transform elapsed time",
        "TransformerWithEnum.transform starting",
        "TransformerWithEnum.transform elapsed time",
        "RandomForestClassifier.predict (ForestClassifier.predict) starting",
        "RandomForestClassifier.predict_proba (ForestClassifier.predict_proba) starting",
        "RandomForestClassifier.predict_proba (ForestClassifier.predict_proba) elapsed time",
        "RandomForestClassifier.predict (ForestClassifier.predict) elapsed time",
        "Pipeline.predict elapsed time",
    ]
    assert len(caplog.records) > 0
    for expected, record in zip(expected_message_parts, caplog.records):
        assert expected in record.getMessage()

    caplog.clear()
    instrumentor.uninstrument_instance(classification_model)
    classification_model.predict(iris.X_test)
    assert len(caplog.records) == 0


def test_getsizeof_logger(classification_model, iris, caplog):
    caplog.set_level(logging.INFO)
    instrumentor = SklearnInstrumentor(instrument=GetSizeOfLogger())
    classification_model.fit(iris.X_train, iris.y_train)
    assert "nbytes" not in caplog.text

    instrumentor.instrument_instance(classification_model)
    classification_model.predict(iris.X_test)
    expected_message_parts = [
        "Pipeline.predict input X nbytes",
        "FeatureUnion.transform input X nbytes",
        "StandardScaler.transform input X nbytes",
        "StandardScaler.transform output X nbytes",
        "PCA.transform (_BasePCA.transform) input X nbytes",
        "PCA.transform (_BasePCA.transform) output X nbytes",
        "FeatureUnion.transform output X nbytes",
        "TransformerWithEnum.transform input X nbytes",
        "TransformerWithEnum.transform output X nbytes",
        "RandomForestClassifier.predict (ForestClassifier.predict) input X nbytes",
        "RandomForestClassifier.predict_proba (ForestClassifier.predict_proba) input X nbytes",
        "RandomForestClassifier.predict_proba (ForestClassifier.predict_proba) output X nbytes",
        "RandomForestClassifier.predict (ForestClassifier.predict) output X nbytes",
        "Pipeline.predict output X nbytes",
    ]
    assert len(caplog.records) > 0
    for expected, record in zip(expected_message_parts, caplog.records):
        assert expected in record.getMessage()

    caplog.clear()
    instrumentor.uninstrument_instance(classification_model)
    classification_model.predict(iris.X_test)
    assert len(caplog.records) == 0


def test_shape_logger(classification_model, iris, caplog):
    caplog.set_level(logging.INFO)
    instrumentor = SklearnInstrumentor(instrument=ShapeLogger())
    classification_model.fit(iris.X_train, iris.y_train)
    assert "shape" not in caplog.text

    instrumentor.instrument_instance(classification_model)
    classification_model.predict(iris.X_test)
    expected_message_parts = [
        "Pipeline.predict input X shape",
        "FeatureUnion.transform input X shape",
        "StandardScaler.transform input X shape",
        "StandardScaler.transform output X shape",
        "PCA.transform (_BasePCA.transform) input X shape",
        "PCA.transform (_BasePCA.transform) output X shape",
        "FeatureUnion.transform output X shape",
        "TransformerWithEnum.transform input X shape",
        "TransformerWithEnum.transform output X shape",
        "RandomForestClassifier.predict (ForestClassifier.predict) input X shape",
        "RandomForestClassifier.predict_proba (ForestClassifier.predict_proba) input X shape",
        "RandomForestClassifier.predict_proba (ForestClassifier.predict_proba) output X shape",
        "RandomForestClassifier.predict (ForestClassifier.predict) output X shape",
        "Pipeline.predict output X shape",
    ]
    assert len(caplog.records) > 0
    for expected, record in zip(expected_message_parts, caplog.records):
        assert expected in record.getMessage()

    caplog.clear()
    instrumentor.uninstrument_instance(classification_model)
    classification_model.predict(iris.X_test)
    assert len(caplog.records) == 0


def test_column_name_logger(classification_model, iris, caplog):
    caplog.set_level(logging.INFO)
    instrumentor = SklearnInstrumentor(instrument=ColumnNameLogger())
    classification_model.fit(iris.X_train, iris.y_train)
    assert "columns" not in caplog.text

    instrumentor.instrument_instance(classification_model)
    classification_model.predict(pd.DataFrame(iris.X_test))
    expected_message_parts = [
        "Pipeline.predict input columns",
        "FeatureUnion.transform input columns",
        "StandardScaler.transform input columns",
        "PCA.transform (_BasePCA.transform) input columns",
    ]
    assert len(caplog.records) > 0
    for expected, record in zip(expected_message_parts, caplog.records):
        assert expected in record.getMessage()

    caplog.clear()
    instrumentor.uninstrument_instance(classification_model)
    classification_model.predict(pd.DataFrame(iris.X_test))
    assert len(caplog.records) == 0

    asserter = SklearnInstrumentionAsserter(instrumentor=instrumentor)
    asserter.assert_uninstrumented_estimator(classification_model)
