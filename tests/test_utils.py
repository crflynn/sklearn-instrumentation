import logging

import pytest
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.utils import compose_decorators
from sklearn_instrumentation.utils import get_arg_by_key
from sklearn_instrumentation.utils import get_descriptor
from sklearn_instrumentation.utils import get_estimators_in_packages
from sklearn_instrumentation.utils import get_sklearn_estimator_from_method
from sklearn_instrumentation.utils import has_instrumentation
from sklearn_instrumentation.utils import is_class_method
from sklearn_instrumentation.utils import is_delegator
from sklearn_instrumentation.utils import is_instance_method
from sklearn_instrumentation.utils import method_is_inherited
from sklearn_instrumentation.utils import non_self_arg


def test_compose_decorators(simple_decorator, caplog):
    caplog.set_level(level=logging.INFO)
    double_simple = compose_decorators([simple_decorator, simple_decorator])

    def noop(*args, **kwargs):
        pass

    est = BaseEstimator()
    noop = double_simple(est, noop)
    noop()

    assert len(caplog.records) == 2
    for record in caplog.records:
        assert "hello world" in record.getMessage()


def test_get_sklearn_estimator_from_method():
    rf = RandomForestClassifier()
    est = get_sklearn_estimator_from_method(rf.fit)
    assert rf == est

    pipeline = Pipeline(steps=[("rf", rf)])
    est = get_sklearn_estimator_from_method(pipeline.predict)
    assert pipeline == est

    class NotBaseEstimator:
        def fit(self, X, y):
            return self

    nba = NotBaseEstimator()
    with pytest.raises(TypeError):
        _ = get_sklearn_estimator_from_method(nba.fit)


def test_is_class_method():
    assert is_class_method(RandomForestClassifier.fit)
    assert not is_class_method(RandomForestClassifier().fit)


def test_is_instance_method():
    assert not is_instance_method(RandomForestClassifier.fit)
    assert is_instance_method(RandomForestClassifier().fit)


def test_get_descriptor():
    assert get_descriptor(Pipeline.predict) is not None
    with pytest.raises(TypeError):
        get_descriptor(Pipeline.fit)


def test_is_delegator():
    assert is_delegator(Pipeline.predict)
    assert not is_delegator(Pipeline.fit)


def test_method_is_inherited():
    assert method_is_inherited(RandomForestClassifier, RandomForestClassifier.fit)
    rfc = RandomForestClassifier()
    assert method_is_inherited(rfc, rfc.fit)
    assert not method_is_inherited(StandardScaler, StandardScaler.transform)
    ss = StandardScaler()
    assert not method_is_inherited(ss, ss.transform)


def test_has_instrumentation(simple_decorator, iris):
    lr = LinearRegression()
    pipeline = Pipeline(steps=[("lr", lr)])
    pipeline.fit(iris.X_train, iris.y_train)
    assert not has_instrumentation(LinearRegression, "fit")
    assert not has_instrumentation(lr, "fit")
    assert not has_instrumentation(pipeline, "predict")
    instrumentor = SklearnInstrumentor(instrument=simple_decorator)
    instrumentor.instrument_packages(["sklearn"])
    assert has_instrumentation(LinearRegression, "fit")
    assert has_instrumentation(lr, "fit")
    assert has_instrumentation(pipeline, "predict")
    instrumentor.uninstrument_packages(["sklearn"])
    assert not has_instrumentation(LinearRegression, "fit")
    assert not has_instrumentation(lr, "fit")
    assert not has_instrumentation(pipeline, "predict")


def test_non_self_arg():
    rf = RandomForestClassifier()
    args = (0, 1, 2)
    assert non_self_arg(RandomForestClassifier.fit, args, 0) == 1
    assert non_self_arg(rf.fit, args, 0) == 0


def test_get_arg_by_key():
    rf = RandomForestClassifier()
    args = (0, 1, 2)
    assert get_arg_by_key(RandomForestClassifier.fit, args, "X") == 1
    assert get_arg_by_key(rf.fit, args, "X") == 0


def test_get_estimators_in_packages():
    assert len(get_estimators_in_packages(["pandas"])) == 0
    assert len(get_estimators_in_packages(["xgboost"])) > 0
    assert len(get_estimators_in_packages(["lightgbm"])) > 0
    sklearn_estimators = get_estimators_in_packages(["sklearn"])
    assert BaseEstimator in sklearn_estimators
    assert RandomForestClassifier in sklearn_estimators
    assert Pipeline in sklearn_estimators
