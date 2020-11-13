import threading
import time

import requests
from prometheus_client import Histogram
from prometheus_client import Summary
from prometheus_client import start_http_server

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.prometheus import PrometheusHistogram
from sklearn_instrumentation.instruments.prometheus import PrometheusSummary


def test_prometheus_histogram(classification_model, iris):
    classification_model.fit(iris.X_train, iris.y_train)

    histogram = Histogram(
        "estimator_processing_seconds",
        "time processing in estimator",
        labelnames=["model_name"],
    )
    instrumentor = SklearnInstrumentor(instrument=PrometheusHistogram(histogram))
    dkwargs = {"labels": {"model_name": "classifier"}}
    instrumentor.instrument_estimator(classification_model, instrument_kwargs=dkwargs)

    def start_server():
        start_http_server(8000)

    thread = threading.Thread(target=start_server)
    thread.start()
    time.sleep(2)

    classification_model.predict(iris.X_test)

    response = requests.get("http://localhost:8000")
    text = """estimator_processing_seconds_bucket{le="0.1",model_name="classifier",qualname="Pipeline.predict"}"""
    assert any([text in line for line in response.text.split("\n")])


def test_prometheus_histogram_multi(classification_model_multi, iris):
    classification_model_multi.fit(iris.X_train, iris.y_train)

    histogram_with_enum = Histogram(
        "estimator_with_enum_processing_seconds",
        "time processing in estimator",
        labelnames=["model_name"],
    )
    instrumentor = SklearnInstrumentor(
        instrument=PrometheusHistogram(histogram=histogram_with_enum, enumerate_=True),
        methods=["transform", "predict", "predict_proba"],
    )
    dkwargs = {"labels": {"model_name": "classifier"}}
    instrumentor.instrument_estimator(
        classification_model_multi, instrument_kwargs=dkwargs
    )

    def start_server():
        start_http_server(8000)

    thread = threading.Thread(target=start_server)
    thread.start()
    time.sleep(2)

    classification_model_multi.predict(iris.X_test)

    response = requests.get("http://localhost:8000")
    print(response.text)
    text = """estimator_with_enum_processing_seconds_bucket{le="0.1",model_name="classifier",qualname="StandardScaler.transform-1"}"""
    assert any([text in line for line in response.text.split("\n")])


def test_prometheus_summary(classification_model, iris):
    classification_model.fit(iris.X_train, iris.y_train)

    summary = Summary(
        "estimator_processing_s_seconds",
        "time processing in estimator",
        labelnames=["model_name"],
    )
    instrumentor = SklearnInstrumentor(instrument=PrometheusSummary(summary))
    dkwargs = {"labels": {"model_name": "classifier"}}
    instrumentor.instrument_estimator(classification_model, instrument_kwargs=dkwargs)

    def start_server():
        start_http_server(8000)

    thread = threading.Thread(target=start_server)
    thread.start()
    time.sleep(2)

    classification_model.predict(iris.X_test)

    response = requests.get("http://localhost:8000")
    text = """estimator_processing_s_seconds_count{model_name="classifier",qualname="StandardScaler.transform"}"""
    assert any([text in line for line in response.text.split("\n")])
    text = """estimator_processing_s_seconds_sum{model_name="classifier",qualname="StandardScaler.transform"}"""
    assert any([text in line for line in response.text.split("\n")])


def test_prometheus_summary_multi(classification_model_multi, iris):
    classification_model_multi.fit(iris.X_train, iris.y_train)

    summary = Summary(
        "estimator_with_enum_processing_s_seconds",
        "time processing in estimator",
        labelnames=["model_name"],
    )
    instrumentor = SklearnInstrumentor(
        instrument=PrometheusSummary(summary=summary, enumerate_=True)
    )
    dkwargs = {"labels": {"model_name": "classifier"}}
    instrumentor.instrument_estimator(
        classification_model_multi, instrument_kwargs=dkwargs
    )

    def start_server():
        start_http_server(8000)

    thread = threading.Thread(target=start_server)
    thread.start()
    time.sleep(2)

    classification_model_multi.predict(iris.X_test)

    response = requests.get("http://localhost:8000")
    text = """estimator_with_enum_processing_s_seconds_count{model_name="classifier",qualname="StandardScaler.transform-1"}"""
    assert any([text in line for line in response.text.split("\n")])
    text = """estimator_with_enum_processing_s_seconds_sum{model_name="classifier",qualname="StandardScaler.transform-1"}"""
    assert any([text in line for line in response.text.split("\n")])
