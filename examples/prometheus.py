import logging

from prometheus_client import Histogram
from prometheus_client import Summary
from prometheus_client import start_http_server
from prometheus_client.utils import INF
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.logging import TimeElapsedLogger
from sklearn_instrumentation.instruments.prometheus import PrometheusHistogram
from sklearn_instrumentation.instruments.prometheus import PrometheusSummary

histogram = Histogram(
    "estimator_processing_seconds",
    "Time estimator spent processing",
    labelnames=["model_name"],
    buckets=(
        0.0001,
        0.00025,
        0.0005,
        0.00075,
        0.001,
        0.0025,
        0.005,
        0.0075,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        INF,
    ),
)
prom_histogram = SklearnInstrumentor(
    instrument=PrometheusHistogram(histogram=histogram, enumerate_=True),
    methods=["transform", "predict", "predict_proba"],
)
# prom_summary = SklearnInstrumentor(
#     instrument=PrometheusSummary(enumerate_=True),
#     methods=["transform", "predict", "predict_proba"],
# )

te_instrumentor = SklearnInstrumentor(instrument=TimeElapsedLogger())


ss = StandardScaler()
ss2 = StandardScaler()
pca = PCA(n_components=3)
pca2 = PCA(n_components=3)
rf = RandomForestClassifier()
classification_model = Pipeline(
    steps=[
        (
            "fu",
            FeatureUnion(
                transformer_list=[
                    ("ss", ss),
                    ("ss2", ss2),
                    ("pca", pca),
                    ("pca2", pca2),
                ]
            ),
        ),
        ("rf", rf),
    ]
)
X, y = load_iris(return_X_y=True)
classification_model.fit(X, y)

prom_histogram.instrument_instance(
    classification_model, instrument_kwargs={"labels": {"model_name": "mymodel"}}
)
# prom_summary.instrument_instance(
#     classification_model, instrument_kwargs={"labels": {"model_name": "mymodel"}}
# )
te_instrumentor.instrument_instance(classification_model)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Start up the server to expose the metrics at localhost:8000
    start_http_server(8000)
    # Generate some predictions.
    while True:
        classification_model.predict(X)
