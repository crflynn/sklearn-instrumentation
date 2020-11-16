import logging
import os
import time

import psutil
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.logging import TimeElapsedLogger

logging.basicConfig(level=logging.INFO)

ss = StandardScaler()
pca = PCA(n_components=3)
rf = RandomForestClassifier()
classification_model = Pipeline(
    steps=[
        (
            "fu",
            FeatureUnion(
                transformer_list=[
                    ("ss", ss),
                    ("pca", pca),
                ]
            ),
        ),
        ("rf", rf),
    ]
)
X, y = load_iris(return_X_y=True)
classification_model.fit(X, y)

instrumentor = SklearnInstrumentor(instrument=TimeElapsedLogger())

process = psutil.Process(os.getpid())
print("Memory before instrumentation: " + str(process.memory_info().rss))
start = time.time()
instrumentor.instrument_estimator_classes(classification_model)
print("Time elapsed class instrumentation: " + str(time.time() - start))
print("Memory after class instrumentation: " + str(process.memory_info().rss))
instrumentor.uninstrument_estimator_classes(classification_model)
start = time.time()
instrumentor.instrument_package("sklearn")
print("Time elapsed package instrumentation: " + str(time.time() - start))
print("Memory after package instrumentation: " + str(process.memory_info().rss))
