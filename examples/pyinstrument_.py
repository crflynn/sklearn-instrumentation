import logging

import numpy as np
from pyinstrument import Profiler
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.pyinstrument import PyInstrumentProfiler

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
                ],
            ),
        ),
        ("rf", rf),
    ],
)
X, y = load_iris(return_X_y=True)

if __name__ == "__main__":
    profiler = PyInstrumentProfiler()
    instrumentor = SklearnInstrumentor(instrument=profiler)

    instrumentor.instrument_estimator(
        classification_model,
        instrument_kwargs={
            "profiler_kwargs": {"interval": 0.001},
            "text_kwargs": dict(show_all=True, unicode=True, color=True),
            "html_dir": "../.ignore/",
        },
    )

    # bigger to make the operations take longer
    Xr = np.repeat(X, 2000, axis=0)
    yr = np.repeat(y, 2000, axis=0)

    classification_model.fit(Xr, yr)

    classification_model.predict(Xr)
