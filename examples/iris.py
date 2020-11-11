import logging

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instrumentation.logging import time_elapsed_logger

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
instrumentor = SklearnInstrumentor(decorator=time_elapsed_logger)
classification_model.fit(X, y)

instrumentor.instrument_estimator(classification_model)
classification_model.predict(pd.DataFrame(X))

instrumentor.uninstrument_estimator(classification_model)
classification_model.predict(pd.DataFrame(X))
