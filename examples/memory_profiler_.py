from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.memory_profiler import MemoryProfiler

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
    profiler = MemoryProfiler()
    instrumentor = SklearnInstrumentor(instrument=profiler)

    instrumentor.instrument_instance(
        classification_model,
    )

    classification_model.fit(X, y)

    classification_model.predict(X)
