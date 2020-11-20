from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsd import StatsClient

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.statsd import StatsdTimer

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


if __name__ == "__main__":
    client = StatsClient()
    timer = StatsdTimer(client=client, enumerate_=True)
    instrumentor = SklearnInstrumentor(
        instrument=timer, instrument_kwargs={"prefix": "mymodel"}
    )
    instrumentor.instrument_estimator(classification_model)
    while True:
        classification_model.predict(X)
