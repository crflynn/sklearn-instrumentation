from io import StringIO

import pytest

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.memory_profiler import MemoryProfiler


@pytest.mark.skip(reason="unable to get profiler output in test; example works")
def test_memory_profiler(classification_model, iris, capsys):
    r"""sample output in stdout:

    PCA._fit
    Filename: /Users/user/projects/sklearn-instrumentation/.venv/lib/python3.8/site-packages/sklearn/decomposition/_pca.py

    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
       388     94.1 MiB     94.1 MiB           1       def _fit(self, X):
    """
    profiler = MemoryProfiler()
    instrumentor = SklearnInstrumentor(instrument=profiler)
    classification_model.fit(iris.X_train, iris.y_train)
    memory_profiler_header = (
        r"""Line #    Mem usage    Increment  Occurences   Line Contents"""
    )
    assert memory_profiler_header not in capsys.readouterr().out

    instrumentor.instrument_estimator(classification_model)
    classification_model.predict(iris.X_test)
    assert memory_profiler_header in capsys.readouterr().out

    instrumentor.uninstrument_estimator(classification_model)
    classification_model.predict(iris.X_test)
    assert memory_profiler_header not in capsys.readouterr().out
