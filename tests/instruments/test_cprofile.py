from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.cprofile import CProfiler


def test_cprofiler(classification_model, iris, capsys):
    instrumentor = SklearnInstrumentor(instrument=CProfiler())
    classification_model.fit(iris.X_train, iris.y_train)
    cprofile_line = """Ordered by: standard name"""
    assert cprofile_line not in capsys.readouterr().out

    instrumentor.instrument_instance(classification_model)
    classification_model.predict(iris.X_test)
    assert cprofile_line in capsys.readouterr().out

    instrumentor.uninstrument_instance(classification_model)
    classification_model.predict(iris.X_test)
    assert cprofile_line not in capsys.readouterr().out
