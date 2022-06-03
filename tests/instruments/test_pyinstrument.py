from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.pyinstrument import PyInstrumentProfiler


def test_py_instrument_profiler(classification_model, iris, capsys):
    r"""sample output in stdout:

      _     ._   __/__   _ _  _  _ _/_   Recorded: 02:05:05  Samples:  4
     /_//_/// /_\ / //_// / //_'/ //     Duration: 0.011     CPU time: 0.011
    /   _/                      v3.2.0"""
    instrumentor = SklearnInstrumentor(instrument=PyInstrumentProfiler())
    classification_model.fit(iris.X_train, iris.y_train)
    pyinstrument_midline = r""" /_//_/// /_\ / //_// / //_'/ //"""
    assert pyinstrument_midline not in capsys.readouterr().out

    instrumentor.instrument_instance(classification_model)
    classification_model.predict(iris.X_test)
    assert pyinstrument_midline in capsys.readouterr().out

    instrumentor.uninstrument_instance(classification_model)
    classification_model.predict(iris.X_test)
    assert pyinstrument_midline not in capsys.readouterr().out
