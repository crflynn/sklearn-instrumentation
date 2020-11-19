PyInstrument Instruments
========================

To use PyInstrument instrumentation, install with the ``pyinstrument`` extra:

.. code-block:: bash

    pip install sklearn-instrumentation[pyinstrument]



Example usage:

.. code-block:: python

    from sklearn_instrumentation import SklearnInstrumentor
    from sklearn_instrumentation.instruments.pyinstrument import PyInstrumentProfiler

    profiler = PyInstrumentProfiler()
    instrumentor = SklearnInstrumentor(instrument=profiler)

    instrumentor.instrument_estimator(
        classification_model,
        instrument_kwargs={
            "profiler_kwargs": dict(interval=0.001),
            "text_kwargs": dict(show_all=True, unicode=True, color=True),
            "html_dir": "../htmlprof/",
        },
    )

    classification_model.fit(X, y)
    classification_model.predict(X)


Example output (partial):

.. code-block:: text

    PCA._fit

      _     ._   __/__   _ _  _  _ _/_   Recorded: 00:41:19  Samples:  45
     /_//_/// /_\ / //_// / //_'/ //     Duration: 0.322     CPU time: 1.617
    /   _/                      v3.2.0

    Program: /Users/user/projects/sklearn-instrumentation/examples/pyinstrument_.py

    0.322 wrapper  sklearn_instrumentation/instruments/pyinstrument.py:41
    └─ 0.320 _fit  sklearn/decomposition/_pca.py:388
       ├─ 0.314 _fit_truncated  sklearn/decomposition/_pca.py:496
       │  ├─ 0.287 inner_f  sklearn/utils/validation.py:59
       │  │  └─ 0.285 randomized_svd  sklearn/utils/extmath.py:246
       │  │     ├─ 0.249 inner_f  sklearn/utils/validation.py:59
       │  │     │  └─ 0.246 randomized_range_finder  sklearn/utils/extmath.py:161
       │  │     │     ├─ 0.151 lu  scipy/linalg/decomp_lu.py:150
       │  │     │     │  ├─ 0.140 [self]
       │  │     │     │  └─ 0.010 asarray_chkfinite  numpy/lib/function_base.py:422
       │  │     │     ├─ 0.052 inner_f  sklearn/utils/validation.py:59
       │  │     │     │  └─ 0.052 safe_sparse_dot  sklearn/utils/extmath.py:119
       │  │     │     ├─ 0.026 qr  scipy/linalg/decomp_qr.py:26
       │  │     │     │  └─ 0.025 safecall  scipy/linalg/decomp_qr.py:11
       │  │     │     └─ 0.016 [self]
       │  │     ├─ 0.028 svd_flip  sklearn/utils/extmath.py:501
       │  │     │  ├─ 0.016 argmax  .ignore/<__array_function__ internals>:2
       │  │     │  │  └─ 0.016 argmax  numpy/core/fromnumeric.py:1114
       │  │     │  │     └─ 0.016 _wrapfunc  numpy/core/fromnumeric.py:52
       │  │     │  │        └─ 0.016 ndarray.argmax  .ignore/<built-in>:0
       │  │     │  └─ 0.012 [self]
       │  │     └─ 0.009 dot  .ignore/<__array_function__ internals>:2
       │  │        └─ 0.009 implement_array_function  .ignore/<built-in>:0
       │  ├─ 0.020 var  .ignore/<__array_function__ internals>:2
       │  │  └─ 0.020 var  numpy/core/fromnumeric.py:3505
       │  │     └─ 0.019 _var  numpy/core/_methods.py:176
       │  │        ├─ 0.012 ufunc.reduce  .ignore/<built-in>:0
       │  │        └─ 0.007 [self]
       │  └─ 0.004 mean  .ignore/<__array_function__ internals>:2
       │     └─ 0.004 mean  numpy/core/fromnumeric.py:3269
       │        └─ 0.004 _mean  numpy/core/_methods.py:143
       │           └─ 0.004 ufunc.reduce  .ignore/<built-in>:0
       └─ 0.006 _validate_data  sklearn/base.py:383
          └─ 0.006 inner_f  sklearn/utils/validation.py:59
             └─ 0.006 check_array  sklearn/utils/validation.py:397
                └─ 0.006 array  .ignore/<built-in>:0


.. autoclass:: sklearn_instrumentation.instruments.pyinstrument.PyInstrumentProfiler
    :members: