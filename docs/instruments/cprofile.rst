cProfile Instruments
====================

Instrument function calls with cProfile, optionally writing stats dumps to disk. Stats dumps can be used with tools like ``snakeviz`` for profile visualization.

Example usage:

.. code-block:: python

    from sklearn_instrumentation import SklearnInstrumentor
    from sklearn_instrumentation.instruments.cprofile import CProfiler

    profiler = CProfiler()
    instrumentor = SklearnInstrumentor(instrument=profiler)

    instrumentor.instrument_estimator(classification_model)

    classification_model.fit(X, y)
    classification_model.predict(X)


Example output (partial):

.. code-block:: text

    StandardScaler.fit
             256 function calls (240 primitive calls) in 0.001 seconds

       Ordered by: standard name

       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
            1    0.000    0.000    0.000    0.000 <__array_function__ internals>:2(any)
            3    0.000    0.000    0.000    0.000 <__array_function__ internals>:2(copyto)
            1    0.000    0.000    0.000    0.000 <__array_function__ internals>:2(nansum)
            1    0.000    0.000    0.000    0.000 <__array_function__ internals>:2(nanvar)
            1    0.000    0.000    0.000    0.000 <__array_function__ internals>:2(ptp)
            6    0.000    0.000    0.000    0.000 <__array_function__ internals>:2(sum)
            1    0.000    0.000    0.000    0.000 <frozen importlib._bootstrap>:389(parent)
            1    0.000    0.000    0.000    0.000 _asarray.py:14(asarray)
            3    0.000    0.000    0.000    0.000 _asarray.py:86(asanyarray)
            1    0.000    0.000    0.000    0.000 _config.py:14(get_config)
            1    0.000    0.000    0.000    0.000 _data.py:63(_handle_zeros_in_scale)
            1    0.000    0.000    0.000    0.000 _data.py:638(_reset)
            1    0.000    0.000    0.001    0.001 _data.py:652(fit)
            1    0.000    0.000    0.001    0.001 _data.py:669(partial_fit)
            1    0.000    0.000    0.000    0.000 _data.py:850(_more_tags)
            1    0.000    0.000    0.000    0.000 _methods.py:245(_ptp)
            6    0.000    0.000    0.000    0.000 _ufunc_config.py:132(geterr)
            6    0.000    0.000    0.000    0.000 _ufunc_config.py:32(seterr)
            3    0.000    0.000    0.000    0.000 _ufunc_config.py:429(__init__)
            3    0.000    0.000    0.000    0.000 _ufunc_config.py:433(__enter__)
            3    0.000    0.000    0.000    0.000 _ufunc_config.py:438(__exit__)
          6/2    0.000    0.000    0.000    0.000 abc.py:100(__subclasscheck__)
            2    0.000    0.000    0.000    0.000 abc.py:96(__instancecheck__)
            2    0.000    0.000    0.000    0.000 base.py:1188(isspmatrix)
            1    0.000    0.000    0.000    0.000 base.py:340(_more_tags)
            1    0.000    0.000    0.000    0.000 base.py:343(_get_tags)
            1    0.000    0.000    0.000    0.000 base.py:354(_check_n_features)
            1    0.000    0.000    0.000    0.000 base.py:383(_validate_data)
            3    0.000    0.000    0.000    0.000 extmath.py:686(_safe_accumulator_op)
            1    0.000    0.000    0.000    0.000 extmath.py:715(_incremental_mean_and_var)
            6    0.000    0.000    0.000    0.000 fromnumeric.py:2100(_sum_dispatcher)
            6    0.000    0.000    0.000    0.000 fromnumeric.py:2105(sum)
            1    0.000    0.000    0.000    0.000 fromnumeric.py:2245(_any_dispatcher)
            1    0.000    0.000    0.000    0.000 fromnumeric.py:2249(any)
            1    0.000    0.000    0.000    0.000 fromnumeric.py:2486(_ptp_dispatcher)
            1    0.000    0.000    0.000    0.000 fromnumeric.py:2490(ptp)
            7    0.000    0.000    0.000    0.000 fromnumeric.py:70(_wrapreduction)
            7    0.000    0.000    0.000    0.000 fromnumeric.py:71(<dictcomp>)
            1    0.000    0.000    0.000    0.000 inspect.py:2845(parameters)
            1    0.000    0.000    0.000    0.000 inspect.py:487(getmro)
            3    0.000    0.000    0.000    0.000 multiarray.py:1043(copyto)
            1    0.000    0.000    0.000    0.000 nanfunctions.py:113(_copyto)
            1    0.000    0.000    0.000    0.000 nanfunctions.py:1416(_nanvar_dispatcher)
            1    0.000    0.000    0.000    0.000 nanfunctions.py:1421(nanvar)
            2    0.000    0.000    0.000    0.000 nanfunctions.py:183(_divide_by_count)
            1    0.000    0.000    0.000    0.000 nanfunctions.py:553(_nansum_dispatcher)
            1    0.000    0.000    0.000    0.000 nanfunctions.py:557(nansum)
            2    0.000    0.000    0.000    0.000 nanfunctions.py:68(_replace_nan)
            1    0.000    0.000    0.000    0.000 numeric.py:1816(isscalar)
            6    0.000    0.000    0.000    0.000 numerictypes.py:286(issubclass_)
            3    0.000    0.000    0.000    0.000 numerictypes.py:360(issubdtype)
            1    0.000    0.000    0.000    0.000 validation.py:180(_num_samples)
            1    0.000    0.000    0.000    0.000 validation.py:390(_ensure_no_complex_data)
            1    0.000    0.000    0.000    0.000 validation.py:397(check_array)
            1    0.000    0.000    0.000    0.000 validation.py:59(inner_f)
            1    0.000    0.000    0.000    0.000 validation.py:71(<dictcomp>)
            1    0.000    0.000    0.000    0.000 validation.py:76(_assert_all_finite)
            1    0.000    0.000    0.000    0.000 warnings.py:165(simplefilter)
            1    0.000    0.000    0.000    0.000 warnings.py:181(_add_filter)
            1    0.000    0.000    0.000    0.000 warnings.py:437(__init__)
            1    0.000    0.000    0.000    0.000 warnings.py:458(__enter__)
            1    0.000    0.000    0.000    0.000 warnings.py:477(__exit__)
            2    0.000    0.000    0.000    0.000 {built-in method _abc._abc_instancecheck}
          6/2    0.000    0.000    0.000    0.000 {built-in method _abc._abc_subclasscheck}
            3    0.000    0.000    0.000    0.000 {built-in method _warnings._filters_mutated}
            1    0.000    0.000    0.000    0.000 {built-in method builtins.getattr}
           16    0.000    0.000    0.000    0.000 {built-in method builtins.hasattr}
           19    0.000    0.000    0.000    0.000 {built-in method builtins.isinstance}
           12    0.000    0.000    0.000    0.000 {built-in method builtins.issubclass}
            3    0.000    0.000    0.000    0.000 {built-in method builtins.len}
            6    0.000    0.000    0.000    0.000 {built-in method numpy.array}
         13/5    0.000    0.000    0.000    0.000 {built-in method numpy.core._multiarray_umath.implement_array_function}
           12    0.000    0.000    0.000    0.000 {built-in method numpy.geterrobj}
            6    0.000    0.000    0.000    0.000 {built-in method numpy.seterrobj}
            1    0.000    0.000    0.000    0.000 {built-in method numpy.zeros}
            1    0.000    0.000    0.000    0.000 {method 'copy' of 'dict' objects}
            1    0.000    0.000    0.000    0.000 {method 'copy' of 'numpy.ndarray' objects}
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}
            1    0.000    0.000    0.000    0.000 {method 'get' of 'dict' objects}
            1    0.000    0.000    0.000    0.000 {method 'insert' of 'list' objects}
            7    0.000    0.000    0.000    0.000 {method 'items' of 'dict' objects}
            9    0.000    0.000    0.000    0.000 {method 'reduce' of 'numpy.ufunc' objects}
            1    0.000    0.000    0.000    0.000 {method 'remove' of 'list' objects}
            1    0.000    0.000    0.000    0.000 {method 'rpartition' of 'str' objects}
            1    0.000    0.000    0.000    0.000 {method 'squeeze' of 'numpy.ndarray' objects}
            3    0.000    0.000    0.000    0.000 {method 'update' of 'dict' objects}


.. autoclass:: sklearn_instrumentation.instruments.cprofile.CProfiler
