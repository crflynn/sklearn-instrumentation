Custom Instrumentation
======================

The package comes with a handful of decorators which log information about ``X`` or timing of execution. You can create your own instrumentation just by creating a decorator, following this pattern

.. code-block:: python

    from functools import wraps


    def my_instrumentation(func, **dkwargs):
        """Wrap an estimator method with instrumentation.

        :param func: The method to be instrumented.
        :param dkwargs: Decorator kwargs, which can be passed to the
            decorator at decoration time. For estimator instrumentation
            this allows different parametrizations for each ml model.
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapping function.

            :param args: The args passed to methods, typically
                just ``X`` and/or ``y``
            :param kwargs: The kwargs passed to methods, usually
                weights or other params
            """
            # Code goes here before execution of the estimator method
            retval = func(*args, **kwargs)
            # Code goes here after execution of the estimator method
            return retval

        return wrapper


To pass kwargs for different ml models using ``dkwargs``:

.. code-block:: python

    instrumentor = SklearnInstrumentor(decorator=my_instrumentation)

    instrumentor.instrument_estimator(estimator=ml_model_1, decorator_kwargs={"name": "awesome_model"})
    instrumentor.instrument_estimator(estimator=ml_model_2, decorator_kwargs={"name": "better_model"})


