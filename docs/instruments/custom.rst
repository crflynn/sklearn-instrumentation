Custom Instruments
==================

The package comes with a handful of instruments which log information about ``X`` or timing of execution. You can create your own instrument just by creating a decorator, following this pattern:

.. code-block:: python

    from functools import wraps


    def my_instrument(obj func, **dkwargs):
        """Wrap an estimator method with instrumentation.

        :param obj: The class or instance on which to apply instrumentation
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


To create a stateful instrument, inherit from the ``BaseInstrument`` class and use the ``__call__`` method for implementing the decorator:

.. code-block:: python

    from functools import wraps

    from sklearn_instrumentation.instruments.base import BaseInstrument


    class MyInstrument(BaseInstrument)

        def __init__(self, *args, **kwargs):
            # handle any statefulness here
            pass

        def __call__(obj, self, func, **dkwargs):
            """Wrap an estimator method with instrumentation.

           :param obj: The class or instance on which to apply instrumentation
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


To pass decorator kwargs for different ml models using ``dkwargs``:

.. code-block:: python

    instrumentor = SklearnInstrumentor(instrument=my_instrument)

    instrumentor.instrument_instance(estimator=ml_model_1, decorator_kwargs={"name": "awesome_model"})
    instrumentor.instrument_instance(estimator=ml_model_2, decorator_kwargs={"name": "better_model"})


