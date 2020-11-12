sklearn-instrumentation
=======================

|actions| |rtd| |pypi| |pyversions|

.. |actions| image:: https://github.com/crflynn/sklearn-instrumentation/workflows/build/badge.svg
    :target: https://github.com/crflynn/sklearn-instrumentation/actions

.. |rtd| image:: https://img.shields.io/readthedocs/sklearn-instrumentation.svg
    :target: http://sklearn-instrumentation.readthedocs.io/en/latest/

.. |pypi| image:: https://img.shields.io/pypi/v/sklearn-instrumentation.svg
    :target: https://pypi.python.org/pypi/sklearn-instrumentation

.. |pyversions| image:: https://img.shields.io/pypi/pyversions/sklearn-instrumentation.svg
    :target: https://pypi.python.org/pypi/sklearn-instrumentation


Generalized instrumentation tooling for scikit-learn models. ``sklearn_instrumentation`` allows instrumenting the ``sklearn`` package and any scikit-learn compatible packages with estimators and transformers inheriting from ``sklearn.base.BaseEstimator``.

Instrumentation applies decorators to methods of ``BaseEstimator``-derived classes or instances. By default the instrumentor applies instrumentation to the following methods (except when they are properties):

* fit
* predict
* predict_proba
* transform
* _fit
* _predict
* _predict_proba
* _transform

**sklearn-instrumentation** supports instrumentation of full sklearn-compatible packages, as well as recursive instrumentation of models (metaestimators like ``Pipeline``, or even single estimators like ``RandomForestClassifier``)

Package instrumentation
-----------------------

Instrument any sklearn compatible package that has ``BaseEstimator``-derived classes.

.. code-block:: python

    from sklearn_instrumentation import SklearnInstrumentor

    instrumentor = SklearnInstrumentor(decorator=my_instrumentation)
    instrumentor.instrument_packages(["sklearn", "xgboost", "lightgbm"])


Full example:

.. code-block:: python

    import logging

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sklearn_instrumentation import SklearnInstrumentor
    from sklearn_instrumentation.instrumentation.logging import time_elapsed_logger

    logging.basicConfig(level=logging.INFO)

    # Create an instrumentor and instrument sklearn
    instrumentor = SklearnInstrumentor(decorator=time_elapsed_logger)
    instrumentor.instrument_packages(["sklearn"])

    # Create a toy model for classification
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

    # Observe logging
    classification_model.fit(X, y)
    # INFO:root:Pipeline.fit starting.
    # INFO:root:Pipeline._fit starting.
    # INFO:root:StandardScaler.fit starting.
    # INFO:root:StandardScaler.fit elapsed time: 0.0006406307220458984 seconds
    # INFO:root:StandardScaler.transform starting.
    # INFO:root:StandardScaler.transform elapsed time: 0.0001430511474609375 seconds
    # INFO:root:PCA._fit starting.
    # INFO:root:PCA._fit elapsed time: 0.0006711483001708984 seconds
    # INFO:root:Pipeline._fit elapsed time: 0.0026731491088867188 seconds
    # INFO:root:BaseForest.fit starting.
    # INFO:root:BaseForest.fit elapsed time: 0.1768970489501953 seconds
    # INFO:root:Pipeline.fit elapsed time: 0.17983102798461914 seconds

    # Observe logging
    classification_model.predict(X)
    # INFO:root:Pipeline.predict starting.
    # INFO:root:FeatureUnion.transform starting.
    # INFO:root:StandardScaler.transform starting.
    # INFO:root:StandardScaler.transform elapsed time: 0.00024509429931640625 seconds
    # INFO:root:_BasePCA.transform starting.
    # INFO:root:_BasePCA.transform elapsed time: 0.0002181529998779297 seconds
    # INFO:root:FeatureUnion.transform elapsed time: 0.0012080669403076172 seconds
    # INFO:root:ForestClassifier.predict starting.
    # INFO:root:ForestClassifier.predict_proba starting.
    # INFO:root:ForestClassifier.predict_proba elapsed time: 0.013531208038330078 seconds
    # INFO:root:ForestClassifier.predict elapsed time: 0.013692140579223633 seconds
    # INFO:root:Pipeline.predict elapsed time: 0.015219926834106445 seconds

    # Remove instrumentation
    instrumentor.uninstrument_packages(["sklearn"])

    # Observe no logging
    classification_model.predict(X)


Machine learning model instrumentation
--------------------------------------

Instrument any sklearn compatible trained estimator or metaestimator.

.. code-block:: python

    from sklearn_instrumentation import SklearnInstrumentor

    instrumentor = SklearnInstrumentor(decorator=my_instrumentation)
    instrumentor.instrument_estimator(estimator=my_ml_pipeline)


.. code-block:: python

    import logging

    from sklearn.datasets import load_iris
    from sklearn_instrumentation import SklearnInstrumentor
    from sklearn_instrumentation.instrumentation.logging import time_elapsed_logger
    from sklearn.ensemble import RandomForestClassifier

    logging.basicConfig(level=logging.INFO)

    # Train a classifier
    X, y = load_iris(return_X_y=True)
    rf = RandomForestClassifier()

    rf.fit(X, y)

    # Create an instrumentor which decorates BaseEstimator methods with
    # logging output when entering and exiting methods, with time elapsed logged
    # on exit.
    instrumentor = SklearnInstrumentor(decorator=time_elapsed_logger)

    # Apply the decorator to all BaseEstimators in each of these libraries
    instrumentor.instrument_estimator(rf)

    # Observe the logging output
    rf.predict(X)
    # INFO:root:ForestClassifier.predict starting.
    # INFO:root:ForestClassifier.predict_proba starting.
    # INFO:root:ForestClassifier.predict_proba elapsed time: 0.014165163040161133 seconds
    # INFO:root:ForestClassifier.predict elapsed time: 0.014327764511108398 seconds

    # Remove the decorator from all BaseEstimators in each of these libraries
    instrumentor.uninstrument_estimator(rf)

    # No more logging
    rf.predict(X)


Instrumentation
---------------

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


To pass kwargs for different ml models:

.. code-block:: python

    instrumentor = SklearnInstrumentor(decorator=my_instrumentation)

    instrumentor.instrument_estimator(estimator=ml_model_1, decorator_kwargs={"name": "awesome_model"})
    instrumentor.instrument_estimator(estimator=ml_model_2, decorator_kwargs={"name": "better_model"})

