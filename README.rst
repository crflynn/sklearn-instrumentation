sklearn-instrumentation
=======================

Generalized instrumentation tooling for scikit-learn models. ``sklearn_instrumentation`` allows instrumenting the ``sklearn`` library and any scikit-learn compatible libraries with estimators and transformers inheriting from ``sklearn.base.BaseEstimator.

Instrumentation applies decorators to methods ``BaseEstimator``-derived classes or instances. By default the instrumentor applies instrumentation to the following methods (if they exist):

* fit
* predict
* predict_proba
* transform
* _fit
* _predict
* _predict_proba
* _transform

Package instrumentation
---------------------------------------

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
    classification_model.fit(X, y)

    instrumentor = SklearnInstrumentor(decorator=time_elapsed_logger)
    instrumentor.instrument_packages(["sklearn"])

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
