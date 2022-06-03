Package Instrumentation
=======================

**Package** instrumentation allows you to instrument any ``sklearn`` compatible class, including classes found in other packages like ``xgboost``, ``lightgbm``, or other packages with transformers or estimator classes.

Package instrumentation works by crawling modules and submodules, dynamically importing any object and checking to see if it is a subclass of ``sklearn.base.BaseEstimator``. It ignores any modules with ``test`` in the name.

Since instrumentation is implemented by crawling, package instrumentation can take some time, and will take longer the more packages you include.

If you want to instrument prior to *fitting* a model, use **package** or **class** instrumentation. Generally, metaestimators like ``sklearn.pipeline.Pipeline`` use cloning when fitting, which won't retain instrumentation applied from **instance** instrumentation.

Examples
--------

Instrument all ``BaseEstimator``-derived classes from your favorite sklearn-compatible packages.

.. code-block:: python

    from sklearn_instrumentation import SklearnInstrumentor

    instrumentor = SklearnInstrumentor(instrument=my_instrument)
    instrumentor.instrument_packages(["sklearn", "xgboost", "lightgbm"])


Instrument the sklearn package. Then train and predict on a Pipeline model for classification.

.. code-block:: python

    import logging

    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sklearn_instrumentation import SklearnInstrumentor
    from sklearn_instrumentation.instruments.logging import TimeElapsedLogger

    logging.basicConfig(level=logging.INFO)

    # Create an instrumentor and instrument sklearn
    instrumentor = SklearnInstrumentor(instrument=TimeElapsedLogger())
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
    # INFO:sklearn_instrumentation.instruments.logging:Pipeline.fit starting.
    # INFO:sklearn_instrumentation.instruments.logging:Pipeline._fit starting.
    # INFO:sklearn_instrumentation.instruments.logging:StandardScaler.fit starting.
    # INFO:sklearn_instrumentation.instruments.logging:StandardScaler.fit elapsed time: 0.0006406307220458984 seconds
    # INFO:sklearn_instrumentation.instruments.logging:StandardScaler.transform starting.
    # INFO:sklearn_instrumentation.instruments.logging:StandardScaler.transform elapsed time: 0.0001430511474609375 seconds
    # INFO:sklearn_instrumentation.instruments.logging:PCA._fit starting.
    # INFO:sklearn_instrumentation.instruments.logging:PCA._fit elapsed time: 0.0006711483001708984 seconds
    # INFO:sklearn_instrumentation.instruments.logging:Pipeline._fit elapsed time: 0.0026731491088867188 seconds
    # INFO:sklearn_instrumentation.instruments.logging:BaseForest.fit starting.
    # INFO:sklearn_instrumentation.instruments.logging:BaseForest.fit elapsed time: 0.1768970489501953 seconds
    # INFO:sklearn_instrumentation.instruments.logging:Pipeline.fit elapsed time: 0.17983102798461914 seconds

    # Observe logging
    classification_model.predict(X)
    # INFO:sklearn_instrumentation.instruments.logging:Pipeline.predict starting.
    # INFO:sklearn_instrumentation.instruments.logging:FeatureUnion.transform starting.
    # INFO:sklearn_instrumentation.instruments.logging:StandardScaler.transform starting.
    # INFO:sklearn_instrumentation.instruments.logging:StandardScaler.transform elapsed time: 0.00024509429931640625 seconds
    # INFO:sklearn_instrumentation.instruments.logging:_BasePCA.transform starting.
    # INFO:sklearn_instrumentation.instruments.logging:_BasePCA.transform elapsed time: 0.0002181529998779297 seconds
    # INFO:sklearn_instrumentation.instruments.logging:FeatureUnion.transform elapsed time: 0.0012080669403076172 seconds
    # INFO:sklearn_instrumentation.instruments.logging:ForestClassifier.predict starting.
    # INFO:sklearn_instrumentation.instruments.logging:ForestClassifier.predict_proba starting.
    # INFO:sklearn_instrumentation.instruments.logging:ForestClassifier.predict_proba elapsed time: 0.013531208038330078 seconds
    # INFO:sklearn_instrumentation.instruments.logging:ForestClassifier.predict elapsed time: 0.013692140579223633 seconds
    # INFO:sklearn_instrumentation.instruments.logging:Pipeline.predict elapsed time: 0.015219926834106445 seconds

    # Remove instrumentation
    instrumentor.uninstrument_packages(["sklearn"])

    # Observe no logging
    classification_model.predict(X)

