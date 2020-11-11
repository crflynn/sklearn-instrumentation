Package Instrumentation
=======================

**Package** instrumentation allows you to instrument any ``sklearn`` compatible estimator, including estimators found in other packages like ``xgboost``, ``lightgbm``, or other packages which offer helpful transformers.

Package instrumentation works by crawling modules and submodules, dynamically importing any object and checking to see if it is a subclass of ``sklearn.base.BaseEstimator``. It ignores any modules with ``test`` in the name.

Since instrumentation is implemented by crawling, package instrumentation can sometimes take a few seconds, and will take longer the more packages you include.

If you want to instrument prior to *fitting* a model, use **package** instrumentation. Certain metaestimators, like ``sklearn.pipeline.Pipeline`` use cloning when fitting, which won't retain instrumentation applied from **estimator** instrumentation.

Examples
--------

Instrument any sklearn compatible package that has ``BaseEstimator``-derived classes.

.. code-block:: python

    from sklearn_instrumentation import SklearnInstrumentor

    instrumentor = SklearnInstrumentor(decorator=my_instrumentation)
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

