Class Instrumentation
=====================

**Class** instrumentation allows you to instrument any ``sklearn`` compatible class which is a component of an estimator instance (or its metaestimator hierarchy).

Class instrumentation crawls the estimator instance's hierarchy, instrumenting only the objects' classes which subclasses of ``sklearn.base.BaseEstimator``.

This is similar to **instance** instrumentation, except we instrument the estimators' classes rather than the estimator instances. Class instrumentation is ideal when performing fit operations, due to the copying/cloning that sometimes happens within sklearn metaestimators.

In general, **class** instrumentation will be faster and consume less memory than **package** instrumentation.

Examples
--------

Instrument an estimator's classes. Inspect the memory usage of the process before and after class instrumentation compared to package instrumentation. Also measure the time elapsed.

.. code-block:: python

    import logging
    import os
    import time

    import psutil
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from sklearn_instrumentation import SklearnInstrumentor
    from sklearn_instrumentation.instruments.logging import TimeElapsedLogger

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

    instrumentor = SklearnInstrumentor(instrument=TimeElapsedLogger())

    process = psutil.Process(os.getpid())
    print("Memory before instrumentation: " + str(process.memory_info().rss))
    start = time.time()
    instrumentor.instrument_instance_classes(classification_model)
    print("Time elapsed class instrumentation: " + str(time.time() - start))
    print("Memory after class instrumentation: " + str(process.memory_info().rss))
    instrumentor.uninstrument_instance_classes(classification_model)
    start = time.time()
    instrumentor.instrument_package("sklearn")
    print("Time elapsed package instrumentation: " + str(time.time() - start))
    print("Memory after package instrumentation: " + str(process.memory_info().rss))

    # Memory before instrumentation: 76288000
    # Time elapsed class instrumentation: 0.0013790130615234375
    # Memory after class instrumentation: 76353536
    # Time elapsed package instrumentation: 0.2482309341430664
    # Memory after package instrumentation: 89726976

