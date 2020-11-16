Estimator Instrumentation
=========================

**Estimator** instrumentation means instrumenting *instances* of sklearn estimators.

Estimator instrumentation works by crawling the attribute hierarchy of the passed estimator. This enables instrumentation of metaestimators like ``Pipeline``.

On metaestimators, estimator instrumentation should be applied after fitting. This is because metaestimators like ``Pipeline`` clone underlying estimators during fitting. The cloning process will cause the pre-fit instrumentation to disappear on some estimators. If you want instrumentation while fitting, use the **package** instrumentation or **class** instrumentation.

If you want to instrument different machine learning models differently, then use the **estimator** instrumentation. You can create multiple instrumentors, and apply them individually to different models.

Examples
--------

Instrument any sklearn compatible trained estimator or metaestimator.

.. code-block:: python

    from sklearn_instrumentation import SklearnInstrumentor

    instrumentor = SklearnInstrumentor(instrument=my_instrumentation)
    instrumentor.instrument_estimator(estimator=my_ml_pipeline)


Apply instrumentation to a classifier after fitting, and then remove it.

.. code-block:: python

    import logging

    from sklearn.datasets import load_iris
    from sklearn_instrumentation import SklearnInstrumentor
    from sklearn_instrumentation.instruments.logging import TimeElapsedLogger
    from sklearn.ensemble import RandomForestClassifier

    logging.basicConfig(level=logging.INFO)

    # Train a classifier
    X, y = load_iris(return_X_y=True)
    rf = RandomForestClassifier()

    rf.fit(X, y)

    # Create an instrumentor which decorates BaseEstimator methods with
    # logging output when entering and exiting methods, with time elapsed logged
    # on exit.
    instrumentor = SklearnInstrumentor(instrument=TimeElapsedLogger())

    # Apply the decorator to all BaseEstimators in each of these libraries
    instrumentor.instrument_estimator(rf)

    # Observe the logging output
    rf.predict(X)
    # INFO:sklearn_instrumentation.instruments.logging:ForestClassifier.predict starting.
    # INFO:sklearn_instrumentation.instruments.logging:ForestClassifier.predict_proba starting.
    # INFO:sklearn_instrumentation.instruments.logging:ForestClassifier.predict_proba elapsed time: 0.014165163040161133 seconds
    # INFO:sklearn_instrumentation.instruments.logging:ForestClassifier.predict elapsed time: 0.014327764511108398 seconds

    # Remove the decorator our classifier
    instrumentor.uninstrument_estimator(rf)

    # No more logging
    rf.predict(X)

