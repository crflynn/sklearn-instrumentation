Instrumentor
------------

The instrumentor class can be used for both **package** and **estimator** instrumentation. Multiple instrumentors can be made to apply different instrumentation decorators. Different instrumentors can be applied separately to different machine learning models.

Instrumenting and uninstrumenting is handled gracefully. Even though instrumentation uses decorators, the order of instrumentation does not matter when applying uninstrumentation. Remaining decorators will be collected and reapplied after individual uninstrumentations.

Multiple decorators can be combined into a single instrumentation decorator using the ``sklearn_instrumentation.utils.composer`` helper function.

By default instrumentation is never applied to classes or instances derived from ``sklearn.tree.BaseDecisionTree``. This can be overridden by the ``exclude`` arg on instantiation.

.. autoclass:: sklearn_instrumentation.instrumentor.SklearnInstrumentor
    :members:
