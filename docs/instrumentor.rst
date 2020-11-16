Instrumentor
------------

The instrumentor class is used for **package**, **class**, and **estimator** instrumentation. Multiple instrumentors can be made to apply different instrumentation decorators. Different instruments can be applied separately to different machine learning models using **estimator** instrumentation.

Instrumenting and uninstrumenting is handled gracefully. Remaining decorators will be collected and reapplied after individual uninstrumentations.

Multiple decorators can be combined into a single instrumentation decorator using the ``sklearn_instrumentation.utils.composer`` helper function.

By default, instrumentation is never applied to classes or instances derived from ``sklearn.tree.BaseDecisionTree``. This can be overridden by the ``exclude`` arg on instantiation.

.. autoclass:: sklearn_instrumentation.instrumentor.SklearnInstrumentor
    :members:
