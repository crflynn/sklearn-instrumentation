Instrumentor
------------

The instrumentor class is used for **package**, **class**, and **instance** instrumentation. Multiple instrumentors can be made to apply different instrumentation decorators. Different instruments can be applied separately to different machine learning models using **instance** instrumentation.

By default, instrumentation is never applied to classes or instances derived from ``sklearn.tree.BaseDecisionTree``. This can be overridden by the ``exclude`` arg on instantiation.

.. autoclass:: sklearn_instrumentation.instrumentor.SklearnInstrumentor
    :members:
