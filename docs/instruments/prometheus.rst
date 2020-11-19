Prometheus Instruments
======================

To use Prometheus instrumentation, install with the ``prometheus`` extra:

.. code-block:: bash

    pip install sklearn-instrumentation[prometheus]


Example usage:

.. code-block:: python

    from prometheus_client import Histogram
    from prometheus_client.utils import INF
    from sklearn_instrumentation import SklearnInstrumentor
    from sklearn_instrumentation.instruments.prometheus import PrometheusHistogram

    histogram = Histogram(
        "estimator_processing_seconds",
        "Time estimator spent processing",
        labelnames=["model_name"],
        buckets=(
            0.0001,
            0.00025,
            0.0005,
            0.00075,
            0.001,
            0.0025,
            0.005,
            0.0075,
            0.01,
            0.025,
            0.05,
            0.075,
            0.1,
            INF,
        ),
    )
    instrumentor = SklearnInstrumentor(
        instrument=PrometheusHistogram(histogram=histogram, enumerate_=True),
    )

    classification_model.fit(X, y)

    instrumentor.instrument_estimator(
        classification_model, instrument_kwargs={"labels": {"model_name": "mymodel"}}
    )

    classification_model.predict(X)


Example ``/metrics`` output:

.. code-block:: text

    # HELP estimator_processing_seconds Time estimator spent processing
    # TYPE estimator_processing_seconds histogram
    estimator_processing_seconds_bucket{le="0.0001",model_name="mymodel",qualname="Pipeline.predict-0"} 0.0
    estimator_processing_seconds_bucket{le="0.00025",model_name="mymodel",qualname="Pipeline.predict-0"} 0.0
    estimator_processing_seconds_bucket{le="0.0005",model_name="mymodel",qualname="Pipeline.predict-0"} 0.0
    estimator_processing_seconds_bucket{le="0.00075",model_name="mymodel",qualname="Pipeline.predict-0"} 0.0
    estimator_processing_seconds_bucket{le="0.001",model_name="mymodel",qualname="Pipeline.predict-0"} 0.0
    estimator_processing_seconds_bucket{le="0.0025",model_name="mymodel",qualname="Pipeline.predict-0"} 0.0
    estimator_processing_seconds_bucket{le="0.005",model_name="mymodel",qualname="Pipeline.predict-0"} 0.0
    estimator_processing_seconds_bucket{le="0.0075",model_name="mymodel",qualname="Pipeline.predict-0"} 0.0
    estimator_processing_seconds_bucket{le="0.01",model_name="mymodel",qualname="Pipeline.predict-0"} 0.0
    estimator_processing_seconds_bucket{le="0.025",model_name="mymodel",qualname="Pipeline.predict-0"} 98.0
    estimator_processing_seconds_bucket{le="0.05",model_name="mymodel",qualname="Pipeline.predict-0"} 154.0
    estimator_processing_seconds_bucket{le="0.075",model_name="mymodel",qualname="Pipeline.predict-0"} 156.0
    estimator_processing_seconds_bucket{le="0.1",model_name="mymodel",qualname="Pipeline.predict-0"} 156.0
    estimator_processing_seconds_bucket{le="+Inf",model_name="mymodel",qualname="Pipeline.predict-0"} 156.0
    estimator_processing_seconds_count{model_name="mymodel",qualname="Pipeline.predict-0"} 156.0
    estimator_processing_seconds_sum{model_name="mymodel",qualname="Pipeline.predict-0"} 3.593184422000001
    estimator_processing_seconds_bucket{le="0.0001",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.00025",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.0005",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.00075",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.001",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.0025",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.005",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.0075",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.01",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.025",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.05",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.075",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.1",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="+Inf",model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_count{model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_sum{model_name="mymodel",qualname="Pipeline.predict_proba-0"} 0.0
    estimator_processing_seconds_bucket{le="0.0001",model_name="mymodel",qualname="FeatureUnion.transform-0"} 0.0
    estimator_processing_seconds_bucket{le="0.00025",model_name="mymodel",qualname="FeatureUnion.transform-0"} 0.0
    estimator_processing_seconds_bucket{le="0.0005",model_name="mymodel",qualname="FeatureUnion.transform-0"} 0.0
    estimator_processing_seconds_bucket{le="0.00075",model_name="mymodel",qualname="FeatureUnion.transform-0"} 0.0
    estimator_processing_seconds_bucket{le="0.001",model_name="mymodel",qualname="FeatureUnion.transform-0"} 0.0
    estimator_processing_seconds_bucket{le="0.0025",model_name="mymodel",qualname="FeatureUnion.transform-0"} 111.0
    estimator_processing_seconds_bucket{le="0.005",model_name="mymodel",qualname="FeatureUnion.transform-0"} 130.0
    estimator_processing_seconds_bucket{le="0.0075",model_name="mymodel",qualname="FeatureUnion.transform-0"} 146.0
    estimator_processing_seconds_bucket{le="0.01",model_name="mymodel",qualname="FeatureUnion.transform-0"} 155.0
    estimator_processing_seconds_bucket{le="0.025",model_name="mymodel",qualname="FeatureUnion.transform-0"} 157.0
    estimator_processing_seconds_bucket{le="0.05",model_name="mymodel",qualname="FeatureUnion.transform-0"} 157.0
    estimator_processing_seconds_bucket{le="0.075",model_name="mymodel",qualname="FeatureUnion.transform-0"} 157.0
    estimator_processing_seconds_bucket{le="0.1",model_name="mymodel",qualname="FeatureUnion.transform-0"} 157.0
    estimator_processing_seconds_bucket{le="+Inf",model_name="mymodel",qualname="FeatureUnion.transform-0"} 157.0
    estimator_processing_seconds_count{model_name="mymodel",qualname="FeatureUnion.transform-0"} 157.0
    estimator_processing_seconds_sum{model_name="mymodel",qualname="FeatureUnion.transform-0"} 0.4322356809999992
    ...


.. autoclass:: sklearn_instrumentation.instruments.prometheus.PrometheusHistogram

.. autoclass:: sklearn_instrumentation.instruments.prometheus.PrometheusSummary
