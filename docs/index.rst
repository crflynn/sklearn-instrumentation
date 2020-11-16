sklearn-instrumentation Documentation
=====================================

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

Instrumentation works by applying decorators to methods of ``BaseEstimator``-derived classes or instances. By default the instrumentor applies instrumentation to the following methods (except when they are properties of instances):

* fit
* predict
* predict_proba
* transform
* _fit
* _predict
* _predict_proba
* _transform

**sklearn-instrumentation** supports instrumentation of full sklearn-compatible packages, as well as recursive instrumentation of models (metaestimators like ``Pipeline``, or even single estimators like ``RandomForestClassifier``)


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   package_instrumentation
   estimator_instrumentation
   instrumentor
   configuration
   instruments/base
   instruments/logging
   instruments/prometheus
   instruments/datadog
   instruments/opentelemetry
   instruments/custom
   utils



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
