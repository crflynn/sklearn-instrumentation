Release Changelog
-----------------

0.6.1 (2021-04-17)
~~~~~~~~~~~~~~~~~~

* Removed an extra type exclusion check in instrumentor.

0.6.0 (2021-04-17)
~~~~~~~~~~~~~~~~~~

* Exclude bytes from instrumentation
* Move ignored types from instrumentor into ``DEFAULT_EXCLUDE``
* Add ``predict_log_proba`` and ``_predict_log_proba`` to ``DEFAULT_METHODS``

0.5.0 (2021-03-24)
~~~~~~~~~~~~~~~~~~

* Exclude enums from instrumentation

0.4.1 (2020-12-14)
~~~~~~~~~~~~~~~~~~

* Avoid instrumentor recursion against instrumentor-created attributes

0.4.0 (2020-12-05)
~~~~~~~~~~~~~~~~~~

* Add instrument for cProfile

0.3.0 (2020-11-19)
~~~~~~~~~~~~~~~~~~

* Add instruments for prometheus, datadog, opentelemetry, statsd, memory-profiler, pyinstrument
* Add class instrumentation
* Add instrumentation of properties on classes (but not on instances)

0.2.0 (2020-11-12)
~~~~~~~~~~~~~~~~~~

* Refactor instruments into classes with base class (breaking)
* Rename module: instrumentation -> instruments (breaking)
* Change args decorator/decorator_kwargs -> instrument/instrument_kwargs (breaking)

0.1.1 (2020-11-11)
~~~~~~~~~~~~~~~~~~

* Use module loggers instead of root logger

0.1.0 (2020-11-11)
~~~~~~~~~~~~~~~~~~

* First release
