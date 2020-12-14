Release Changelog
-----------------

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
