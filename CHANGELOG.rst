Release Changelog
-----------------

0.12.0 (2022-06-06)
~~~~~~~~~~~~~~~~~~~

* Breaking: Changed `instrument_estimator` to `instrument_instance`
* Breaking: Instrument API now includes the class/instance on which instrumentation is applied
* Bugfix: Class instrumentation now properly handles inherited methods, properties,and delegates.
* Bugfix: `_get_estimator_classes` method now excludes inherited classes.

0.11.0 (2022-05-22)
~~~~~~~~~~~~~~~~~~~

* Fix a bug with full uninstrumentation

0.10.0 (2022-03-23)
~~~~~~~~~~~~~~~~~~

* Reverting from 0.8.0: Change the order of instrumentation so that an estimator is instrumented after its attributes

0.9.0 (2022-03-22)
~~~~~~~~~~~~~~~~~~

* Skip instrumentation recursion of estimator attributes which share a name with a method intended to be instrumented

0.8.0 (2022-03-22)
~~~~~~~~~~~~~~~~~~

* Change the order of instrumentation so that an estimator is instrumented after its attributes

0.7.0 (2021-05-10)
~~~~~~~~~~~~~~~~~~

* Removed numpy as dependency since it is already a dependency of sklearn
* Modified instrumentation to prevent double instrumentation of methods
* Updated dependencies for some instruments

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
