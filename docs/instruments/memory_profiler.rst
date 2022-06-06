Memory Profiler Instruments
===========================

To use memory-profiler instrumentation, install with the ``memory-profiler`` extra:

.. code-block:: bash

    pip install sklearn-instrumentation[memory-profiler]


Example usage:

.. code-block:: python

    from sklearn_instrumentation import SklearnInstrumentor
    from sklearn_instrumentation.instruments.memory_profiler import MemoryProfiler

    profiler = MemoryProfiler()
    instrumentor = SklearnInstrumentor(instrument=profiler)

    instrumentor.instrument_instance(classification_model)

    classification_model.fit(X, y)
    classification_model.predict(X)


Example output (partial):

.. code-block:: text

    ForestClassifier.predict_proba
    Filename: /Users/user/projects/sklearn-instrumentation/.venv/lib/python3.8/site-packages/sklearn/ensemble/_forest.py

    Line #    Mem usage    Increment  Occurences   Line Contents
    ============================================================
       648     70.3 MiB     70.3 MiB           1       def predict_proba(self, X):
       649                                                 \"\"\"
       650                                                 Predict class probabilities for X.
       651
       652                                                 The predicted class probabilities of an input sample are computed as
       653                                                 the mean predicted class probabilities of the trees in the forest.
       654                                                 The class probability of a single tree is the fraction of samples of
       655                                                 the same class in a leaf.
       656
       657                                                 Parameters
       658                                                 ----------
       659                                                 X : {array-like, sparse matrix} of shape (n_samples, n_features)
       660                                                     The input samples. Internally, its dtype will be converted to
       661                                                     ``dtype=np.float32``. If a sparse matrix is provided, it will be
       662                                                     converted into a sparse ``csr_matrix``.
       663
       664                                                 Returns
       665                                                 -------
       666                                                 p : ndarray of shape (n_samples, n_classes), or a list of n_outputs
       667                                                     such arrays if n_outputs > 1.
       668                                                     The class probabilities of the input samples. The order of the
       669                                                     classes corresponds to that in the attribute :term:`classes_`.
       670                                                 \"\"\"
       671     70.3 MiB      0.0 MiB           1           check_is_fitted(self)
       672                                                 # Check data
       673     70.3 MiB      0.0 MiB           1           X = self._validate_X_predict(X)
       674
       675                                                 # Assign chunk of trees to jobs
       676     70.3 MiB      0.0 MiB           1           n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)
       677
       678                                                 # avoid storing the output of every estimator by summing them here
       679     70.3 MiB      0.0 MiB           6           all_proba = [np.zeros((X.shape[0], j), dtype=np.float64)
       680     70.3 MiB      0.0 MiB           2                        for j in np.atleast_1d(self.n_classes_)]
       681     70.3 MiB      0.0 MiB           1           lock = threading.Lock()
       682     70.4 MiB      0.0 MiB           3           Parallel(n_jobs=n_jobs, verbose=self.verbose,
       683     70.4 MiB      0.0 MiB         105                    **_joblib_parallel_args(require="sharedmem"))(
       684     70.4 MiB      0.0 MiB         300               delayed(_accumulate_prediction)(e.predict_proba, X, all_proba,
       685     70.4 MiB      0.0 MiB         100                                               lock)
       686     70.4 MiB      0.0 MiB         101               for e in self.estimators_)
       687
       688     70.4 MiB      0.0 MiB           2           for proba in all_proba:
       689     70.4 MiB      0.0 MiB           1               proba /= len(self.estimators_)
       690
       691     70.4 MiB      0.0 MiB           1           if len(all_proba) == 1:
       692     70.4 MiB      0.0 MiB           1               return all_proba[0]
       693                                                 else:
       694                                                     return all_proba


.. autoclass:: sklearn_instrumentation.instruments.memory_profiler.MemoryProfiler
