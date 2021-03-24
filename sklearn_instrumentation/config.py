from enum import Enum

from sklearn.tree import BaseDecisionTree

#: By default, exclude these types from instrumentation.
#: The default setting is to exclude ``BaseDecisionTree`` and ``Enum``
DEFAULT_EXCLUDE = [BaseDecisionTree, Enum]

#: By default, instrument these methods on all BaseEstimators.
#: The default is to instrument ``_fit``, ``_predict``,
#: ``_predict_proba``, ``_transform``, ``fit``, ``predict``,
#: ``predict_proba``, ``transform``.
DEFAULT_METHODS = [
    "_fit",
    "_predict",
    "_predict_proba",
    "_transform",
    "fit",
    "predict",
    "predict_proba",
    "transform",
]
