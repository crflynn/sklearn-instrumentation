from enum import Enum

import numpy as np
from sklearn.tree import BaseDecisionTree

#: By default, exclude these types from instrumentation.
#: The default setting is to exclude
#: ``bytes``,
#: ``str``,
#: ``ndarray``,
#: ``Enum``
#: and ``BaseDecisionTree``.
DEFAULT_EXCLUDE = [bytes, str, np.ndarray, Enum, BaseDecisionTree]

#: By default, instrument these methods on all BaseEstimators.
#: The default is to instrument
#: ``_fit``,
#: ``_predict``,
#: ``_predict_log_proba``,
#: ``_predict_proba``,
#: ``_transform``,
#: ``fit``,
#: ``predict``,
#: ``predict_log_proba``,
#: ``predict_proba``,
#: and ``transform``.
DEFAULT_METHODS = [
    "_fit",
    "_predict",
    "_predict_log_proba",
    "_predict_proba",
    "_transform",
    "fit",
    "predict",
    "predict_log_proba",
    "predict_proba",
    "transform",
]
