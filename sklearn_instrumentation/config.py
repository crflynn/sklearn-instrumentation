from sklearn.tree import BaseDecisionTree

# By default, exclude these types from instrumentation
DEFAULT_EXCLUDE = [BaseDecisionTree]

# By default, instrument these methods on all BaseEstimators
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
