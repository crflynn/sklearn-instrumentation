import logging
from dataclasses import dataclass
from enum import Enum
from functools import wraps

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainTestSet:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


class SomeEnum(Enum):
    A = 1
    B = 2


class TransformerWithEnum(BaseEstimator, TransformerMixin):
    def __init__(self, param: SomeEnum):
        self.param = param

    def fit(self, X, y):
        return self

    def transform(self, X):
        return X


@pytest.fixture(scope="function")
def regression_model(request):
    pipeline = Pipeline(
        steps=[
            (
                "fu",
                FeatureUnion(
                    transformer_list=[
                        ("ss", StandardScaler()),
                        ("pca", PCA()),
                    ]
                ),
            ),
            ("et", TransformerWithEnum(param=SomeEnum.A)),
            ("rf", RandomForestRegressor()),
        ]
    )
    return pipeline


@pytest.fixture(scope="function")
def classification_model(request):
    pipeline = Pipeline(
        steps=[
            (
                "fu",
                FeatureUnion(
                    transformer_list=[
                        ("ss", StandardScaler()),
                        ("pca", PCA()),
                    ]
                ),
            ),
            ("et", TransformerWithEnum(param=SomeEnum.A)),
            ("rf", RandomForestClassifier()),
        ]
    )
    return pipeline


@pytest.fixture(scope="function")
def classification_model_multi(request):
    pipeline = Pipeline(
        steps=[
            (
                "fu",
                FeatureUnion(
                    transformer_list=[
                        ("ss", StandardScaler()),
                        ("ss2", StandardScaler()),
                        ("pca", PCA()),
                        ("pca2", PCA()),
                    ]
                ),
            ),
            ("et", TransformerWithEnum(param=SomeEnum.A)),
            ("rf", RandomForestClassifier()),
        ]
    )
    return pipeline


@pytest.fixture
def boston(request):
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@pytest.fixture
def iris(request):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    iris = TrainTestSet(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
    return iris


@pytest.fixture(scope="function")
def simple_decorator(request):
    def decorator(estimator, func, **dkwargs):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info("hello world")
            return func(*args, **kwargs)

        return wrapper

    return decorator


@pytest.fixture(params=[False, True])
def full(request):
    return request.param
