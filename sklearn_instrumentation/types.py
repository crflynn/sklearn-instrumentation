from typing import Type
from typing import Union

from sklearn.base import BaseEstimator

Estimator = Union[Type[BaseEstimator], BaseEstimator]
