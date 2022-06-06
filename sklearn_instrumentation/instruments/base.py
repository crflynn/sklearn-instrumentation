from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from functools import wraps

from sklearn_instrumentation.types import Estimator


class BaseInstrument(ABC):
    """Base class for instruments."""

    @abstractmethod
    def __call__(
        self, estimator: Estimator, func: Callable, **dkwargs
    ):  # pragma: no cover
        pass


class Identity(BaseInstrument):
    """Identity instrument which decorates with a no-op."""

    def __call__(
        self, estimator: Estimator, func: Callable, **dkwargs
    ):  # pragma: no cover
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
