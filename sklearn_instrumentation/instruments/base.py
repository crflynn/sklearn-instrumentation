from abc import ABC
from abc import abstractmethod
from collections.abc import Callable
from functools import wraps


class BaseInstrument(ABC):
    """Base class for instruments."""

    @abstractmethod
    def __call__(self, func: Callable, **dkwargs):  # pragma: no cover
        pass


class Identity(BaseInstrument):
    """Identity instrument which decorates with a no-op."""

    def __call__(self, func: Callable, **dkwargs):  # pragma: no cover
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
