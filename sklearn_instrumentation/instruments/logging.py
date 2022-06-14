import logging
import time
from collections.abc import Callable
from functools import wraps
from sys import getsizeof

from sklearn_instrumentation.instruments.base import BaseInstrument
from sklearn_instrumentation.types import Estimator
from sklearn_instrumentation.utils import get_arg_by_key
from sklearn_instrumentation.utils import get_name

logger = logging.getLogger(__name__)


class ColumnNameLogger(BaseInstrument):
    """Instrument which logs the columns of X on input and output.

    Only works if X is a pandas DataFrame.
    """

    def __call__(self, estimator: Estimator, func: Callable, **dkwargs):
        name = get_name(estimator, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            X = get_arg_by_key(func, args, "X")
            if hasattr(X, "columns"):
                logger.info(f"{name} input columns: {list(X.columns)}")
            retval = func(*args, **kwargs)
            if hasattr(retval, "columns"):
                logger.info(f"{name} output columns: {list(retval.columns)}")
            return retval

        return wrapper


class ShapeLogger(BaseInstrument):
    """Instrument which logs the shape of X on input and output."""

    def __call__(self, estimator: Estimator, func: Callable, **dkwargs):
        name = get_name(estimator, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            X = get_arg_by_key(func, args, "X")
            logger.info(f"{name} input X shape: {X.shape}")
            retval = func(*args, **kwargs)
            if hasattr(retval, "shape"):
                logger.info(f"{name} output X shape: {retval.shape}")
            return retval

        return wrapper


class GetSizeOfLogger(BaseInstrument):
    """Instrument which logs ``sys.getsizeof(X)`` on input and output."""

    def __call__(self, estimator: Estimator, func: Callable, **dkwargs):
        name = get_name(estimator, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            X = get_arg_by_key(func, args, "X")
            logger.info(f"{name} input X nbytes: {getsizeof(X)}")
            retval = func(*args, **kwargs)
            if hasattr(retval, "shape"):
                logger.info(f"{name} output X nbytes: {getsizeof(retval)}")
            return retval

        return wrapper


class TimeElapsedLogger(BaseInstrument):
    """Instrument which logs execution time elapsed."""

    def __call__(self, estimator: Estimator, func: Callable, **dkwargs):
        name = get_name(estimator, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"{name} starting.")
            start = time.time()
            retval = func(*args, **kwargs)
            elapsed = time.time() - start
            logger.info(f"{name} elapsed time: {elapsed} seconds")
            return retval

        return wrapper
