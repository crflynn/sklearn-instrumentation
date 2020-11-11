import inspect
import logging
import time
from collections.abc import Callable
from functools import wraps
from sys import getsizeof

from sklearn_instrumentation.utils import get_arg_by_key


def column_logger(func: Callable, **dkwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        X = get_arg_by_key(func, args, "X")
        if hasattr(X, "columns"):
            logging.info(f"{func.__qualname__} input columns: {list(X.columns)}")
        retval = func(*args, **kwargs)
        if hasattr(retval, "columns"):
            logging.info(f"{func.__qualname__} output columns: {list(retval.columns)}")
        return retval

    return wrapper


def shape_logger(func: Callable, **dkwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        X = get_arg_by_key(func, args, "X")
        logging.info(f"{func.__qualname__} input X shape: {X.shape}")
        retval = func(*args, **kwargs)
        if hasattr(retval, "shape"):
            logging.info(f"{func.__qualname__} output X shape: {retval.shape}")
        return retval

    return wrapper


def getsizeof_logger(func: Callable, **dkwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        X = get_arg_by_key(func, args, "X")
        logging.info(f"{func.__qualname__} input X nbytes: {getsizeof(X)}")
        retval = func(*args, **kwargs)
        if hasattr(retval, "shape"):
            logging.info(f"{func.__qualname__} output X nbytes: {getsizeof(retval)}")
        return retval

    return wrapper


def time_elapsed_logger(func: Callable, **dkwargs):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"{func.__qualname__} starting.")
        start = time.time()
        retval = func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(f"{func.__qualname__} elapsed time: {elapsed} seconds")
        return retval

    return wrapper
