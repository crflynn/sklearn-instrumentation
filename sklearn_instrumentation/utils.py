import inspect
import logging
import os
from collections.abc import Callable
from functools import wraps
from importlib import import_module
from inspect import isclass
from inspect import ismethod
from pkgutil import iter_modules
from typing import Dict
from typing import List
from typing import Type
from typing import Union

from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import _IffHasAttrDescriptor


def compose_decorators(decorators: List[Callable]) -> Callable:
    """Compose multiple decorators into one.

    Helper function for combining multiple instrumentation decorators into one.

    :param list(Callable) decorators: A list of instrumentation decorators to be
        combined into a single decorator.
    """

    def composed(func: Callable, **dkwargs) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            wrapped_func = func
            for decorator in decorators:
                wrapped_func = decorator(wrapped_func, **dkwargs)
            return wrapped_func(*args, **kwargs)

        return wrapper

    return composed


def get_sklearn_estimator_from_method(func: Callable) -> BaseEstimator:
    """Get the estimator of a method or delegate.

    Raises TypeError is the instance is not a BaseEstimator.

    :param Callable func: A bound method or delegator function of a BaseEstimator
        instance
    :return: The BaseEstimator instance of the method or delegator.
    """
    err = "Passed function is not a method or delegate of a BaseEstimator"
    if ismethod(func):
        obj = func.__self__
        if isinstance(obj, BaseEstimator):
            return func.__self__
    else:
        try:
            for cell in func.__closure__:
                obj = cell.cell_contents
                if isinstance(obj, BaseEstimator):
                    return obj
        except TypeError as exc:
            raise TypeError(err)
    raise TypeError(err)


def is_class_method(func: Callable) -> bool:
    if list(inspect.signature(func).parameters.keys())[0] == "self":
        return True


def is_instance_method(func: Callable) -> bool:
    return not is_class_method(func)


def get_delegator(func: Callable) -> _IffHasAttrDescriptor:
    for cell in func.__closure__:
        obj = cell.cell_contents
        if isinstance(obj, _IffHasAttrDescriptor):
            return obj


def is_delegator(func: Callable) -> bool:
    try:
        for cell in func.__closure__:
            obj = cell.cell_contents
            if isinstance(obj, _IffHasAttrDescriptor):
                return True
    except TypeError as exc:
        pass
    return False


def method_is_inherited(
    estimator: Union[BaseEstimator, Type[BaseEstimator]], method_name: str
) -> bool:
    method = getattr(estimator, method_name)
    method_class = method.__qualname__.split(".")[0]
    try:
        estimator_class_name = estimator.__name__
    except AttributeError:
        estimator_class_name = estimator.__class__.__name__
    return method_class != estimator_class_name


def has_instrumentation(
    estimator: Union[BaseEstimator, Type[BaseEstimator]], method_name: str
) -> bool:
    method = getattr(estimator, method_name)
    instr_attrib_name = f"_skli_{method_name}"
    if hasattr(estimator, instr_attrib_name):
        return True
    if is_delegator(method):
        descriptor = get_delegator(method)
        if hasattr(descriptor, instr_attrib_name):
            return True
    return False


def non_self_arg(func: Callable, args: tuple, idx: int):
    if is_class_method(func):
        return args[idx + 1]
    else:
        return args[idx]


def get_arg_by_key(func: Callable, args: tuple, key: str):
    keys = list(inspect.signature(func).parameters.keys())
    idx = keys.index(key)
    if is_delegator(func):
        return args[idx - 1]
    return args[idx]


def get_estimators_in_packages(
    package_names: List[str],
) -> Dict[str, Type[BaseEstimator]]:
    """Get all BaseEstimators from a list of packages.

    :param list(str) package_names: a list of package names from which to get BaseEstimators
    :return: A dictionary of fully qualified class names as keys and classes as values
    """
    base_estimators = dict()
    for package_name in package_names:
        base_estimators.update(get_estimators_in_package(package_name=package_name))
    return base_estimators


def get_estimators_in_package(
    package_name: str = "sklearn",
) -> Dict[str, Type[BaseEstimator]]:
    """Get all BaseEstimators from a package.

    :param str package_name: a package name from which to get BaseEstimators
    :return: A dictionary of fully qualified class names as keys and classes as values
    """
    base_estimators = dict()
    package = import_module(package_name)
    package_dir = os.path.dirname(package.__file__)
    for (_, module_name, _) in iter_modules([package_dir]):
        try:
            module = import_module(package_name + "." + module_name)
        except ImportError:
            logging.warning(f"Unable to import {package_name}.{module_name}")
            continue
        for module_attribute_name in dir(module):
            module_attribute = getattr(module, module_attribute_name)
            if isclass(module_attribute) and issubclass(
                module_attribute, BaseEstimator
            ):
                full_qualname = ".".join(
                    [package_name, module_name, module_attribute_name]
                )
                base_estimators[full_qualname] = module_attribute
    return base_estimators
