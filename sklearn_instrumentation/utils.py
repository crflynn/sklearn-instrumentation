import inspect
import logging
import os
import warnings
from collections.abc import Callable
from functools import wraps
from importlib import import_module
from inspect import isclass
from inspect import ismethod
from pkgutil import walk_packages
from typing import List
from typing import Set
from typing import Type
from typing import Union

from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import _IffHasAttrDescriptor

logger = logging.getLogger(__name__)


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


def get_method_class_name(method: Callable) -> str:
    if isinstance(method, property):
        return method.fget.__qualname__.split(".")[0]
    else:
        return method.__qualname__.split(".")[0]


def get_method_class(estimator: Type[BaseEstimator], method_name: str) -> Type:
    """Get the class owner of the (possibly inherited) method."""
    method = getattr(estimator, method_name)
    method_class_name = get_method_class_name(method=method)
    if estimator.__name__ == method_class_name:
        return estimator
    for class_ in estimator.mro():
        if class_.__name__ == method_class_name:
            return class_
    raise AttributeError("Unable to determine method's class.")


def is_class_method(func: Callable) -> bool:
    """Indicate if the method belongs to a class (opposed to an instance)."""
    if list(inspect.signature(func).parameters.keys())[0] == "self":
        return True


def is_instance_method(func: Callable) -> bool:
    """Indicate if the method belongs to an instance of a class (opposed to the class)."""
    return not is_class_method(func)


def get_delegator(func: Callable) -> _IffHasAttrDescriptor:
    """Get the corresponding ``_IffHasAttrDescriptor``."""
    for cell in func.__closure__:
        obj = cell.cell_contents
        if isinstance(obj, _IffHasAttrDescriptor):
            return obj


def is_delegator(func: Callable) -> bool:
    """Indicate if the method is delegated using ``_IffHasAttrDescriptor``."""
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
    """Indicate if the estimator's method is inherited from a parent class."""
    method = getattr(estimator, method_name)
    method_class_name = get_method_class_name(method=method)

    try:
        estimator_class_name = estimator.__name__
    except AttributeError:
        estimator_class_name = estimator.__class__.__name__

    return method_class_name != estimator_class_name


def has_instrumentation(
    estimator: Union[BaseEstimator, Type[BaseEstimator]], method_name: str
) -> bool:
    """Indicate if the estimator's method is instrumented."""
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
    """Get the value of a corresponding arg index ignoring self for class methods."""
    if is_class_method(func):
        return args[idx + 1]
    else:
        return args[idx]


def get_arg_by_key(func: Callable, args: tuple, key: str):
    """Get the value of a corresponding arg name as found in a function's signature."""
    keys = list(inspect.signature(func).parameters.keys())
    idx = keys.index(key)
    if is_delegator(func):
        return args[idx - 1]
    return args[idx]


def get_estimators_in_packages(
    package_names: List[str],
) -> Set[Type[BaseEstimator]]:
    """Get all BaseEstimators from a list of packages.

    :param list(str) package_names: a list of package names from which to get BaseEstimators
    :return: A dictionary of fully qualified class names as keys and classes as values
    """
    base_estimators = set()
    for package_name in package_names:
        base_estimators = base_estimators.union(
            get_estimators_in_package(package_name=package_name)
        )
    return base_estimators


def get_estimators_in_package(
    package_name: str = "sklearn",
) -> Set[Type[BaseEstimator]]:
    """Get all BaseEstimators from a package.

    :param str package_name: a package name from which to get BaseEstimators
    :return: A dictionary of fully qualified class names as keys and classes as values
    """
    base_estimators = set()
    package = import_module(package_name)
    package_dir = os.path.dirname(package.__file__)
    for (_, module_name, _) in walk_packages(
        [package_dir], prefix=package.__name__ + "."
    ):
        if "test" in module_name:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module = import_module(module_name)
        except ImportError:
            logger.warning(f"Unable to import {package_name}.{module_name}")
            continue
        for module_attribute_name in dir(module):
            module_attribute = getattr(module, module_attribute_name)
            if isclass(module_attribute) and issubclass(
                module_attribute, BaseEstimator
            ):
                base_estimators.add(module_attribute)
    return base_estimators
