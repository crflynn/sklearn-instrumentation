import copy
import logging
from collections.abc import MutableMapping
from collections.abc import Sequence
from typing import Callable
from typing import List
from typing import Type

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion

from sklearn_instrumentation.config import DEFAULT_EXCLUDE
from sklearn_instrumentation.config import DEFAULT_METHODS
from sklearn_instrumentation.utils import get_delegator
from sklearn_instrumentation.utils import get_estimators_in_package
from sklearn_instrumentation.utils import is_delegator
from sklearn_instrumentation.utils import method_is_inherited

logger = logging.getLogger(__name__)


class SklearnMethodInstrumentation:
    """Container for multiple decorators of the same function.

    :param Callable func: The function to be decorated.
    """

    def __init__(self, func: Callable):
        self.wrapped = func
        self.wrapper = func
        self.decorators = []
        self.decorator_kwargs = []

    def add(self, decorator: Callable, decorator_kwargs: dict):
        """Add a decorator and its kwargs.

        Adds decorator and its kwargs and updates the internal ``wrapper`` attribute.

        :param Callable decorator: The decorator to apply
        :param dict decorator_kwargs: Keyword args for the decorator
        """
        self.decorators.append(decorator)
        self.decorator_kwargs.append(copy.deepcopy(decorator_kwargs))
        self._wrap_function()

    def _wrap_function(self):
        self.wrapper = self.wrapped
        for decorator, kwargs in zip(self.decorators, self.decorator_kwargs):
            self.wrapper = decorator(self.wrapper, **kwargs)

    def remove(self, decorator: Callable):
        """Remove instances of a decorator.

        Removes all instances of decorator and updates the internal ``wrapper``
        attribute.

        :param Callable decorator: The decorator to remove
        """
        idxs = []
        for idx, dec in enumerate(self.decorators):
            if dec == decorator:
                idxs.append(idx)

        for idx in idxs[::-1]:
            self.decorators.pop(idx)
            self.decorator_kwargs.pop(idx)

        self._wrap_function()

    def empty(self) -> bool:
        """Indicate whether there are any decorators.

        :return: True if ``decorators`` attribute is empty.
        """
        return len(self.decorators) == 0


class SklearnInstrumentor:
    """Instrumentor for sklearn estimators.

    A container for instrumentation configuration.

    The decorator should be similar to the following:

    .. code-block:: python

        def instrumenting_decorator(func: Callable, **dkwargs):

            @wraps(func)
            def wrapper(*args, **kwargs):
                print("Before executing")
                retval = func(*args, **kwargs)
                print("After executing")
                return retval

            return wrapper


    :param Callable decorator: A decorator to apply to sklearn estimator methods. The
        wrapping function signature should be ``(func, **dkwargs)``, where ``func`` is
        the target method and ``dkwargs`` is the decorator_kwargs argument of the
        instrumentor.
    :param dict decorator_kwargs: Keyword args to be passed to the decorator.
    :param list(str) methods: A list of method names on which to apply decorators.
    :param list(BaseEstimator) exclude: A list of classes for which instrumentation
        should be skipped.
    """

    def __init__(
        self,
        decorator: Callable,
        decorator_kwargs: dict = None,
        methods: List[str] = None,
        exclude: List[BaseEstimator] = None,
    ):
        self.decorator = decorator
        self.decorator_kwargs = decorator_kwargs or {}
        self.methods = methods or DEFAULT_METHODS
        self.exclude = tuple(exclude or DEFAULT_EXCLUDE)

    # region instrumentation estimator
    def instrument_estimator(
        self,
        estimator: BaseEstimator,
        recursive: bool = True,
        decorator_kwargs: dict = None,
    ):
        """Instrument a BaseEstimator instance.

        Decorate the methods of the estimator instance.

        :param BaseEstimator estimator: An instance of a BaseEstimator-derived class.
        :param bool recursive: Whether to iterate recursively through metaestimators
            to apply the same instrumentation.
        :param dict decorator_kwargs: Keyword args to be passed to the decorator,
            overriding the ones initialized by the instrumentor
        """
        if recursive:
            self._instrument_recursively(
                obj=estimator, decorator_kwargs=decorator_kwargs
            )
        else:
            self._instrument_estimator(
                estimator=estimator, decorator_kwargs=decorator_kwargs
            )

    def _instrument_estimator(self, estimator: BaseEstimator, decorator_kwargs=None):
        for method_name in self.methods:
            self._instrument_instance_method(
                estimator=estimator,
                method_name=method_name,
                decorator_kwargs=decorator_kwargs,
            )

    def _instrument_recursively(self, obj: object, decorator_kwargs=None):
        if isinstance(obj, FeatureUnion):
            pass
        if isinstance(obj, (*self.exclude, str, np.ndarray)):
            return

        if isinstance(obj, BaseEstimator):
            self._instrument_estimator(estimator=obj, decorator_kwargs=decorator_kwargs)

        if hasattr(obj, "__dict__"):
            for v in obj.__dict__.values():
                self._instrument_recursively(obj=v, decorator_kwargs=decorator_kwargs)
        elif isinstance(obj, MutableMapping):
            for v in obj.values():
                self._instrument_recursively(obj=v, decorator_kwargs=decorator_kwargs)
        elif isinstance(obj, Sequence):
            for o in obj:
                self._instrument_recursively(obj=o, decorator_kwargs=decorator_kwargs)

    def _instrument_instance_method(
        self,
        estimator: BaseEstimator,
        method_name: str,
        decorator_kwargs: dict = None,
    ):
        class_attribute = getattr(type(estimator), method_name, None)
        if isinstance(class_attribute, property):
            logger.debug(
                f"Not instrumenting property: {estimator.__class__.__qualname__}.{method_name}",
            )
            return

        method = getattr(estimator, method_name, None)
        if method is None:
            return

        dkwargs = decorator_kwargs or self.decorator_kwargs

        instr = getattr(estimator, f"_skli_{method_name}", None)
        if instr is None:
            instr = SklearnMethodInstrumentation(method)
            setattr(estimator, f"_skli_{method_name}", instr)

        instr.add(decorator=self.decorator, decorator_kwargs=dkwargs)
        setattr(
            estimator,
            method_name,
            instr.wrapper,
        )

    # endregion

    # region uninstrumentation estimator
    def uninstrument_estimator(
        self, estimator: BaseEstimator, recursive: bool = True, full: bool = False
    ):
        """Uninstrument a BaseEstimator instance.

        Remove this instrumentor's decorators on the methods of the estimator instance.

        :param BaseEstimator estimator: An instance of a BaseEstimator-derived class.
        :param bool recursive: Whether to iterate recursively through metaestimators
            to apply the same uninstrumentation.
        :param bool full: Whether to fully uninstrument the estimator.
        """
        if recursive:
            self._uninstrument_recursively(obj=estimator, full=full)
        else:
            self._uninstrument_estimator(estimator=estimator, full=full)

    def _uninstrument_recursively(self, obj: object, full: bool = False):
        if isinstance(obj, (*self.exclude, str, np.ndarray)):
            return

        if isinstance(obj, BaseEstimator):
            self._uninstrument_estimator(estimator=obj, full=full)

        if hasattr(obj, "__dict__"):
            for v in obj.__dict__.values():
                self._uninstrument_recursively(obj=v, full=full)
        elif isinstance(obj, MutableMapping):
            for v in obj.values():
                self._uninstrument_recursively(obj=v, full=full)
        elif isinstance(obj, Sequence):
            for o in obj:
                self._uninstrument_recursively(obj=o, full=full)

    def _uninstrument_estimator(self, estimator: BaseEstimator, full: bool = False):
        if full:
            methods = [f.__name__ for f in estimator.__dict__ if callable(f)]
        else:
            methods = self.methods

        for method_name in methods:
            self._uninstrument_method(
                estimator=estimator, method_name=method_name, full=full
            )

    def _uninstrument_method(
        self, estimator: BaseEstimator, method_name: str, full: bool = False
    ):
        class_attribute = getattr(type(estimator), method_name, None)
        if isinstance(class_attribute, property):
            return

        method = getattr(estimator, method_name, None)
        if method is None:
            return

        instr_attrib_name = self._get_instrumentation_attribute_name(method_name)
        instr: SklearnMethodInstrumentation = getattr(
            estimator, instr_attrib_name, None
        )
        if instr is None:
            return

        instr.remove(decorator=self.decorator)

        if full or instr.empty():
            setattr(
                estimator,
                method_name,
                instr.wrapped,
            )
            delattr(estimator, instr_attrib_name)
        else:
            setattr(estimator, method_name, instr.wrapper)

    # endregion

    # region instrumentation package
    def instrument_packages(self, package_names: List[str]):
        """Instrument multiple packages.

        Apply this instrumentor's decorator to the components of the packages.

        :param list(str) package_names: A list of package names.
        """
        for package_name in package_names:
            self.instrument_package(package_name=package_name)

    def instrument_package(self, package_name: str = "sklearn"):
        """Instrument a package.

        Apply this instrumentor's decorator to the components of the package.

        :param str package_name: A list of package names.
        """
        estimators = get_estimators_in_package(package_name=package_name)
        for estimator in estimators:
            self.instrument_class(estimator=estimator)

    def instrument_class(self, estimator: Type[BaseEstimator]):
        """Instrument a BaseEstimator class.

        Apply this instrumentor's decorator to the methods of a BaseEstimator class.

        :param Type[BaseEstimator] estimator: A class on which to apply instrumentation.
        """
        if issubclass(estimator, self.exclude):
            logger.debug(f"Not instrumenting (excluded): {str(estimator)}")
            return
        logger.debug(f"Instrumenting: {str(estimator)}")
        for method_name in self.methods:
            self._instrument_class_method(estimator=estimator, method_name=method_name)

    def _instrument_class_method(
        self,
        estimator: Type[BaseEstimator],
        method_name: str,
    ):
        class_method = getattr(estimator, method_name, None)
        if class_method is None:
            return

        if isinstance(class_method, property):
            logger.debug(
                f"Not instrumenting property: {estimator.__qualname__}.{method_name}",
            )
            return

        if method_is_inherited(estimator=estimator, method_name=method_name):
            return

        if is_delegator(func=class_method):
            self._instrument_delegator(delegator=class_method, method_name=method_name)
            return

        instr = getattr(estimator, f"_skli_{method_name}", None)
        if instr is None:
            instr = SklearnMethodInstrumentation(class_method)
            setattr(estimator, f"_skli_{method_name}", instr)
        instr.add(decorator=self.decorator, decorator_kwargs=self.decorator_kwargs)
        setattr(
            estimator,
            method_name,
            instr.wrapper,
        )

    def _instrument_delegator(self, delegator: Callable, method_name: str):
        instr_attrib_name = self._get_instrumentation_attribute_name(method_name)
        descriptor = get_delegator(delegator)
        instr = getattr(descriptor, instr_attrib_name, None)
        if instr is None:
            instr = SklearnMethodInstrumentation(descriptor.fn)
            setattr(descriptor, instr_attrib_name, instr)

        instr.add(decorator=self.decorator, decorator_kwargs=self.decorator_kwargs)

        setattr(
            descriptor,
            "fn",
            instr.wrapper,
        )

    # endregion

    # region uninstrumentation package
    def uninstrument_packages(self, package_names: List[str], full: bool = False):
        """Uninstrument multiple packages.

        Remove this instrumentor's decorators from the components of the packages.

        :param list(str) package_names: A list of package names.
        :param bool full: Whether to fully uninstrument the packages.
        """
        for package_name in package_names:
            self.uninstrument_package(package_name=package_name, full=full)

    def uninstrument_package(self, package_name: str = "sklearn", full: bool = False):
        """Uninstrument a package.

        Remove this instrumentor's decorators from the components of the package.

        :param str package_name: A package name.
        :param bool full: Whether to fully uninstrument the packages.
        """
        estimators = get_estimators_in_package(package_name=package_name)
        for estimator in estimators:
            self.uninstrument_class(estimator=estimator, full=full)

    def uninstrument_class(self, estimator: Type[BaseEstimator], full: bool = False):
        """Instrument a BaseEstimator class.

        Apply this instrumentor's decorator to the methods of a BaseEstimator class.

        :param Type[BaseEstimator] estimator: A class on which to apply instrumentation.
        """
        for method_name in self.methods:
            self._uninstrument_class_method(
                estimator=estimator, method_name=method_name, full=full
            )

    def _uninstrument_class_method(
        self,
        estimator: Type[BaseEstimator],
        method_name: str,
        full: bool = False,
    ):
        class_method = getattr(estimator, method_name, None)
        if class_method is None:
            return

        if isinstance(class_method, property):
            return

        if method_is_inherited(estimator=estimator, method_name=method_name):
            return

        if is_delegator(func=class_method):
            self._uninstrument_delegator(
                delegator=class_method,
                method_name=method_name,
                full=full,
            )
            return

        instr_attrib_name = self._get_instrumentation_attribute_name(method_name)
        instr = getattr(estimator, instr_attrib_name, None)
        if instr is None:
            return

        instr.remove(decorator=self.decorator)

        if full or instr.empty():
            setattr(estimator, method_name, instr.wrapped)
            delattr(estimator, instr_attrib_name)
        else:
            setattr(estimator, method_name, instr.wrapper)

    def _uninstrument_delegator(
        self, delegator: Callable, method_name: str, full: bool = False
    ):
        instr_attrib_name = self._get_instrumentation_attribute_name(method_name)
        descriptor = get_delegator(delegator)
        instr = getattr(descriptor, instr_attrib_name, None)
        if instr is None:
            return

        instr: SklearnMethodInstrumentation
        instr.remove(decorator=self.decorator)

        if full or instr.empty():
            setattr(
                descriptor,
                "fn",
                instr.wrapped,
            )
            delattr(descriptor, instr_attrib_name)
        else:
            setattr(descriptor, "fn", instr.wrapper)

    def _get_instrumentation_attribute_name(self, method_name: str):
        return f"_skli_{method_name}"

    # endregion
