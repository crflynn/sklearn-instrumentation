import copy
import logging
from collections.abc import MutableMapping
from collections.abc import Sequence
from enum import Enum
from typing import Callable
from typing import Iterable
from typing import List
from typing import Type

import numpy as np
from sklearn.base import BaseEstimator

from sklearn_instrumentation.config import DEFAULT_EXCLUDE
from sklearn_instrumentation.config import DEFAULT_METHODS
from sklearn_instrumentation.utils import get_delegator
from sklearn_instrumentation.utils import get_estimators_in_package
from sklearn_instrumentation.utils import get_method_class
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
        self.instruments = []
        self.instrument_kwargs = []

    def add(self, instrument: Callable, instrument_kwargs: dict):
        """Add a decorator and its kwargs.

        Adds decorator and its kwargs and updates the internal ``wrapper`` attribute.

        :param Callable instrument: The decorator to apply
        :param dict instrument_kwargs: Keyword args for the decorator
        """
        self.instruments.append(instrument)
        self.instrument_kwargs.append(copy.deepcopy(instrument_kwargs))
        self._wrap_function()

    def _wrap_function(self):
        self.wrapper = self.wrapped
        for instrument, kwargs in zip(self.instruments, self.instrument_kwargs):
            self.wrapper = instrument(self.wrapper, **kwargs)

    def remove(self, instrument: Callable):
        """Remove instances of a decorator.

        Removes all instances of decorator and updates the internal ``wrapper``
        attribute.

        :param Callable instrument: The decorator to remove
        """
        idxs = []
        for idx, dec in enumerate(self.instruments):
            if dec == instrument:
                idxs.append(idx)

        for idx in idxs[::-1]:
            self.instruments.pop(idx)
            self.instrument_kwargs.pop(idx)

        self._wrap_function()

    def empty(self) -> bool:
        """Indicate whether there are any decorators.

        :return: True if ``instruments`` attribute is empty.
        """
        return len(self.instruments) == 0


class SklearnInstrumentor:
    """Instrumentor for sklearn estimators.

    A container for instrumentation configuration.

    The instrument (a decorator) should be similar to the following:

    .. code-block:: python

        def instrument_decorator(func: Callable, **dkwargs):

            @wraps(func)
            def wrapper(*args, **kwargs):
                print("Before executing")
                retval = func(*args, **kwargs)
                print("After executing")
                return retval

            return wrapper


    By default, classes derived from ``sklearn.tree._classes.BaseDecisionTree`` are
    excluded from instrumentation.

    By default, methods on which instrumentation is applied includes
    ``_fit``, ``_predict``, ``_predict_proba``, ``_transform``, ``fit``, ``predict``,
    ``predict_proba``, and ``transform``.

    Methods which are properties are **not** instrumented on instances, but **are**
    instrumented on classes.

    :param Callable instrument: A decorator to apply to sklearn estimator methods. The
        wrapping function signature should be ``(func, **dkwargs)``, where ``func`` is
        the target method and ``dkwargs`` is the instrument_kwargs argument of the
        instrumentor.
    :param dict instrument_kwargs: Keyword args to be passed to the decorator.
    :param list(str) methods: A list of method names on which to apply decorators.
    :param list(Type) exclude: A list of types for which instrumentation
        should be skipped.
    """

    def __init__(
        self,
        instrument: Callable,
        instrument_kwargs: dict = None,
        methods: List[str] = None,
        exclude: List[Type] = None,
    ):
        self.instrument = instrument
        self.instrument_kwargs = instrument_kwargs or {}
        self.methods = methods or DEFAULT_METHODS
        self.exclude = tuple(exclude or DEFAULT_EXCLUDE)

    @classmethod
    def _get_instrumentation_attribute_name(cls, method_name: str):
        prefix = cls._get_instrumentation_attribute_prefix()
        return f"{prefix}{method_name}"

    @classmethod
    def _get_instrumentation_attribute_prefix(cls):
        return "_skli_"

    # region instrumentation estimator
    def instrument_estimator(
        self,
        estimator: BaseEstimator,
        recursive: bool = True,
        instrument_kwargs: dict = None,
    ):
        """Instrument a BaseEstimator instance.

        Decorate the methods of the estimator instance.

        :param BaseEstimator estimator: An instance of a BaseEstimator-derived class.
        :param bool recursive: Whether to iterate recursively through metaestimators
            to apply the same instrumentation.
        :param dict instrument_kwargs: Keyword args to be passed to the decorator,
            overriding the ones initialized by the instrumentor
        """
        if recursive:
            self._instrument_recursively(
                obj=estimator, instrument_kwargs=instrument_kwargs
            )
        else:
            self._instrument_estimator(
                estimator=estimator, instrument_kwargs=instrument_kwargs
            )

    def _instrument_estimator(self, estimator: BaseEstimator, instrument_kwargs=None):
        for method_name in self.methods:
            self._instrument_instance_method(
                estimator=estimator,
                method_name=method_name,
                instrument_kwargs=instrument_kwargs,
            )

    def _instrument_recursively(self, obj: object, instrument_kwargs=None):
        if isinstance(obj, tuple(self.exclude)):
            return

        if isinstance(obj, BaseEstimator):
            self._instrument_estimator(
                estimator=obj, instrument_kwargs=instrument_kwargs
            )

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k.startswith(self._get_instrumentation_attribute_prefix()):
                    continue
                self._instrument_recursively(obj=v, instrument_kwargs=instrument_kwargs)
        elif isinstance(obj, MutableMapping):
            for v in obj.values():
                self._instrument_recursively(obj=v, instrument_kwargs=instrument_kwargs)
        elif isinstance(obj, Sequence):
            for o in obj:
                self._instrument_recursively(obj=o, instrument_kwargs=instrument_kwargs)

    def _instrument_instance_method(
        self,
        estimator: BaseEstimator,
        method_name: str,
        instrument_kwargs: dict = None,
    ):
        class_attribute = getattr(estimator.__class__, method_name, None)
        if isinstance(class_attribute, property):
            logger.debug(
                f"Not instrumenting property on instance of: {estimator.__class__.__qualname__}.{method_name}",
            )
            return

        method = getattr(estimator, method_name, None)
        if method is None:
            return

        dkwargs = instrument_kwargs or self.instrument_kwargs

        instr_attrib_name = self._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr = getattr(
            estimator,
            instr_attrib_name,
            None,
        )
        if instr is None:
            instr = SklearnMethodInstrumentation(method)
            setattr(
                estimator,
                instr_attrib_name,
                instr,
            )

        instr.add(instrument=self.instrument, instrument_kwargs=dkwargs)
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
        if isinstance(obj, tuple(self.exclude)):
            return

        if isinstance(obj, BaseEstimator):
            self._uninstrument_estimator(estimator=obj, full=full)

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k.startswith(self._get_instrumentation_attribute_prefix()):
                    continue
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

        instr.remove(instrument=self.instrument)

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

    # region estimator classes
    def instrument_estimators_classes(self, estimators: Iterable[BaseEstimator]):
        """Instrument the classes (not the instances) found in the estimators.

        Lighter version of ``instrument_package``. Instead of crawling the package
        modules and dynamically importing objects, crawls the estimators' hierarchies
        and only instruments classes which are already imported. This is generally
        faster and uses less memory.

        Inspects the (meta)estimator hierarchy, only instrumenting the classes found
        in the estimator, its methods, and its attributes.

        :param Iterable[BaseEstimator] estimators: Several estimator instances of which
            to instrument related classes.
        """
        classes = set()
        for estimator in estimators:
            classes = classes.union(self._get_estimator_classes(estimator))
            self.instrument_classes(estimators=classes)

    def instrument_estimator_classes(self, estimator: BaseEstimator):
        """Instrument the classes (not the instances) found in the estimator.

        Lighter version of ``instrument_package``. Instead of crawling the package
        modules and dynamically importing objects, crawls the estimator hierarchy
        and only instruments classes which are already imported. This is generally
        faster and uses less memory.

        Inspects the (meta)estimator hierarchy, only instrumenting the classes found
        in the estimator, its methods, and its attributes.

        :param BaseEstimator estimator: An estimator instance with which to instrument
            related classes
        """
        classes = self._get_estimator_classes(estimator)
        self.instrument_classes(estimators=classes)

    def uninstrument_estimator_classes(
        self, estimator: BaseEstimator, full: bool = False
    ):
        """Uninstrument the classes (not the instances) found in the estimator.

        Inspects the (meta)estimator hierarchy, only uninstrumenting the classes found
        in the estimator, its methods, and its attributes.

        :param BaseEstimator estimator: An estimator instance with which to uninstrument
            related classes
        :param bool full: Whether to fully uninstrument the estimator classes.
        """
        classes = self._get_estimator_classes(estimator)
        self.uninstrument_classes(estimators=classes, full=full)

    def _get_estimator_classes(self, obj):
        classes = set()
        if isinstance(obj, tuple(self.exclude)):
            return classes

        if isinstance(obj, BaseEstimator):
            class_ = obj.__class__
            classes.add(class_)
            for k in dir(class_):
                v = getattr(class_, k, None)
                if v is None or not callable(v):
                    continue
                v = getattr(obj, k, None)
                if hasattr(v, "__self__") and k in self.methods:
                    classes.add(get_method_class(class_, k))

        if hasattr(obj, "__dict__"):
            for v in obj.__dict__.values():
                classes = classes.union(self._get_estimator_classes(v))
        elif isinstance(obj, MutableMapping):
            for v in obj.values():
                classes = classes.union(self._get_estimator_classes(v))
        elif isinstance(obj, Sequence):
            for o in obj:
                classes = classes.union(self._get_estimator_classes(o))

        return classes

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
        self.instrument_classes(estimators=estimators)

    def instrument_classes(self, estimators: Iterable[Type[BaseEstimator]]):
        """Instrument multiple BaseEstimator classes.

        Apply this instrumentor's decorator to the methods of several BaseEstimator
        classes.

        :param Iterable[Type[BaseEstimator]] estimators: Iterable of classes on which to
            apply instrumentation.
        """
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

        if method_is_inherited(estimator=estimator, method_name=method_name):
            return

        if isinstance(class_method, property):
            self._instrument_property(estimator=estimator, method_name=method_name)
            return

        if is_delegator(func=class_method):
            self._instrument_delegator(delegator=class_method, method_name=method_name)
            return

        instr_attrib_name = self._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr = getattr(
            estimator,
            instr_attrib_name,
            None,
        )
        if instr is None:
            instr = SklearnMethodInstrumentation(class_method)
            setattr(
                estimator,
                instr_attrib_name,
                instr,
            )

        instr.add(instrument=self.instrument, instrument_kwargs=self.instrument_kwargs)
        setattr(
            estimator,
            method_name,
            instr.wrapper,
        )

    def _instrument_property(self, estimator: Type[BaseEstimator], method_name: str):
        instr_attrib_name = self._get_instrumentation_attribute_name(
            method_name=method_name
        )
        property_: property = getattr(estimator, method_name)
        wrapped_method = property_.fget
        instr = getattr(estimator, instr_attrib_name, None)
        if instr is None:
            instr = SklearnMethodInstrumentation(wrapped_method)
            setattr(estimator, instr_attrib_name, instr)

        instr.add(instrument=self.instrument, instrument_kwargs=self.instrument_kwargs)

        setattr(estimator, method_name, property(instr.wrapper))

    def _instrument_delegator(self, delegator: Callable, method_name: str):
        instr_attrib_name = self._get_instrumentation_attribute_name(
            method_name=method_name
        )
        descriptor = get_delegator(delegator)
        instr = getattr(descriptor, instr_attrib_name, None)
        if instr is None:
            instr = SklearnMethodInstrumentation(descriptor.fn)
            setattr(descriptor, instr_attrib_name, instr)

        instr.add(instrument=self.instrument, instrument_kwargs=self.instrument_kwargs)

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
        self.uninstrument_classes(estimators=estimators, full=full)

    def uninstrument_classes(
        self, estimators: Iterable[Type[BaseEstimator]], full: bool = False
    ):
        """Uninstrument BaseEstimator classes.

        Remove this instrumentor's decorator from the methods of a BaseEstimator
        classes.

        :param Iterable[Type[BaseEstimator]] estimators: Classes from which to remove
            instrumentation.
        :param bool full: Whether to fully uninstrument the estimator classes.
        """
        for estimator in estimators:
            self.uninstrument_class(estimator=estimator, full=full)

    def uninstrument_class(self, estimator: Type[BaseEstimator], full: bool = False):
        """Uninstrument a BaseEstimator class.

        Remove this instrumentor's decorator to the methods of a BaseEstimator class.

        :param Type[BaseEstimator] estimator: A class from which to remove
            instrumentation.
        :param bool full: Whether to fully uninstrument the estimator class.
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

        if method_is_inherited(estimator=estimator, method_name=method_name):
            return

        if isinstance(class_method, property):
            self._uninstrument_property(estimator=estimator, method_name=method_name)
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

        instr.remove(instrument=self.instrument)

        if full or instr.empty():
            setattr(estimator, method_name, instr.wrapped)
            delattr(estimator, instr_attrib_name)
        else:
            setattr(estimator, method_name, instr.wrapper)

    def _uninstrument_property(
        self, estimator: Type[BaseEstimator], method_name: str, full: bool = False
    ):
        instr_attrib_name = self._get_instrumentation_attribute_name(method_name)
        instr = getattr(estimator, instr_attrib_name, None)
        if instr is None:
            return

        instr: SklearnMethodInstrumentation
        instr.remove(instrument=self.instrument)

        if full or instr.empty():
            setattr(
                estimator,
                method_name,
                property(instr.wrapped),
            )
            delattr(estimator, instr_attrib_name)
        else:
            setattr(estimator, method_name, property(instr.wrapper))

    def _uninstrument_delegator(
        self, delegator: Callable, method_name: str, full: bool = False
    ):
        instr_attrib_name = self._get_instrumentation_attribute_name(method_name)
        descriptor = get_delegator(delegator)
        instr = getattr(descriptor, instr_attrib_name, None)
        if instr is None:
            return

        instr: SklearnMethodInstrumentation
        instr.remove(instrument=self.instrument)

        if full or instr.empty():
            setattr(
                descriptor,
                "fn",
                instr.wrapped,
            )
            delattr(descriptor, instr_attrib_name)
        else:
            setattr(descriptor, "fn", instr.wrapper)

    # endregion
