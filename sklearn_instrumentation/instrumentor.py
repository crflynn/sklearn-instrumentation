import logging
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Callable
from typing import Iterable
from typing import List
from typing import Set
from typing import Type

from sklearn.base import BaseEstimator
from sklearn.utils.metaestimators import if_delegate_has_method

from sklearn_instrumentation.config import DEFAULT_EXCLUDE
from sklearn_instrumentation.config import DEFAULT_METHODS
from sklearn_instrumentation.types import Estimator
from sklearn_instrumentation.utils import get_descriptor
from sklearn_instrumentation.utils import get_estimators_in_package
from sklearn_instrumentation.utils import is_delegator
from sklearn_instrumentation.utils import method_is_inherited

logger = logging.getLogger(__name__)


@dataclass
class ImplementedInstrument:
    callable: Callable
    kwargs: dict


class SklearnMethodInstrumentation:
    """Container for multiple decorators of the same function.

    :param Callable func: The function to be decorated.
    """

    def __init__(
        self,
        estimator: Estimator,
        func: Callable,
        inherited: bool = False,
    ):
        self.estimator = estimator
        self.func = func
        self.instrumented_func = func
        self.inherited = inherited  # for class instrumentation
        self.instruments: List[ImplementedInstrument] = []

    @property
    def callables(self):
        return [i.callable for i in self.instruments]

    @property
    def kwargs(self):
        return [i.kwargs for i in self.instruments]

    def add(self, instrument: Callable, instrument_kwargs: dict):
        """Add a decorator and its kwargs.

        Adds decorator and its kwargs and updates the internal ``wrapper`` attribute.

        :param Callable instrument: The decorator to apply
        :param dict instrument_kwargs: Keyword args for the decorator
        """
        instrument = ImplementedInstrument(
            callable=instrument,
            kwargs=instrument_kwargs,
        )
        self.instruments.append(instrument)
        self._wrap_function()

    def contains(self, instrument: Callable, instrument_kwargs: dict) -> bool:
        """Detect if instrumentation contains instrument.

        Returns ``True`` if instrument with kwargs is already present.

        :param Callable instrument: The decorator to apply
        :param dict instrument_kwargs: Keyword args for the decorator
        """
        instrument = ImplementedInstrument(
            callable=instrument,
            kwargs=instrument_kwargs,
        )
        if instrument in self.instruments:
            return True
        return False

    def _wrap_function(self):
        self.instrumented_func = self.func
        for instrument in self.instruments:
            self.instrumented_func = instrument.callable(
                self.estimator, self.instrumented_func, **instrument.kwargs
            )

    def remove(self, instrument: Callable):
        """Remove instances of a decorator.

        Removes all instances of decorator and updates the internal ``wrapper``
        attribute.

        :param Callable instrument: The decorator to remove
        """
        idxs = []
        for idx, instrumented in enumerate(self.instruments):
            if instrumented.callable == instrument:
                idxs.append(idx)

        for idx in idxs[::-1]:
            self.instruments.pop(idx)

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

    # region instrumentation instance
    def instrument_instance(
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
            self._instrument_instance_recursively(
                obj=estimator, instrument_kwargs=instrument_kwargs
            )
        else:
            self._instrument_instance(
                estimator=estimator, instrument_kwargs=instrument_kwargs
            )

    def _instrument_instance(self, estimator: BaseEstimator, instrument_kwargs=None):
        for method_name in self.methods:
            self._instrument_instance_method(
                estimator=estimator,
                method_name=method_name,
                instrument_kwargs=instrument_kwargs,
            )

    def _instrument_instance_recursively(self, obj: object, instrument_kwargs=None):
        if isinstance(obj, tuple(self.exclude)):
            return

        if isinstance(obj, BaseEstimator):
            self._instrument_instance(
                estimator=obj, instrument_kwargs=instrument_kwargs
            )

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k.startswith(self._get_instrumentation_attribute_prefix()):
                    continue
                if isinstance(obj, BaseEstimator) and k in self.methods:
                    continue
                self._instrument_instance_recursively(
                    obj=v, instrument_kwargs=instrument_kwargs
                )
        elif isinstance(obj, MutableMapping):
            for v in obj.values():
                self._instrument_instance_recursively(
                    obj=v, instrument_kwargs=instrument_kwargs
                )
        elif isinstance(obj, Iterable):
            for o in obj:
                self._instrument_instance_recursively(
                    obj=o, instrument_kwargs=instrument_kwargs
                )

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
            instr = SklearnMethodInstrumentation(estimator, method)
            setattr(
                estimator,
                instr_attrib_name,
                instr,
            )

        if not instr.contains(self.instrument, instrument_kwargs=dkwargs):
            instr.add(instrument=self.instrument, instrument_kwargs=dkwargs)

        setattr(
            estimator,
            method_name,
            instr.instrumented_func,
        )

    # endregion

    # region uninstrumentation estimator
    def uninstrument_instance(
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
            self._uninstrument_instance(estimator=estimator, full=full)

    def _uninstrument_recursively(self, obj: object, full: bool = False):
        if isinstance(obj, tuple(self.exclude)):
            return

        if isinstance(obj, BaseEstimator):
            self._uninstrument_instance(estimator=obj, full=full)

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k.startswith(self._get_instrumentation_attribute_prefix()):
                    continue
                if isinstance(obj, BaseEstimator) and k in self.methods:
                    continue
                self._uninstrument_recursively(obj=v, full=full)
        elif isinstance(obj, MutableMapping):
            for v in obj.values():
                self._uninstrument_recursively(obj=v, full=full)
        elif isinstance(obj, Iterable):
            for o in obj:
                self._uninstrument_recursively(obj=o, full=full)

    def _uninstrument_instance(self, estimator: BaseEstimator, full: bool = False):
        if full:
            methods = []
            prefix = self._get_instrumentation_attribute_prefix()
            for name, attr in estimator.__dict__.items():
                if name.startswith(prefix) and isinstance(
                    attr, SklearnMethodInstrumentation
                ):
                    method = name.replace(prefix, "")
                    methods.append(method)
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
                instr.func,
            )
            delattr(estimator, instr_attrib_name)
        else:
            setattr(estimator, method_name, instr.instrumented_func)

    # endregion

    # region estimator classes
    def instrument_instances_classes(self, estimators: Iterable[BaseEstimator]):
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
            classes = classes.union(self._get_instance_classes(estimator))
            self.instrument_classes(estimators=classes)

    def instrument_instance_classes(self, estimator: BaseEstimator):
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
        classes = self._get_instance_classes(estimator)
        self.instrument_classes(estimators=classes)

    def uninstrument_instance_classes(
        self, estimator: BaseEstimator, full: bool = False
    ):
        """Uninstrument the classes (not the instances) found in the estimator.

        Inspects the (meta)estimator hierarchy, only uninstrumenting the classes found
        in the estimator, its methods, and its attributes.

        :param BaseEstimator estimator: An estimator instance with which to uninstrument
            related classes
        :param bool full: Whether to fully uninstrument the estimator classes.
        """
        classes = self._get_instance_classes(estimator)
        self.uninstrument_classes(estimators=classes, full=full)

    def _get_instance_classes(self, obj, seen: Set = None):
        seen = seen or set()
        classes = set()
        if isinstance(obj, tuple(self.exclude)):
            return classes

        if isinstance(obj, BaseEstimator):
            classes.add(obj.__class__)

        if id(obj) in seen:
            return classes
        seen.add(id(obj))

        if hasattr(obj, "__dict__"):
            for v in obj.__dict__.values():
                classes = classes.union(self._get_instance_classes(v, seen))
        elif isinstance(obj, MutableMapping):
            for v in obj.values():
                classes = classes.union(self._get_instance_classes(v, seen))
        elif isinstance(obj, Iterable):
            for o in obj:
                classes = classes.union(self._get_instance_classes(o, seen))

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

        if isinstance(class_method, property):
            self._instrument_property(estimator=estimator, method_name=method_name)
            return

        if is_delegator(func=class_method):
            self._instrument_delegator(estimator=estimator, method_name=method_name)
            return

        instr_attrib_name = self._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr = getattr(
            estimator,
            instr_attrib_name,
            None,
        )
        # if the parent is instrumented already, but the current is not
        if instr and instr.estimator != estimator:
            class_method = instr.func
            instr = None

        if instr is None:
            inherited = method_is_inherited(
                estimator=estimator,
                method=class_method,
            )
            instr = SklearnMethodInstrumentation(
                estimator=estimator,
                func=class_method,
                inherited=inherited,
            )
            setattr(
                estimator,
                instr_attrib_name,
                instr,
            )

        if not instr.contains(
            self.instrument, instrument_kwargs=self.instrument_kwargs
        ):
            instr.add(
                instrument=self.instrument, instrument_kwargs=self.instrument_kwargs
            )
        setattr(
            estimator,
            method_name,
            instr.instrumented_func,
        )

    def _instrument_property(self, estimator: Type[BaseEstimator], method_name: str):
        instr_attrib_name = self._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr = getattr(estimator, instr_attrib_name, None)
        # if parent is instrumented but current is not
        if instr and instr.estimator != estimator:
            wrapped_method = instr.instrumented_func
            instr = None
        else:
            property_: property = getattr(estimator, method_name)
            wrapped_method = property_.fget

        if instr is None:
            inherited = method_is_inherited(
                estimator=estimator,
                method=wrapped_method,
            )
            instr = SklearnMethodInstrumentation(
                estimator=estimator, func=wrapped_method, inherited=inherited
            )
            setattr(estimator, instr_attrib_name, instr)

        if not instr.contains(
            self.instrument, instrument_kwargs=self.instrument_kwargs
        ):
            instr.add(
                instrument=self.instrument, instrument_kwargs=self.instrument_kwargs
            )
        setattr(estimator, method_name, property(instr.instrumented_func))

    def _instrument_delegator(self, estimator: Type[BaseEstimator], method_name: str):
        instr_attrib_name = self._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr = getattr(estimator, instr_attrib_name, None)
        # if parent is instrumented but current is not
        if instr and instr.estimator != estimator:
            instr = None

        delegator = getattr(estimator, method_name)
        descriptor = get_descriptor(delegator)
        if instr is None:
            inherited = method_is_inherited(estimator=estimator, method=descriptor.fn)
            instr = SklearnMethodInstrumentation(
                estimator=estimator, func=descriptor.fn, inherited=inherited
            )
            setattr(estimator, instr_attrib_name, instr)

        if not instr.contains(
            self.instrument, instrument_kwargs=self.instrument_kwargs
        ):
            instr.add(
                instrument=self.instrument, instrument_kwargs=self.instrument_kwargs
            )
        setattr(
            estimator,
            method_name,
            if_delegate_has_method(descriptor.delegate_names)(instr.instrumented_func),
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

        if isinstance(class_method, property):
            self._uninstrument_property(estimator=estimator, method_name=method_name)
            return

        if is_delegator(func=class_method):
            self._uninstrument_delegator(
                estimator=estimator,
                method_name=method_name,
                full=full,
            )
            return

        instr_attrib_name = self._get_instrumentation_attribute_name(method_name)
        instr = getattr(estimator, instr_attrib_name, None)
        if instr is None:
            return

        instr: SklearnMethodInstrumentation
        instr.remove(instrument=self.instrument)

        if full or instr.empty():
            if instr.inherited:
                delattr(estimator, method_name)
            else:
                setattr(estimator, method_name, instr.func)
            delattr(estimator, instr_attrib_name)
        else:
            setattr(estimator, method_name, instr.instrumented_func)

    def _uninstrument_property(
        self, estimator: Type[BaseEstimator], method_name: str, full: bool = False
    ):
        instr_attrib_name = self._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr = getattr(estimator, instr_attrib_name, None)
        # if parent is instrumented but current is not
        if instr is None or instr.estimator != estimator:
            return

        instr: SklearnMethodInstrumentation
        instr.remove(instrument=self.instrument)

        if full or instr.empty():
            if instr.inherited:
                delattr(estimator, method_name)
            else:
                setattr(
                    estimator,
                    method_name,
                    property(instr.func),
                )
            delattr(estimator, instr_attrib_name)
        else:
            setattr(estimator, method_name, property(instr.instrumented_func))

    def _uninstrument_delegator(
        self, estimator: Type[BaseEstimator], method_name: str, full: bool = False
    ):
        instr_attrib_name = self._get_instrumentation_attribute_name(method_name)
        instr = getattr(estimator, instr_attrib_name, None)
        # if parent is instrumented but current is not
        if instr is None or instr.estimator != estimator:
            return

        instr: SklearnMethodInstrumentation
        instr.remove(instrument=self.instrument)

        delegator = getattr(estimator, method_name)
        descriptor = get_descriptor(delegator)

        if full or instr.empty():
            if instr.inherited:
                delattr(estimator, method_name)
            else:
                setattr(
                    estimator,
                    method_name,
                    if_delegate_has_method(descriptor.delegate_names)(instr.func),
                )
            delattr(estimator, instr_attrib_name)
        else:
            setattr(
                estimator,
                method_name,
                if_delegate_has_method(descriptor.delegate_names)(
                    instr.instrumented_func
                ),
            )

    # endregion
