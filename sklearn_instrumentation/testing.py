from enum import Enum
from typing import Callable
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import Sequence
from typing import Type

import numpy as np
from sklearn.base import BaseEstimator

from sklearn_instrumentation.instrumentor import SklearnInstrumentor
from sklearn_instrumentation.instrumentor import SklearnMethodInstrumentation
from sklearn_instrumentation.utils import get_delegator
from sklearn_instrumentation.utils import get_estimators_in_package
from sklearn_instrumentation.utils import is_delegator
from sklearn_instrumentation.utils import method_is_inherited


class SklearnInstrumentionAsserter:
    def __init__(self, instrumentor):
        self.instrumentor = instrumentor

    def assert_instrumented_estimator(
        self, estimator: BaseEstimator, recursive: bool = True
    ):
        if recursive:
            self._assert_instrumented_recursively(obj=estimator)
        else:
            self._assert_instrumented_estimator(estimator=estimator)

    def _assert_instrumented_estimator(self, estimator: BaseEstimator):
        for method_name in self.instrumentor.methods:
            self._assert_instrumented_method(
                estimator=estimator,
                method_name=method_name,
            )

    def _assert_instrumented_recursively(self, obj: object, full: bool = False):
        if isinstance(obj, self.instrumentor.exclude):
            if isinstance(obj, BaseEstimator):
                self._assert_uninstrumented_estimator(estimator=obj, full=full)
            return

        if isinstance(obj, BaseEstimator):
            self._assert_instrumented_estimator(estimator=obj)

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k.startswith(
                    SklearnInstrumentor._get_instrumentation_attribute_prefix()
                ):
                    continue
                self._assert_instrumented_recursively(obj=v)
        elif isinstance(obj, MutableMapping):
            for v in obj.values():
                self._assert_instrumented_recursively(obj=v)
        elif isinstance(obj, Sequence):
            for o in obj:
                self._assert_instrumented_recursively(obj=o)

    def _assert_instrumented_method(
        self,
        estimator: BaseEstimator,
        method_name: str,
    ):
        class_attribute = getattr(type(estimator), method_name, None)
        if isinstance(class_attribute, property):
            return

        instr_attrib_name = SklearnInstrumentor._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr: SklearnMethodInstrumentation = getattr(
            estimator, instr_attrib_name, None
        )
        if hasattr(estimator, method_name):
            assert instr is not None
            assert self.instrumentor.instrument in instr.instruments
        else:
            assert instr is None

    def assert_uninstrumented_estimator(
        self, estimator: BaseEstimator, recursive: bool = True, full: bool = False
    ):
        if recursive:
            self._assert_uninstrumented_recursively(obj=estimator, full=full)
        else:
            self._assert_uninstrumented_estimator(estimator=estimator, full=full)

    def _assert_uninstrumented_recursively(self, obj: object, full: bool = False):
        if isinstance(obj, self.instrumentor.exclude):
            if isinstance(obj, BaseEstimator):
                self._assert_uninstrumented_estimator(estimator=obj, full=full)
            return

        if isinstance(obj, BaseEstimator):
            self._assert_uninstrumented_estimator(estimator=obj, full=full)

        if hasattr(obj, "__dict__"):
            for k, v in obj.__dict__.items():
                if k.startswith(
                    SklearnInstrumentor._get_instrumentation_attribute_prefix()
                ):
                    continue
                self._assert_uninstrumented_recursively(obj=v, full=full)
        elif isinstance(obj, MutableMapping):
            for v in obj.values():
                self._assert_uninstrumented_recursively(obj=v, full=full)
        elif isinstance(obj, Sequence):
            for o in obj:
                self._assert_uninstrumented_recursively(obj=o, full=full)

    def _assert_uninstrumented_estimator(
        self, estimator: BaseEstimator, full: bool = False
    ):
        if full:
            methods = [f.__name__ for f in estimator.__dict__ if callable(f)]
        else:
            methods = self.instrumentor.methods

        for method_name in methods:
            self._assert_uninstrumented_method(
                estimator=estimator, method_name=method_name, full=full
            )

    def _assert_uninstrumented_method(
        self, estimator: BaseEstimator, method_name: str, full: bool = False
    ):
        class_attribute = getattr(type(estimator), method_name, None)
        if isinstance(class_attribute, property):
            return

        instr: SklearnMethodInstrumentation = getattr(
            estimator, f"_skli_{method_name}", None
        )
        if full:
            assert instr is None

        if instr is not None:
            assert self.instrumentor.instrument not in instr.instruments

    def assert_instrumented_packages(self, package_names: List[str]):
        for package_name in package_names:
            self.assert_instrumented_package(package_name=package_name)

    def assert_instrumented_package(self, package_name: str = "sklearn"):
        estimators = get_estimators_in_package(package_name=package_name)
        self.assert_instrumented_classes(estimators=estimators)

    def assert_instrumented_classes(self, estimators: Iterable[Type[BaseEstimator]]):
        for estimator in estimators:
            self.assert_instrumented_class(estimator=estimator)

    def assert_instrumented_class(self, estimator: Type[BaseEstimator]):
        if issubclass(estimator, self.instrumentor.exclude):
            return
        for method_name in self.instrumentor.methods:
            self._assert_instrumented_class_method(
                estimator=estimator, method_name=method_name
            )

    def _assert_instrumented_class_method(
        self, estimator: Type[BaseEstimator], method_name: str
    ):
        class_method = getattr(estimator, method_name, None)
        if class_method is None:
            return

        if method_is_inherited(estimator=estimator, method_name=method_name):
            return

        if isinstance(class_method, property):
            self._assert_instrumented_property(
                estimator=estimator, method_name=method_name
            )
            return

        if is_delegator(func=class_method):
            self._assert_instrumented_delegator(
                delegator=class_method, method_name=method_name
            )
            return

        instr = getattr(estimator, f"_skli_{method_name}")
        if hasattr(estimator, method_name):
            assert instr is not None
            assert self.instrumentor.instrument in instr.instruments
        else:
            assert instr is None

    def _assert_instrumented_property(
        self, estimator: Type[BaseEstimator], method_name: str
    ):
        instr_attrib_name = SklearnInstrumentor._get_instrumentation_attribute_name(
            method_name=method_name
        )
        property_: property = getattr(estimator, method_name)
        instr: SklearnMethodInstrumentation = getattr(estimator, instr_attrib_name)

        assert self.instrumentor.instrument in instr.instruments
        assert self.instrumentor.instrument_kwargs in instr.instrument_kwargs
        assert instr.wrapper == property_.fget

    def _assert_instrumented_delegator(self, delegator: Callable, method_name: str):
        instr_attrib_name = SklearnInstrumentor._get_instrumentation_attribute_name(
            method_name=method_name
        )
        descriptor = get_delegator(delegator)
        instr: SklearnMethodInstrumentation = getattr(
            descriptor, instr_attrib_name, None
        )

        assert self.instrumentor.instrument in instr.instruments
        assert self.instrumentor.instrument_kwargs in instr.instrument_kwargs
        assert instr.wrapper == descriptor.fn

    def assert_uninstrumented_packages(
        self, package_names: List[str], full: bool = False
    ):
        for package_name in package_names:
            self.assert_uninstrumented_package(package_name=package_name, full=full)

    def assert_uninstrumented_package(self, package_name: str, full: bool = False):
        estimators = get_estimators_in_package(package_name=package_name)
        self.assert_uninstrumented_classes(estimators=estimators, full=full)

    def assert_uninstrumented_classes(
        self, estimators: Iterable[Type[BaseEstimator]], full: bool = False
    ):
        for estimator in estimators:
            self.assert_uninstrumented_class(estimator=estimator, full=full)

    def assert_uninstrumented_class(
        self, estimator: Type[BaseEstimator], full: bool = False
    ):
        if issubclass(estimator, self.instrumentor.exclude):
            return
        for method_name in self.instrumentor.methods:
            self._assert_uninstrumented_class_method(
                estimator=estimator, method_name=method_name, full=full
            )

    def _assert_uninstrumented_class_method(
        self, estimator: Type[BaseEstimator], method_name: str, full: bool = False
    ):
        class_method = getattr(estimator, method_name, None)
        if class_method is None:
            return

        if method_is_inherited(estimator=estimator, method_name=method_name):
            return

        if isinstance(class_method, property):
            self._assert_uninstrumented_property(
                estimator=estimator, method_name=method_name, full=full
            )
            return

        if is_delegator(func=class_method):
            self._assert_uninstrumented_delegator(
                delegator=class_method, method_name=method_name, full=full
            )
            return

        instr = getattr(estimator, f"_skli_{method_name}", None)
        if full:
            assert instr is None

        if instr is not None:
            assert self.instrumentor.instrument not in instr.instruments

    def _assert_uninstrumented_property(
        self, estimator: Type[BaseEstimator], method_name: str, full: bool = False
    ):
        instr_attrib_name = SklearnInstrumentor._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr: SklearnMethodInstrumentation = getattr(
            estimator, instr_attrib_name, None
        )

        if full:
            assert instr is None

        if instr is not None:
            assert self.instrumentor.instrument not in instr.instruments
            assert self.instrumentor.instrument_kwargs not in instr.instrument_kwargs
            property_: property = getattr(estimator, method_name)
            assert instr.wrapped == property_.fget

    def _assert_uninstrumented_delegator(
        self, delegator: Callable, method_name: str, full: bool = False
    ):
        instr_attrib_name = SklearnInstrumentor._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr: SklearnMethodInstrumentation = getattr(
            delegator, instr_attrib_name, None
        )

        if full:
            assert instr is None

        if instr is not None:
            assert self.instrumentor.instrument not in instr.instruments
            assert self.instrumentor.instrument_kwargs not in instr.instrument_kwargs
            descriptor = get_delegator(delegator)
            assert instr.wrapped == descriptor.fn
