from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import Sequence
from typing import Type

from sklearn.base import BaseEstimator

from sklearn_instrumentation.instrumentor import SklearnInstrumentor
from sklearn_instrumentation.instrumentor import SklearnMethodInstrumentation
from sklearn_instrumentation.utils import get_descriptor
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
                if isinstance(obj, BaseEstimator) and k in self.instrumentor.methods:
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
            assert self.instrumentor.instrument in instr.callables
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
                if isinstance(obj, BaseEstimator) and k in self.instrumentor.methods:
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
            methods = []
            prefix = SklearnInstrumentor._get_instrumentation_attribute_prefix()
            for name, attr in estimator.__dict__.items():
                if name.startswith(prefix) and isinstance(
                    attr, SklearnMethodInstrumentation
                ):
                    method = name.replace(prefix, "")
                    methods.append(method)
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

        instr_attrib_name = SklearnInstrumentor._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr: SklearnMethodInstrumentation = getattr(
            estimator, instr_attrib_name, None
        )
        if full:
            assert instr is None

        if instr is not None:
            assert self.instrumentor.instrument not in instr.callables

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

        if isinstance(class_method, property):
            self._assert_instrumented_property(
                estimator=estimator, method_name=method_name
            )
            return

        if is_delegator(func=class_method):
            self._assert_instrumented_delegator(
                estimator=estimator, method_name=method_name
            )
            return

        instr_attrib_name = SklearnInstrumentor._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr: SklearnMethodInstrumentation = getattr(estimator, instr_attrib_name)
        if hasattr(estimator, method_name):
            assert instr is not None
            assert self.instrumentor.instrument in instr.callables
            try:
                assert instr.estimator == estimator
            except AssertionError:
                pass
            assert instr.inherited == method_is_inherited(estimator, instr.func)
        else:
            assert instr is None

    def _assert_instrumented_property(
        self, estimator: Type[BaseEstimator], method_name: str
    ):
        instr_attrib_name = SklearnInstrumentor._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr: SklearnMethodInstrumentation = getattr(estimator, instr_attrib_name)
        try:
            assert instr.estimator == estimator
        except AssertionError:
            pass
        assert instr.inherited == method_is_inherited(estimator, instr.func)

        property_: property = getattr(estimator, method_name)

        assert self.instrumentor.instrument in instr.callables
        assert self.instrumentor.instrument_kwargs in instr.kwargs
        assert instr.instrumented_func == property_.fget

    def _assert_instrumented_delegator(
        self, estimator: Type[BaseEstimator], method_name: str
    ):
        instr_attrib_name = SklearnInstrumentor._get_instrumentation_attribute_name(
            method_name=method_name
        )
        instr = getattr(estimator, instr_attrib_name, None)
        assert instr.estimator == estimator
        assert instr.inherited == method_is_inherited(estimator, instr.func)

        delegator = getattr(estimator, method_name)
        descriptor = get_descriptor(delegator)

        assert self.instrumentor.instrument in instr.callables
        assert self.instrumentor.instrument_kwargs in instr.kwargs
        assert instr.instrumented_func == descriptor.fn

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

        if isinstance(class_method, property):
            self._assert_uninstrumented_property(
                estimator=estimator, method_name=method_name, full=full
            )
            return

        if is_delegator(func=class_method):
            self._assert_uninstrumented_delegator(
                estimator=estimator, method_name=method_name, full=full
            )
            return

        instr_attrib_name = SklearnInstrumentor._get_instrumentation_attribute_name(
            method_name
        )
        instr = getattr(estimator, instr_attrib_name, None)
        if full:
            assert instr is None

        if instr is not None:
            assert self.instrumentor.instrument not in instr.callables
            assert instr.func == class_method
            assert instr.estimator == estimator
            assert instr.inherited == method_is_inherited(estimator, class_method)
            if instr.inherited:
                assert method_name not in estimator.__dict__
            else:
                assert method_name in estimator.__dict__

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
            assert self.instrumentor.instrument not in instr.callables
            assert self.instrumentor.instrument_kwargs not in instr.kwargs
            property_: property = getattr(estimator, method_name)
            assert instr.func == property_.fget
            assert instr.estimator == estimator
            assert instr.inherited == method_is_inherited(estimator, property_.fget)

    def _assert_uninstrumented_delegator(
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
            delegator = getattr(estimator, method_name)
            descriptor = get_descriptor(delegator)
            assert self.instrumentor.instrument not in instr.callables
            assert self.instrumentor.instrument_kwargs not in instr.kwargs
            assert instr.func == descriptor.fn
            assert instr.estimator == estimator
            assert instr.inherited == method_is_inherited(estimator, descriptor.fn)
