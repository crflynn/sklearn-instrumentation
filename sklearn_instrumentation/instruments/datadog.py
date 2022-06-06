from collections.abc import Callable
from functools import wraps

from ddtrace import Tracer
from ddtrace import tracer as ddtracer

from sklearn_instrumentation.instruments.base import BaseInstrument
from sklearn_instrumentation.types import Estimator
from sklearn_instrumentation.utils import get_name


class DatadogSpanner(BaseInstrument):
    """Instrument for ddtrace spans.

    :param ddtrace.tracer.Tracer tracer: A tracer instance. Defaults to the
        ddtrace global tracer.
    """

    def __init__(self, tracer: Tracer = None):
        self.tracer = tracer or ddtracer

    def __call__(self, estimator: Estimator, func: Callable, **dkwargs):
        name = get_name(estimator, func)

        tracer = self.tracer

        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace(name, **dkwargs):
                return func(*args, **kwargs)

        return wrapper
