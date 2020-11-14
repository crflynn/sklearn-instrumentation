from collections.abc import Callable
from functools import wraps

from ddtrace import Tracer
from ddtrace import tracer as ddtracer

from sklearn_instrumentation.instruments.base import BaseInstrument


class DatadogSpanner(BaseInstrument):
    """Instrument for ddtrace spans.

    :param ddtrace.tracer.Tracer tracer: A tracer instance. Defaults to the
        ddtrace global tracer.
    """

    def __init__(self, tracer: Tracer = None):
        self.tracer = tracer or ddtracer

    def __call__(self, func: Callable, **dkwargs):

        tracer = self.tracer

        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.trace(func.__qualname__, **dkwargs):
                return func(*args, **kwargs)

        return wrapper
