from collections.abc import Callable
from functools import wraps

from memory_profiler import profile

from sklearn_instrumentation.instruments.base import BaseInstrument


class MemoryProfiler(BaseInstrument):
    """Instrument which measures memory usage over function calls.

    Uses the ``memory-profiler`` library. Outputs line-by-line memory usage for
    instrumented function.

    ``dkwargs`` are passed to the ``memory_profiler.profile`` function decorator.
    """

    def __call__(self, func: Callable, **dkwargs):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(func.__qualname__)
            return profile(func, **dkwargs)(*args, **kwargs)

        return wrapper
