from collections.abc import Callable
from functools import wraps

from memory_profiler import profile

from sklearn_instrumentation.instruments.base import BaseInstrument
from sklearn_instrumentation.types import Estimator
from sklearn_instrumentation.utils import get_name


class MemoryProfiler(BaseInstrument):
    """Instrument which measures memory usage over function calls.

    Uses the ``memory-profiler`` library. Outputs line-by-line memory usage for
    instrumented function.

    ``dkwargs`` are passed to the ``memory_profiler.profile`` function decorator.
    """

    def __call__(self, estimator: Estimator, func: Callable, **dkwargs):
        name = get_name(estimator, func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(name)
            return profile(func, **dkwargs)(*args, **kwargs)

        return wrapper
