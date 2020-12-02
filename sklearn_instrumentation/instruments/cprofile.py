import cProfile
from collections.abc import Callable
from functools import wraps
from pathlib import Path

from sklearn_instrumentation.instruments.base import BaseInstrument


class CProfiler(BaseInstrument):
    """cProfile instrument which outputs stats dumps to disk.

    A new profiler is instantiated for each function call. After calling, profiling
    output is enabled by default, can be disabled if ``dkwargs.print_kwargs`` is set to
    ``None``. Profiling file output is created if ``dkwargs.out_dir`` is passed.

    When instrumenting, ``dkwargs`` has the following keys which contain profiler
    configuration.

    ``dkwargs``:
        * ``out_dir``: output directory for stats dumps
        * ``profiler_kwargs``: kwargs passed to ``cProfile.Profile.__init__()``
        * ``print_kwargs``: kwargs passed to ``Profiler.print_stats()``
    """

    def __init__(self):
        self.count = 0

    def __call__(self, func: Callable, **dkwargs):
        out_dir = dkwargs.get("out_dir", None)
        profiler_kwargs = dkwargs.get("profiler_kwargs", {})
        print_kwargs = dkwargs.get("print_kwargs", {})

        count = self.count
        self.count += 1

        @wraps(func)
        def wrapper(*args, **kwargs):
            pr = cProfile.Profile(**profiler_kwargs)
            pr.enable()
            retval = func(*args, **kwargs)
            pr.disable()
            if print_kwargs is not None:
                print(func.__qualname__)
                pr.print_stats(**print_kwargs)
            if out_dir is not None:
                pr.dump_stats(
                    Path(out_dir)
                    / Path(str(count) + "-" + func.__qualname__ + ".cprofile")
                )
            return retval

        return wrapper

    def reset(self):
        """Reset the instrumentation enumeration counter to 0."""
        self.count = 0
