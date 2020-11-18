from collections.abc import Callable
from functools import wraps
from pathlib import Path

from pyinstrument import Profiler

from sklearn_instrumentation.instruments.base import BaseInstrument


class PyInstrumentProfiler(BaseInstrument):
    """Instrument which prints pyinstrument output after function calls.

    A new profiler is instantiated for each function call. After calling, profiling
    output is enabled by default, can be disabled if ``dkwargs.text_kwargs`` is set to
    ``None``. Profiling html output is created if ``dkwargs.html_dir`` is passed.

    When instrumenting, ``dkwargs`` has two keys which contain profiler configuration.

    ``dkwargs``:
        * ``profiler_kwargs``: kwargs passed to ``Profiler.__init__()``
        * ``text_kwargs``: kwargs passed to ``Profiler.output_text()``
        * ``html_kwargs``: kwargs passed to ``Profiler.output_html()``
        * ``html_dir``: location for saving html output

    HTML output is created by enumerating instrumentations. The enumeration can be reset
    manually using the ``reset`` method.
    """

    def __init__(self):
        self.count = 0

    def __call__(self, func: Callable, **dkwargs):

        prof = Profiler(**dkwargs.get("profiler_kwargs", {}))
        text_kwargs = dkwargs.get("text_kwargs", {})
        html_kwargs = dkwargs.get("html_kwargs", {})
        html_dir = dkwargs.get("html_dir", None)
        count = self.count
        self.count += 1

        @wraps(func)
        def wrapper(*args, **kwargs):
            prof.start()
            retval = func(*args, **kwargs)
            prof.stop()
            if text_kwargs is not None:
                print(func.__qualname__)
                print(prof.output_text(**text_kwargs))
            if html_dir is not None:
                with open(
                    Path(html_dir)
                    / Path(str(count) + "-" + func.__qualname__ + ".html"),
                    "w",
                ) as f:
                    f.write(prof.output_html(**html_kwargs))
            return retval

        return wrapper

    def reset(self):
        """Reset the instrumentation enumeration counter to 0."""
        self.count = 0
