from collections import defaultdict
from collections.abc import Callable
from functools import wraps

from statsd import StatsClient

from sklearn_instrumentation.instruments.base import BaseInstrument


class StatsdTimer(BaseInstrument):
    r"""Instrument which times function calls with statsd.

    ``dkwargs`` can contain a ``prefix`` field which gets prefixed to the statsd
    timer label.

    :param statsd.StatsClient client: A statsd client
    :param bool enumerate\_: Whether to enumerate multiple instances of the
        same estimator type by appending the qualname with "-N" where N is
        the count of estimator types found in the estimator hierarchy
    """

    def __init__(self, client: StatsClient, enumerate_: bool = False):
        self.client = client
        self.enumerate = enumerate_
        self.enumerations = defaultdict(list)

    def __call__(self, func: Callable, **dkwargs):
        if self.enumerate:
            key = str(sorted({**dkwargs, "func": func}.items()))
            try:
                idx = self.enumerations[key].index(func)
            except ValueError:
                idx = len(self.enumerations[key])
                self.enumerations[key].append(func)
            suffix = f"-{idx}"
        else:
            suffix = ""

        client = self.client
        prefix = dkwargs.get("prefix", "")
        if prefix != "":
            prefix = prefix + "."
        label = prefix + func.__qualname__ + suffix

        @wraps(func)
        def wrapper(*args, **kwargs):
            with client.timer(label):
                return func(*args, **kwargs)

        return wrapper
