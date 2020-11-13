from collections import Callable
from collections import defaultdict
from threading import Lock

from prometheus_client import Histogram
from prometheus_client import Summary
from prometheus_client.metrics import MetricWrapperBase
from prometheus_client.metrics import _validate_labelnames

from sklearn_instrumentation.instruments.base import BaseInstrument
from sklearn_instrumentation.utils import wraps


class BasePrometheus(BaseInstrument):
    def __init__(self, metric: MetricWrapperBase, enumerate_: bool = False):
        if "qualname" not in metric._labelnames:
            metric._labelnames = _validate_labelnames(
                metric, list(metric._labelnames) + ["qualname"]
            )
        self.metric = metric
        self.enumerate = enumerate_
        self.enumerations = defaultdict(list)

    def __call__(self, func: Callable, **dkwargs):
        labels = {"qualname": func.__qualname__}
        labels.update(dkwargs.get("labels", {}))

        if self.enumerate:
            key = str(sorted({**dkwargs, "func": func}.items()))
            try:
                idx = self.enumerations[key].index(func)
            except ValueError:
                idx = 0
                self.enumerations[key].append(func)
            labels["qualname"] = f"{func.__qualname__}-{idx}"

        summarylabels = self.metric.labels(**labels)

        @wraps(func)
        def wrapper(*args, **kwargs):
            with summarylabels.time():
                return func(*args, **kwargs)

        return wrapper


class PrometheusSummary(BasePrometheus):
    def __init__(self, summary: Summary = None, enumerate_: bool = False):
        if summary is None:
            summary = Summary(
                "estimator_processing_seconds", "Time estimator spent processing"
            )
        if "qualname" not in summary._labelnames:
            summary._labelnames = _validate_labelnames(
                summary, list(summary._labelnames) + ["qualname"]
            )
            if summary._is_parent():
                summary._lock = Lock()
                summary._metrics = {}
        super().__init__(metric=summary, enumerate_=enumerate_)


class PrometheusHistogram(BasePrometheus):
    def __init__(self, histogram: Histogram = None, enumerate_: bool = False):
        if histogram is None:
            histogram = Histogram(
                "estimator_processing_seconds",
                "Time estimator spent processing",
            )
        super().__init__(metric=histogram, enumerate_=enumerate_)
