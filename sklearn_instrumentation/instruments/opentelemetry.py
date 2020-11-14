from collections.abc import Callable

from opentelemetry.sdk.resources import Attributes
from opentelemetry.trace import get_tracer

from sklearn_instrumentation._version import __version__
from sklearn_instrumentation.instruments.base import BaseInstrument
from sklearn_instrumentation.utils import wraps


class OpenTelemetrySpanner(BaseInstrument):
    """Instrument for OpenTelemetry spans.

    Attributes may be passes on instantiation to be applied during
    instrumentation. Attributes can be supplementally passed on
    instrumentation using ``dkwargs`` under the key ``attributes``.
    Supplemental attributes will override attributes passed on
    instantiation.

    :param Attributes attributes: A map of attributes to apply to spans
        for the instrument instance.
    """

    def __init__(self, attributes: Attributes = None):
        self.attributes = attributes or {}

    def __call__(self, func: Callable, **dkwargs):

        attributes = self.attributes

        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer("sklearn_instrumentation", __version__)
            with tracer.start_as_current_span(name=func.__qualname__) as span:
                if span.is_recording():
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                    for k, v in dkwargs.get("attributes", {}).items():
                        span.set_attribute(k, v)
                return func(*args, **kwargs)

        return wrapper
