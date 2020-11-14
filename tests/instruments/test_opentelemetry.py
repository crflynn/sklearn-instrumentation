from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace import _Span
from opentelemetry.sdk.trace.export import SimpleExportSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.opentelemetry import OpenTelemetrySpanner


def test_opentelemetry_spanner(classification_model, iris):
    memory_exporter = InMemorySpanExporter()
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(
        SimpleExportSpanProcessor(memory_exporter)
    )
    tracer = trace.get_tracer(__name__)

    instrumentor = SklearnInstrumentor(instrument=OpenTelemetrySpanner())
    instrumentor.instrument_estimator(classification_model)

    with tracer.start_as_current_span("parent"):
        classification_model.fit(iris.X_train, iris.y_train)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) > 0

    span: _Span = spans[0]
    assert span.name == "StandardScaler.fit"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[3].context.span_id
    span: _Span = spans[1]
    assert span.name == "StandardScaler.transform"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[3].context.span_id
    span: _Span = spans[2]
    assert span.name == "PCA._fit"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[3].context.span_id
    span: _Span = spans[3]
    assert span.name == "Pipeline._fit"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[5].context.span_id
    span: _Span = spans[4]
    assert span.name == "BaseForest.fit"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[5].context.span_id
    span: _Span = spans[5]
    assert span.name == "Pipeline.fit"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[6].context.span_id
    span: _Span = spans[6]
    assert span.name == "parent"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent is None

    memory_exporter.clear()
    instrumentor.uninstrument_estimator(classification_model)

    with tracer.start_as_current_span("parent"):
        classification_model.fit(iris.X_train, iris.y_train)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span: _Span = spans[0]
    assert span.name == "parent"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent is None
