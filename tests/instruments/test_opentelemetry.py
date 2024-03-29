from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace import _Span
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.opentelemetry import OpenTelemetrySpanner


def test_opentelemetry_spanner(classification_model, iris):
    memory_exporter = InMemorySpanExporter()
    trace.set_tracer_provider(TracerProvider())
    trace.get_tracer_provider().add_span_processor(SimpleSpanProcessor(memory_exporter))
    tracer = trace.get_tracer(__name__)

    instrumentor = SklearnInstrumentor(instrument=OpenTelemetrySpanner())
    instrumentor.instrument_instance(classification_model)

    with tracer.start_as_current_span("parent"):
        classification_model.fit(iris.X_train, iris.y_train)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 13

    span: _Span = spans[0]
    assert span.name == "StandardScaler.fit"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[2].context.span_id
    span: _Span = spans[1]
    assert span.name == "StandardScaler.transform"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[2].context.span_id
    span: _Span = spans[2]
    assert span.name == "StandardScaler.fit_transform (TransformerMixin.fit_transform)"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[5].context.span_id
    span: _Span = spans[3]
    assert span.name == "PCA._fit"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[4].context.span_id
    span: _Span = spans[4]
    assert span.name == "PCA.fit_transform"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[5].context.span_id
    span: _Span = spans[5]
    assert span.name == "FeatureUnion.fit_transform"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[9].context.span_id
    span: _Span = spans[6]
    assert span.name == "TransformerWithEnum.fit"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id == spans[8].context.span_id
    span: _Span = spans[7]
    assert span.name == "TransformerWithEnum.transform"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id is spans[8].context.span_id
    span: _Span = spans[8]
    assert (
        span.name
        == "TransformerWithEnum.fit_transform (TransformerMixin.fit_transform)"
    )
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id is spans[9].context.span_id
    span: _Span = spans[9]
    assert span.name == "Pipeline._fit"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id is spans[11].context.span_id
    span: _Span = spans[10]
    assert span.name == "RandomForestClassifier.fit (BaseForest.fit)"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id is spans[11].context.span_id
    span: _Span = spans[11]
    assert span.name == "Pipeline.fit"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent.span_id is spans[12].context.span_id
    span: _Span = spans[12]
    assert span.name == "parent"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent is None

    memory_exporter.clear()
    instrumentor.uninstrument_instance(classification_model)

    with tracer.start_as_current_span("parent"):
        classification_model.fit(iris.X_train, iris.y_train)

    spans = memory_exporter.get_finished_spans()
    assert len(spans) == 1

    span: _Span = spans[0]
    assert span.name == "parent"
    assert span.kind == SpanKind.INTERNAL
    assert span.parent is None
