import json
from io import StringIO

from ddtrace import Tracer
from ddtrace.internal.writer import LogWriter

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.datadog import DatadogTraceSpanner


def _extract_spans(out: StringIO):
    all_spans = []
    captured = out.getvalue()
    for line in captured.split("\n"):
        if line == "":
            continue
        payload = json.loads(line)
        traces = payload["traces"]
        for trace in traces:
            for span in trace:
                all_spans.append(span)
    return all_spans


def test_ddtrace_spanner(classification_model, iris):

    classification_model.fit(iris.X_train, iris.y_train)

    t = Tracer()
    out = StringIO()
    t.configure(writer=LogWriter(out=out))
    instrumentor = SklearnInstrumentor(instrument=DatadogTraceSpanner(tracer=t))

    instrumentor.instrument_estimator(classification_model)

    with t.start_span("parent"):
        classification_model.predict(iris.X_test)

    all_spans = _extract_spans(out=out)
    num_spans = len(all_spans)
    assert num_spans == 7

    assert all_spans[0]["name"] == "Pipeline.predict"
    assert all_spans[0]["parent_id"] == "0000000000000000"
    assert all_spans[1]["name"] == "FeatureUnion.transform"
    assert all_spans[1]["parent_id"] == all_spans[0]["span_id"]
    assert all_spans[2]["name"] == "StandardScaler.transform"
    assert all_spans[2]["parent_id"] == all_spans[1]["span_id"]
    assert all_spans[3]["name"] == "_BasePCA.transform"
    assert all_spans[3]["parent_id"] == all_spans[1]["span_id"]
    assert all_spans[4]["name"] == "ForestClassifier.predict"
    assert all_spans[4]["parent_id"] == all_spans[0]["span_id"]
    assert all_spans[5]["name"] == "ForestClassifier.predict_proba"
    assert all_spans[5]["parent_id"] == all_spans[4]["span_id"]
    assert all_spans[6]["name"] == "parent"
    assert all_spans[6]["parent_id"] == "0000000000000000"

    instrumentor.uninstrument_estimator(classification_model)

    with t.start_span("parent"):
        classification_model.predict(iris.X_test)

    all_spans = _extract_spans(out=out)

    # only one new span
    assert len(all_spans) == num_spans + 1
