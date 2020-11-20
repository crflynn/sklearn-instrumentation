import os
import socketserver
import time
from threading import Thread

import pytest
from statsd import StatsClient

from sklearn_instrumentation import SklearnInstrumentor
from sklearn_instrumentation.instruments.statsd import StatsdTimer

# global var to store udp data
received = ""


class StatsUDPHandler(socketserver.BaseRequestHandler):
    """Socket handler to push all data into variable."""

    def handle(self):
        global received
        data = self.request[0]
        received += data.decode()


server = socketserver.UDPServer(("localhost", 8125), StatsUDPHandler)


@pytest.mark.skipif(os.getenv("CI") is not None, reason="no port allocation on ci")
def test_statsd_timer(classification_model, iris):
    thread = Thread(target=server.serve_forever)
    thread.start()
    time.sleep(1)

    client = StatsClient()
    statsd_timer = StatsdTimer(client=client, enumerate_=True)
    instrumentor = SklearnInstrumentor(
        instrument=statsd_timer, instrument_kwargs={"prefix": "mymodel"}
    )
    classification_model.fit(iris.X_train, iris.y_train)

    instrumentor.instrument_estimator_classes(classification_model)
    classification_model.predict(iris.X_test)

    time.sleep(1)
    server.shutdown()
    server.server_close()
    thread.join(1)

    # assert received == "false"
    idxs = []
    for idx, c in enumerate(received):
        if c == "|":
            idxs.append(idx + 3)
    parts = [received[i:j] for i, j in zip(idxs, idxs[1:] + [None])]
    assert len(parts) == 6
    expected_starts_with = [
        "mymodel._BasePCA.transform-0:",
        "mymodel.FeatureUnion.transform-0:",
        "mymodel.ForestClassifier.predict_proba-0:",
        "mymodel.ForestClassifier.predict-0:",
        "mymodel.Pipeline.predict-0:",
    ]
    for part, start in zip(parts[:-1], expected_starts_with):
        assert part.startswith(start)
        assert part.endswith("|ms")
