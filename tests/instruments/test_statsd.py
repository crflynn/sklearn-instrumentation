import socketserver
from threading import Thread

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


def test_statsd_timer(classification_model, iris):
    thread = Thread(target=server.serve_forever)
    thread.start()

    client = StatsClient()
    statsd_timer = StatsdTimer(client=client)
    instrumentor = SklearnInstrumentor(instrument=statsd_timer)
    classification_model.fit(iris.X_train, iris.y_train)

    instrumentor.uninstrument_estimator(classification_model)
    classification_model.predict(iris.X_test)

    server.shutdown()
    server.server_close()
    thread.join(1)

    # assert received == "false"
    idxs = []
    for idx, c in enumerate(received):
        if c == "|":
            idxs.append(idx + 3)
    parts = [received[i:j] for i, j in zip(idxs, idxs[1:] + [None])]
    assert len(parts) > 0
    for part in parts[:-1]:
        assert part.startswith("mymodel.")
        assert part.endswith("|ms")
