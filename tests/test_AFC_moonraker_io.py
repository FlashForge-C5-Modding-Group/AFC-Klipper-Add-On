"""Real-worker persistence contracts, using only an event-controlled loopback server."""

from __future__ import annotations

import json
import logging
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue
from urllib.parse import parse_qs
from unittest.mock import patch

import pytest

from extras.AFC_stats import AFCStats_var
from extras.AFC_utils import AFC_moonraker
from tests.conftest import MockLogger, MockReactor


WAIT = 5


class DeferredReactor(MockReactor):
    def __init__(self):
        super().__init__()
        self.callbacks = Queue()

    def register_async_callback(self, callback, waketime=None):
        self.callbacks.put(callback)

    def run_callbacks(self):
        while True:
            try:
                callback = self.callbacks.get_nowait()
            except Empty:
                return
            callback(0.0)


class RecordHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append((record.levelname.lower(), record.getMessage(), record.thread))


class ThreadLogger(MockLogger):
    def __init__(self):
        super().__init__()
        self.threads = []
        self.details = []
        self.handler = RecordHandler()
        self.logger = logging.Logger("afc-loopback-test", logging.DEBUG)
        self.logger.addHandler(self.handler)

    def _record(self, level, message, kwargs):
        self.messages.append((level, message))
        self.threads.append(threading.get_ident())
        self.details.append(kwargs)

    def debug(self, message, **kwargs):
        self._record("debug", message, kwargs)

    def info(self, message, **kwargs):
        self._record("info", message, kwargs)

    def error(self, message, **kwargs):
        self._record("error", message, kwargs)


class LoopbackIO:
    def __init__(self):
        self.records = []
        self.requests = []
        self.contents = {}
        self.statuses = {}
        self.responses = {}
        self.gates = {}
        self.escaped = []
        self.condition = threading.Condition()
        self.reactor = DeferredReactor()
        self.logger = ThreadLogger()
        self.startup_values = None
        self.startup_reads = 0
        self.workers = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self._respond()

            def do_DELETE(self):
                self._respond()

            def do_GET(self):
                self._respond()

            def log_message(self, *args):
                pass

            def _respond(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length).decode()
                if self.headers.get("Content-Type") == "application/json":
                    form = {}
                    data = json.loads(body)
                else:
                    form = parse_qs(body)
                    data = None
                namespace = form.get("namespace", [""])[0]
                key = form.get("key", [""])[0]
                value = form.get("value", [None])[0]
                with owner.condition:
                    index = len(owner.records)
                    owner.records.append((self.command, namespace, key, value))
                    owner.requests.append((self.path, form, data))
                    owner.condition.notify_all()
                gate = owner.gates.get(index)
                if gate is not None:
                    if not gate.wait(WAIT / 2):
                        owner.escaped.append(index)
                status = owner.statuses.get(index, 200)
                if status == 200:
                    if self.command == "POST":
                        owner.contents[(namespace, key)] = value
                    elif self.command == "DELETE":
                        owner.contents.pop((namespace, key), None)
                result = owner.responses.get(index, {"value": value})
                payload = json.dumps({"result": result}).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = False
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.client = self.new_client()
        self.logger.messages.clear()
        self.logger.threads.clear()
        self.logger.details.clear()

    def new_client(self):
        client = AFC_moonraker("http://127.0.0.1", str(self.server.server_port),
                              self.logger, self.reactor)
        self.workers.append(client)
        return client

    def startup_stats(self):
        self.startup_reads += 1
        return self.startup_values

    def wait_requests(self, count):
        with self.condition:
            assert self.condition.wait_for(lambda: len(self.records) >= count, timeout=WAIT)

    def wait_idle(self):
        completed = threading.Event()
        self.client._write_queue.put_nowait((completed.set, ()))
        assert completed.wait(WAIT), "worker did not reach the completion barrier"

    def close(self):
        for gate in self.gates.values():
            gate.set()
        for client in self.workers:
            client.join_thread()
            assert not client._write_thread.is_alive()
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(WAIT)
        assert not self.server_thread.is_alive()


@pytest.fixture
def moonraker_io():
    io = LoopbackIO()
    try:
        yield io
    finally:
        io.close()


@contextmanager
def joining(client):
    """Expose the point where shutdown admission has finished, without timing sleeps."""
    entered = threading.Event()
    completed = threading.Event()
    errors = []
    original_join = client._write_thread.join

    def observed_join(*args, **kwargs):
        entered.set()
        return original_join(*args, **kwargs)

    def close():
        try:
            client.join_thread()
        except BaseException as error:
            errors.append(error)
        finally:
            completed.set()

    client._write_thread.join = observed_join
    helper = threading.Thread(target=close)
    helper.start()
    try:
        assert entered.wait(WAIT)
        yield completed
    finally:
        # The caller must release its HTTP gate before this context exits.
        helper.join(WAIT + 1)
        client._write_thread.join = original_join
        assert not helper.is_alive()
        assert errors == []


class TestAFCMoonrakerInit:
    def test_constructor_owns_a_working_writer_until_explicit_join(self, moonraker_io):
        io = moonraker_io
        client = io.new_client()
        gate = io.gates[0] = threading.Event()
        client.update_afc_stats("constructed.count", 6)
        io.wait_requests(1)
        try:
            assert client._write_thread.is_alive()
            assert not gate.is_set()
            assert io.escaped == []
        finally:
            gate.set()
            client.join_thread()
        assert not client._write_thread.is_alive()
        assert io.records == [("POST", "afc_stats", "constructed.count", "6")]
        assert io.contents == {("afc_stats", "constructed.count"): "6"}
        assert io.logger.messages == [
            ("debug", f"Moonraker url: http://127.0.0.1:{io.server.server_port}"),
        ]


class TestAFCMoonrakerUpdateAfcStats:
    def test_increments_and_reset_return_before_response(self, moonraker_io):
        io = moonraker_io
        gate = io.gates[0] = threading.Event()
        stat = AFCStats_var("lane1", "load_count", None, io.client)
        stat.increase_count()
        io.wait_requests(1)
        stat.increase_count()
        stat.reset_count()
        stat.increase_count()
        try:
            assert not gate.is_set()
            assert io.escaped == []
            assert stat.value == 1
        finally:
            gate.set()
        io.wait_idle()
        assert io.records == [
            ("POST", "afc_stats", "lane1.load_count", "1"),
            ("POST", "afc_stats", "lane1.load_count", "2"),
            ("POST", "afc_stats", "lane1.load_count", "0"),
            ("POST", "afc_stats", "lane1.load_count", "1"),
        ]
        assert io.contents == {("afc_stats", "lane1.load_count"): "1"}
        assert io.logger.messages == []


class TestAFCMoonrakerRemoveDatabaseEntry:
    def test_migration_precedes_increment_and_removes_legacy_key(self, moonraker_io):
        io = moonraker_io
        io.contents[("afc_stats", "legacy.count")] = "7"
        stat = AFCStats_var("legacy", "count", {"legacy": {"count": 7}},
                            io.client, "lane1")
        stat.increase_count()
        io.wait_idle()
        io.reactor.run_callbacks()
        assert stat.value == 8
        assert io.records == [
            ("POST", "afc_stats", "lane1.count", "7"),
            ("DELETE", "afc_stats", "legacy.count", None),
            ("POST", "afc_stats", "lane1.count", "8"),
        ]
        assert io.contents == {("afc_stats", "lane1.count"): "8"}
        assert io.logger.messages == [("debug", "Removing legacy.count from afc_stats")]
        assert io.logger.threads == [threading.get_ident()]


class TestAFCMoonrakerWriteWorker:
    def test_http_error_does_not_stop_later_persistence(self, moonraker_io):
        io = moonraker_io
        io.statuses[0] = 500
        io.client.update_afc_stats("failed", 1)
        io.client.update_afc_stats("later", 2)
        io.wait_idle()
        assert io.client._write_thread.is_alive()
        assert io.contents == {("afc_stats", "later"): "2"}
        assert io.logger.messages == []
        io.reactor.run_callbacks()
        assert io.logger.messages == [
            ("error", io.client.ERROR_STRING),
            ("error", "Error when trying to update failed in moonraker, see AFC.log for more info"),
        ]
        assert "HTTP Error 500" in io.logger.details[0]["traceback"]
        assert io.logger.threads == [threading.get_ident(), threading.get_ident()]

    def test_callable_exception_is_contained(self, moonraker_io):
        io = moonraker_io

        def fail():
            raise ValueError("controlled worker failure")

        with patch("extras.AFC_utils.traceback.format_exc",
                   return_value="ValueError: controlled worker failure"):
            io.client._write_queue.put_nowait((fail, ()))
            io.client.update_afc_stats("survivor", 3)
            io.wait_idle()
        assert io.contents == {("afc_stats", "survivor"): "3"}
        assert io.logger.messages == []
        io.reactor.run_callbacks()
        assert io.logger.messages == [
            ("error", "Unexpected error in moonraker background writer"),
            ("debug", "ValueError: controlled worker failure"),
        ]
        assert io.logger.threads == [threading.get_ident(), threading.get_ident()]

    def test_native_thread_naming_failure_does_not_prevent_persistence(self, moonraker_io):
        io = moonraker_io
        completed = threading.Event()
        with patch("chelper.get_ffi", side_effect=RuntimeError("native naming unavailable")):
            client = io.new_client()
            client.update_afc_stats("survivor", 9)
            client._write_queue.put_nowait((completed.set, ()))
            assert completed.wait(WAIT)
        assert io.contents == {("afc_stats", "survivor"): "9"}
        assert io.logger.messages == [
            ("debug", f"Moonraker url: http://127.0.0.1:{io.server.server_port}"),
        ]


class TestAFCMoonrakerJoinThread:
    def test_disconnect_attempts_all_admitted_operations_in_order(self, moonraker_io):
        io = moonraker_io
        gate = io.gates[0] = threading.Event()
        io.client.update_afc_stats("first", 1)
        io.wait_requests(1)
        io.client.update_afc_stats("second", 2)
        io.client.remove_database_entry("afc_stats", "legacy.count")
        io.client.update_afc_stats("last", 3)
        with joining(io.client) as completed:
            try:
                assert not completed.is_set()
            finally:
                gate.set()
        assert completed.is_set()
        assert io.records == [
            ("POST", "afc_stats", "first", "1"),
            ("POST", "afc_stats", "second", "2"),
            ("DELETE", "afc_stats", "legacy.count", None),
            ("POST", "afc_stats", "last", "3"),
        ]
        assert io.contents == {
            ("afc_stats", "first"): "1",
            ("afc_stats", "second"): "2",
            ("afc_stats", "last"): "3",
        }
        io.client.join_thread()
        assert not io.client._write_thread.is_alive()
        with pytest.raises(RuntimeError, match="^AFC Moonraker worker is closed$"):
            io.client.update_afc_stats("rejected", 4)
        assert len(io.records) == 4


class TestAFCMoonrakerEnqueue:
    @pytest.mark.parametrize("producer", [
        lambda client: client.update_afc_stats("key", 1),
        lambda client: client.remove_database_entry("afc_stats", "key"),
        lambda client: client.get_file_metadata("file.gcode", lambda result: None),
        lambda client: client.get_spool(1, lambda result: None),
        lambda client: client.get_td1_data_async(lambda result: None),
        lambda client: client.send_lane_data({"lane1": {}}),
    ])
    def test_closed_worker_rejects_every_producer(self, moonraker_io, producer):
        io = moonraker_io
        io.client.join_thread()
        with pytest.raises(RuntimeError, match="^AFC Moonraker worker is closed$"):
            producer(io.client)
        assert io.records == []
        assert not io.client._write_thread.is_alive()

    def test_open_admission_executes_callable(self, moonraker_io):
        io = moonraker_io
        result = []
        io.client._enqueue(result.append, ("accepted",))
        io.wait_idle()
        assert result == ["accepted"]
        assert io.logger.messages == []


class TestAFCMoonrakerLogAsync:
    def test_logging_is_deferred_and_runs_on_reactor_thread(self, moonraker_io):
        io = moonraker_io
        io.client._log_async(io.logger.info, "deferred diagnostic")
        assert io.logger.messages == []
        io.reactor.run_callbacks()
        assert io.logger.messages == [("info", "deferred diagnostic")]
        assert io.logger.threads == [threading.get_ident()]
        io.reactor.run_callbacks()
        assert io.logger.messages == [("info", "deferred diagnostic")]


class TestAFCMoonrakerDrainLogs:
    def test_shutdown_preserves_diagnostics_without_ui_or_duplicates(self, moonraker_io):
        io = moonraker_io
        gate = io.gates[0] = threading.Event()
        io.statuses[0] = 500
        io.client._log_async(io.logger.info, "pending info")
        io.client._log_async(io.logger.debug, "pending debug", traceback="supplied traceback")
        io.client.update_afc_stats("failed", 1)
        io.wait_requests(1)
        io.client.update_afc_stats("later", 2)
        with joining(io.client):
            gate.set()
        assert io.logger.messages == []
        assert io.records == [
            ("POST", "afc_stats", "failed", "1"),
            ("POST", "afc_stats", "later", "2"),
        ]
        assert io.contents == {("afc_stats", "later"): "2"}
        records = io.logger.handler.records
        assert [level for level, _, _ in records] == ["info", "debug", "error", "error"]
        assert records[0][1] == "pending info"
        assert "pending debug" in records[1][1]
        assert "supplied traceback" in records[1][1]
        assert io.client.ERROR_STRING in records[2][1]
        assert "HTTP Error 500" in records[2][1]
        assert records[3][1] == (
            "Error when trying to update failed in moonraker, see AFC.log for more info")
        assert len({thread for _, _, thread in records}) == 1
        before = list(records)
        io.reactor.run_callbacks()
        io.client.join_thread()
        assert io.logger.messages == []
        assert records == before

    def test_empty_drain_is_harmless(self, moonraker_io):
        io = moonraker_io
        io.client._drain_logs()
        io.client._drain_logs(shutdown=True)
        assert io.logger.messages == []
        assert io.logger.handler.records == []


class TestAFCMoonrakerGetResults:
    def test_post_timeout_reports_error_and_next_request_persists(self, moonraker_io):
        io = moonraker_io
        io.client.REQUEST_TIMEOUT = 0.05
        gate = io.gates[0] = threading.Event()
        io.client.update_afc_stats("stalled", 1)
        io.wait_requests(1)
        io.client.update_afc_stats("later", 2)
        try:
            io.wait_idle()
            assert io.contents == {("afc_stats", "later"): "2"}
            assert io.escaped == []
        finally:
            gate.set()
        assert io.logger.messages == []
        io.reactor.run_callbacks()
        assert io.logger.messages == [
            ("error", io.client.ERROR_STRING),
            ("error", "Error when trying to update stalled in moonraker, see AFC.log for more info"),
        ]
        assert "timed out" in io.logger.details[0]["traceback"]
        assert io.logger.threads == [threading.get_ident(), threading.get_ident()]


class TestAFCMoonrakerRemoveDatabaseEntrySync:
    def test_delete_timeout_is_contained_and_later_write_persists(self, moonraker_io):
        io = moonraker_io
        io.client.REQUEST_TIMEOUT = 0.05
        gate = io.gates[0] = threading.Event()
        io.client.remove_database_entry("afc_stats", "stalled")
        io.wait_requests(1)
        io.client.update_afc_stats("later", 2)
        try:
            io.wait_idle()
            assert io.contents == {("afc_stats", "later"): "2"}
            assert io.escaped == []
        finally:
            gate.set()
        assert io.logger.messages == []
        io.reactor.run_callbacks()
        assert [level for level, _ in io.logger.messages] == ["error", "debug"]
        assert io.logger.messages[0] == (
            "error", "Unexpected error in moonraker background writer")
        assert "timed out" in io.logger.messages[1][1]
        assert io.logger.threads == [threading.get_ident(), threading.get_ident()]

    def test_http_delete_error_does_not_stop_later_write(self, moonraker_io):
        io = moonraker_io
        io.statuses[0] = 500
        io.client.remove_database_entry("afc_stats", "failed")
        io.client.update_afc_stats("later", 2)
        io.wait_idle()
        assert io.contents == {("afc_stats", "later"): "2"}
        assert io.logger.messages == []
        io.reactor.run_callbacks()
        assert io.logger.messages == [
            ("debug", "Error occurred when trying to delete failed from afc_stats namespace"),
            ("debug", "HTTP Error 500: Internal Server Error"),
        ]
        assert io.logger.threads == [threading.get_ident(), threading.get_ident()]


class TestAFCMoonrakerGetFileMetadata:
    def test_metadata_is_delivered_only_on_reactor_thread(self, moonraker_io):
        io = moonraker_io
        gate = io.gates[0] = threading.Event()
        io.responses[0] = {"filament_change_count": 3}
        results = []
        io.client.get_file_metadata("sub dir/file.gcode",
                                   lambda data: results.append((data, threading.get_ident())))
        io.wait_requests(1)
        try:
            assert results == []
        finally:
            gate.set()
        io.wait_idle()
        assert results == []
        io.reactor.run_callbacks()
        assert results == [({"filament_change_count": 3}, threading.get_ident())]
        assert io.requests == [
            ("/server/files/metadata?filename=sub%20dir/file.gcode", {}, None),
        ]
        assert io.logger.messages == []


class TestAFCMoonrakerGetSpool:
    def test_spool_is_delivered_only_on_reactor_thread(self, moonraker_io):
        io = moonraker_io
        io.responses[0] = {"id": 42, "filament": {"color_hex": "ff0000"}}
        results = []
        io.client.get_spool(42, lambda data: results.append((data, threading.get_ident())))
        io.wait_idle()
        assert results == []
        io.reactor.run_callbacks()
        assert results == [
            ({"id": 42, "filament": {"color_hex": "ff0000"}}, threading.get_ident()),
        ]
        assert io.requests == [
            ("/server/spoolman/proxy", {"request_method": ["GET"], "path": ["/v1/spool/42"]},
             None),
        ]
        assert io.logger.messages == []


class TestAFCMoonrakerGetTd1DataAsync:
    def test_device_data_is_delivered_only_on_reactor_thread(self, moonraker_io):
        io = moonraker_io
        io.responses[0] = {"devices": {"serial1": {"color": "red"}}}
        results = []
        io.client.get_td1_data_async(lambda data: results.append((data, threading.get_ident())))
        io.wait_idle()
        assert results == []
        io.reactor.run_callbacks()
        assert results == [({"serial1": {"color": "red"}}, threading.get_ident())]
        assert io.requests == [("/machine/td1/data", {}, None)]
        assert io.logger.messages == []


class TestAFCMoonrakerSendLaneData:
    def test_lane_data_persists_while_caller_is_not_waiting(self, moonraker_io):
        io = moonraker_io
        gate = io.gates[0] = threading.Event()
        io.client.send_lane_data({"lane1": {"color": "red"}})
        io.wait_requests(1)
        try:
            assert not gate.is_set()
            assert io.escaped == []
            assert io.requests == [
                ("/server/database/item", {}, {"lane1": {"color": "red"}}),
            ]
        finally:
            gate.set()
        io.wait_idle()
        assert io.logger.messages == []
