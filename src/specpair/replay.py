"""Practice mock: local HTTP servers that replay recorded llama-server timings.

The bench client talks to these over real HTTP, so the whole client and the aggregation code run
exactly as they do against a real llama-server. Only the numbers are recorded.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any


def bundled_replay() -> dict[str, Any]:
    text = (resources.files("specpair") / "demo_files" / "bench_replay.json").read_text(encoding="utf-8")
    return json.loads(text)


class _Handler(BaseHTTPRequestHandler):
    server: "_ReplayHTTPServer"

    def log_message(self, format: str, *args: Any) -> None:  # silence request logging
        return

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/completion":
            self._send(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            self._send(400, {"error": "bad json"})
            return
        self._send(200, self.server.answer(str(body.get("prompt", ""))))


class _ReplayHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, records: list[dict[str, Any]], prompts: list[str], placeholder: str) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.records = records
        self.prompts = prompts
        self.placeholder = placeholder
        self._calls = 0
        self._lock = threading.Lock()

    def answer(self, prompt: str) -> dict[str, Any]:
        with self._lock:
            index = self.prompts.index(prompt) if prompt in self.prompts else self._calls % len(self.records)
            self._calls += 1
        rec = self.records[index % len(self.records)]
        timings = dict(rec)
        timings["predicted_per_second"] = rec["predicted_n"] / rec["predicted_ms"] * 1000.0
        return {"content": self.placeholder, "tokens_predicted": rec["predicted_n"], "timings": timings}


class ReplayServers:
    """Context manager that starts one replay server per recorded configuration (ephemeral ports)."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data = data or bundled_replay()
        self.prompts: list[str] = list(self.data["prompts"])
        self._servers: dict[str, _ReplayHTTPServer] = {}
        self._threads: list[threading.Thread] = []
        self._started: set[str] = set()

    def __enter__(self) -> "ReplayServers":
        try:
            for cfg in self.data["configs"]:
                srv = _ReplayHTTPServer(cfg["records"], self.prompts, self.data["content_placeholder"])
                self._servers[cfg["label"]] = srv
                thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
                thread.start()
                self._threads.append(thread)
                self._started.add(cfg["label"])
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        for label, srv in self._servers.items():
            if label in self._started:  # shutdown() blocks forever if serve_forever never ran
                srv.shutdown()
            srv.server_close()
        for thread in self._threads:
            thread.join(timeout=5)
        self._servers.clear()
        self._threads.clear()
        self._started.clear()

    def url(self, label: str) -> str:
        host, port = self._servers[label].server_address[:2]
        return f"http://{host}:{port}"

    @property
    def base(self) -> tuple[str, str]:
        label = next(c["label"] for c in self.data["configs"] if c["role"] == "base")
        return label, self.url(label)

    @property
    def specs(self) -> list[tuple[str, str]]:
        return [(c["label"], self.url(c["label"])) for c in self.data["configs"] if c["role"] != "base"]
