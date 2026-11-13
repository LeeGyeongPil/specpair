"""AC-6: bench numbers come from server-reported timings; warnings and exit codes; error paths."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from specpair.bench import DEFAULT_PROMPTS, BenchError, load_prompts, parse_spec, run_bench
from specpair.replay import ReplayServers, bundled_replay


@pytest.fixture(scope="module")
def mock_result():
    with ReplayServers() as servers:
        return run_bench(servers.base, servers.specs, prompts=servers.prompts, mock=True)


def cfg(result, label):
    return next(c for c in result.configs if c.label == label)


def test_n3_acceptance_and_speedup(mock_result):
    n3 = cfg(mock_result, "n3")
    assert (n3.draft_accepted, n3.draft_n) == (160, 230)
    assert round(n3.acceptance * 100, 1) == 69.6
    assert round(n3.speedup * 100, 1) == 5.0


def test_n8_is_slower_than_baseline(mock_result):
    n8 = cfg(mock_result, "n8")
    assert (n8.draft_accepted, n8.draft_n) == (186, 397)
    assert round(n8.acceptance * 100, 1) == 46.9
    assert round(n8.speedup * 100, 1) == -29.7
    warning = next(w for w in mock_result.warnings if w.code == "slower")
    assert warning.label == "n8"
    assert "slower than baseline" in warning.message("en")


def test_rejected_pair_reports_draft_not_used(mock_result):
    rejected = cfg(mock_result, "rejected")
    assert not rejected.draft_used and rejected.acceptance is None
    warning = next(w for w in mock_result.warnings if w.code == "draft_unused")
    assert warning.label == "rejected"
    assert "draft not used" in warning.message("en")
    assert mock_result.exit_code == 1


def test_baseline_throughput_matches_recorded_timings(mock_result):
    base = cfg(mock_result, "baseline")
    assert base.tokens == 240
    assert round(base.tps, 1) == 65.4


def test_n3_alone_is_a_clean_run():
    data = bundled_replay()
    data["configs"] = [c for c in data["configs"] if c["label"] in ("baseline", "n3")]
    with ReplayServers(data) as servers:
        result = run_bench(servers.base, servers.specs, prompts=servers.prompts)
    assert result.warnings == [] and result.exit_code == 0


def test_cli_mock_output_and_exit_code(cli):
    code, out, _ = cli("bench", "--mock")
    assert code == 1
    assert out.splitlines()[0].startswith("PRACTICE MOCK")
    for fragment in ("69.6%", "+5.0%", "46.9%", "-29.7%", "slower than baseline", "draft not used"):
        assert fragment in out


def test_cli_mock_json(cli):
    code, out, _ = cli("bench", "--mock", "--json")
    data = json.loads(out)
    assert code == 1 and data["mock"] is True and data["exit_code"] == 1
    by_label = {c["label"]: c for c in data["configs"]}
    assert by_label["n3"]["acceptance_percent"] == 69.6 and by_label["n3"]["speedup_percent"] == 5.0
    assert by_label["n8"]["speedup_percent"] == -29.7
    assert by_label["rejected"]["draft_used"] is False
    assert {w["code"] for w in data["warnings"]} == {"slower", "draft_unused"}
    assert by_label["n3"]["url"] is None  # replay ports are random and never shown


def test_every_server_gets_one_warmup_request():
    with ReplayServers() as servers:
        run_bench(servers.base, servers.specs, prompts=servers.prompts, repeat=2)
        calls = {label: srv._calls for label, srv in servers._servers.items()}
    assert set(calls.values()) == {1 + 3 * 2}


def test_replay_servers_release_their_ports():
    with ReplayServers() as servers:
        url = servers.url("n3")
    host, port = url.removeprefix("http://").split(":")
    with pytest.raises(OSError), socket.create_connection((host, int(port)), timeout=1):
        pass


def test_default_prompts_match_the_recording():
    assert DEFAULT_PROMPTS == bundled_replay()["prompts"]


# --- servers that are not there or misbehave ---------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    health_status = 200
    completion_body = b"{}"

    def log_message(self, *args):
        return

    def do_GET(self):
        self.send_response(self.health_status)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.completion_body)))
        self.end_headers()
        self.wfile.write(self.completion_body)


@pytest.fixture
def fake_server():
    servers = []

    def start(health_status=200, completion_body=b"{}"):
        handler = type("H", (_Handler,), {"health_status": health_status, "completion_body": completion_body})
        srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        servers.append((srv, thread))
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for srv, thread in servers:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)


def free_port_url():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def timings_body(content, **timings):
    return json.dumps({"content": content, "timings": timings}).encode()


def test_connection_failure_is_exit_2(cli):
    code, out, err = cli("bench", "--base", free_port_url(), "--spec", free_port_url())
    assert code == 2 and out == ""
    assert "cannot reach llama-server" in err and "Traceback" not in err


def test_health_not_200_is_exit_2(cli, fake_server):
    loading = fake_server(health_status=503)
    code, _, err = cli("bench", "--base", loading, "--spec", loading)
    assert code == 2 and "/health answered HTTP 503" in err


def test_answer_without_timings_is_exit_2(cli, fake_server):
    url = fake_server(completion_body=b'{"content": "hi"}')
    code, _, err = cli("bench", "--base", url, "--spec", url + "/")
    assert code == 2 and "timings" in err


def test_non_json_answer_is_exit_2(cli, fake_server):
    url = fake_server(completion_body=b"not json")
    code, _, err = cli("bench", "--base", url, "--spec", url)
    assert code == 2 and "Traceback" not in err


def test_missing_arguments_and_bad_values(cli):
    assert cli("bench")[0] == 2
    code, _, err = cli("bench", "--base", "ftp://x", "--spec", "http://127.0.0.1:1")
    assert code == 2 and "not an http(s) URL" in err
    code, _, err = cli("bench", "--base", "http://127.0.0.1:1", "--spec", "http://127.0.0.1:2", "--repeat", "0")
    assert code == 2 and "must be positive" in err
    code, _, err = cli("bench", "--mock", "--base", "http://127.0.0.1:1")
    assert code == 2 and "--mock cannot be combined" in err


def test_duplicate_labels_are_rejected():
    with pytest.raises(BenchError):
        run_bench(("a", "http://127.0.0.1:1"), [("a", "http://127.0.0.1:2")])


def test_two_times_faster_server_with_75_percent_acceptance(fake_server):
    base = fake_server(completion_body=timings_body("x", predicted_n=10, predicted_ms=200.0))
    fast = fake_server(completion_body=timings_body("x", predicted_n=10, predicted_ms=100.0,
                                                    draft_n=8, draft_n_accepted=6))
    result = run_bench(("baseline", base), [("fast", fast)], prompts=["p1", "p2"])
    fast_cfg = result.configs[1]
    assert round(fast_cfg.speedup, 2) == 1.0
    assert fast_cfg.acceptance == 0.75
    assert result.warnings == [] and result.exit_code == 0


def test_output_difference_is_a_warning_not_a_failure(fake_server):
    base = fake_server(completion_body=timings_body("a", predicted_n=10, predicted_ms=200.0))
    other = fake_server(completion_body=timings_body("b", predicted_n=10, predicted_ms=100.0,
                                                     draft_n=4, draft_n_accepted=3))
    result = run_bench(("baseline", base), [("x", other)], prompts=["p"])
    assert [w.code for w in result.warnings] == ["output_differs"]
    assert result.exit_code == 0


def test_parse_spec_and_prompt_files(tmp_path):
    assert parse_spec("fast=http://127.0.0.1:8081", 1) == ("fast", "http://127.0.0.1:8081")
    assert parse_spec("http://127.0.0.1:8081", 2) == ("spec2", "http://127.0.0.1:8081")
    txt = tmp_path / "p.txt"
    txt.write_text("one\n\n two \n", encoding="utf-8")
    assert load_prompts(str(txt)) == ["one", "two"]
    jl = tmp_path / "p.jsonl"
    jl.write_text('"a"\n{"prompt": "b"}\n', encoding="utf-8")
    assert load_prompts(str(jl)) == ["a", "b"]


@pytest.mark.parametrize(("name", "content", "fragment"), [
    ("bad.txt", b"\xff\xfe\x00", "not valid UTF-8"),
    ("bad.jsonl", b'{"nope": 1}\n', "line 1"),
    ("empty.txt", b"\n\n", "no prompts"),
])
def test_prompt_file_errors_are_exit_2(tmp_path, cli, name, content, fragment):
    path = tmp_path / name
    path.write_bytes(content)
    code, _, err = cli("bench", "--base", "http://127.0.0.1:1", "--spec", "http://127.0.0.1:2",
                       "--prompts", str(path))
    assert code == 2 and fragment in err and "Traceback" not in err


def test_missing_prompt_file_is_exit_2(cli):
    code, _, err = cli("bench", "--base", "http://127.0.0.1:1", "--spec", "http://127.0.0.1:2",
                       "--prompts", "/home/me/prompts.txt")
    assert code == 2 and "cannot read prompts file" in err
