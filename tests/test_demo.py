"""AC-7: the offline tour, and the extracted practice files."""

from __future__ import annotations

import time

from conftest import clean_env, run_module


def test_demo_runs_offline_within_ten_seconds():
    start = time.monotonic()
    proc = run_module("demo")
    elapsed = time.monotonic() - start
    assert proc.returncode == 0, proc.stderr
    assert elapsed < 10
    assert proc.stderr == ""
    first_line = proc.stdout.splitlines()[0]
    assert first_line.startswith("PRACTICE MOCK")
    for fragment in ("rejected", "--spec-type draft-simple", "draft not used", "slower than baseline"):
        assert fragment in proc.stdout


def test_demo_shows_every_rejection_reason():
    out = run_module("demo").stdout
    for fragment in ("x bos:", "x vocab_size:", "x token_text:", "first mismatch at id 5"):
        assert fragment in out
    assert "demo-files/target.gguf" in out  # shown paths never reveal the temp folder
    assert "/tmp" not in out and "/var/" not in out


def test_demo_in_korean_starts_with_the_mock_notice():
    proc = run_module("--lang", "ko", "demo")
    assert proc.returncode == 0
    assert proc.stdout.splitlines()[0].startswith("연습용 목(mock)")
    assert "draft not used" in proc.stdout or "드래프트가 쓰이지 않았습니다" in proc.stdout


def test_extract_then_check_by_hand(tmp_path):
    proc = run_module("demo", "--extract", "demo-files", cwd=tmp_path)
    assert proc.returncode == 0
    assert proc.stdout.splitlines()[0].startswith("PRACTICE MOCK")
    assert "specpair check demo-files/target.gguf demo-files/draft-other.gguf" in proc.stdout
    for name in ("target.gguf", "draft-ok.gguf", "draft-wide.gguf", "draft-other.gguf", "draft-spm.gguf",
                 "broken.gguf", "hf-draft/tokenizer.json", "bench_replay.json"):
        assert (tmp_path / "demo-files" / name).is_file(), name
    check = run_module("check", "demo-files/target.gguf", "demo-files/draft-other.gguf", cwd=tmp_path,
                       env=clean_env())
    assert check.returncode == 1
    assert "demo-files/draft-other.gguf" in check.stdout and str(tmp_path) not in check.stdout


def test_extract_is_repeatable(tmp_path):
    for _ in range(2):
        assert run_module("demo", "--extract", str(tmp_path / "out")).returncode == 0
