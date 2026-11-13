"""Shared fixtures: fixed locale, the synthetic-file writer, helpers to run the CLI."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_fixtures", ROOT / "tools" / "make_fixtures.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_fixtures"] = module
    spec.loader.exec_module(module)
    return module


MK = _load_tool()
ENV_KEYS = ("SPECPAIR_LANG", "LC_ALL", "LC_MESSAGES", "LANG")


@pytest.fixture(autouse=True)
def fixed_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from LANG=C.UTF-8 and no other language variable, so output is English."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LANG", "C.UTF-8")


@pytest.fixture
def mk() -> ModuleType:
    """tools/make_fixtures.py as a module (``build_gguf``, ``tokenizer_kvs``, ``base_tokens`` ...)."""
    return MK


@pytest.fixture
def demo_dir(tmp_path: Path) -> Path:
    """The bundled synthetic files copied into a temp folder."""
    from specpair.demo import extract_files

    dest = tmp_path / "demo-files"
    extract_files(str(dest))
    return dest


@pytest.fixture
def write_gguf(tmp_path: Path):
    """Write a synthetic GGUF from key/value triples and return its path."""
    counter = {"n": 0}

    def _write(kvs, name: str | None = None, **kwargs) -> Path:
        counter["n"] += 1
        path = tmp_path / (name or f"synthetic-{counter['n']}.gguf")
        path.write_bytes(MK.build_gguf(kvs, **kwargs))
        return path

    return _write


@pytest.fixture
def cli(capsys: pytest.CaptureFixture[str]):
    """Run ``specpair.cli.main`` in-process and return ``(exit_code, stdout, stderr)``."""
    from specpair.cli import main

    def _run(*args: str) -> tuple[int, str, str]:
        try:
            code = main(list(args))
        except SystemExit as exc:  # argparse --help / usage errors
            code = exc.code if isinstance(exc.code, int) else 0
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return _run


def clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
    env["LANG"] = "C.UTF-8"
    env["PYTHONIOENCODING"] = "utf-8"
    env.update(extra)
    return env


def run_module(*args: str, env: dict[str, str] | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run ``python -m specpair`` as a real subprocess (to prove there is no traceback)."""
    return subprocess.run(
        [sys.executable, "-m", "specpair", *args],
        capture_output=True, text=True, encoding="utf-8", timeout=120,
        env=env or clean_env(), cwd=cwd,
    )
