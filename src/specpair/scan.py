"""Scan a folder for GGUF files that can act as draft models for a target."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .errors import InputNotFoundError, SpecpairError
from .gguf_header import split_info
from .rules import CheckResult, check_pair
from .ruleset import DEFAULT_RUNTIME
from .vocab import Vocab, load_vocab


@dataclass
class ScanRow:
    """One scanned file: ``status`` is ``compatible``, ``rejected`` or ``error``."""

    path: str
    status: str
    result: CheckResult | None = None
    error: SpecpairError | None = None


def _candidates(directory: str, shown_dir: str) -> list[tuple[str, str]]:
    """Return ``(display_path, real_path)`` for ``*.gguf`` in ``directory`` and its direct sub-folders."""
    found: list[tuple[str, str]] = []
    try:
        entries = sorted(os.listdir(directory))
    except OSError as exc:
        raise InputNotFoundError("unreadable", path=directory, reason=exc.strerror or str(exc)) from exc
    for name in entries:
        if name.startswith("."):
            continue
        real = os.path.join(directory, name)
        if os.path.isdir(real):
            try:
                sub = sorted(os.listdir(real))
            except OSError:
                continue
            found += [(os.path.join(shown_dir, name, s), os.path.join(real, s))
                      for s in sub if s.lower().endswith(".gguf") and not s.startswith(".")
                      and os.path.isfile(os.path.join(real, s))]
        elif name.lower().endswith(".gguf"):
            found.append((os.path.join(shown_dir, name), real))
    return found


def scan_folder(target: Vocab, target_path: str, directory: str, runtime: str = DEFAULT_RUNTIME,
                display_dir: str | None = None) -> list[ScanRow]:
    """Check every GGUF under ``directory`` as a draft for ``target``.

    Corrupt or truncated files become ``error`` rows and scanning continues; later split parts
    (``-00002-of-00003``) are skipped because only the first part holds the tokenizer.
    ``display_dir`` replaces ``directory`` in the paths shown to the user.
    """
    if not os.path.isdir(directory):
        raise InputNotFoundError("not_a_folder", path=directory)
    rows: list[ScanRow] = []
    for shown, real in _candidates(directory, display_dir or directory):
        info = split_info(real)
        if info and info[0] > 1:
            continue
        try:
            if os.path.exists(target_path) and os.path.samefile(real, target_path):
                continue
        except OSError:
            pass
        try:
            draft = load_vocab(real, runtime, label=shown)
            result = check_pair(target, draft, runtime)
            rows.append(ScanRow(shown, result.verdict, result))
        except SpecpairError as exc:
            rows.append(ScanRow(shown, "error", error=exc))
    return rows


def scan_exit_code(rows: list[ScanRow]) -> int:
    """0 when at least one compatible draft exists, otherwise 1."""
    return 0 if any(r.status == "compatible" for r in rows) else 1


def row_dict(row: ScanRow, lang: str) -> dict[str, Any]:
    out: dict[str, Any] = {"path": row.path, "status": row.status}
    if row.result is not None:
        out["failed_rules"] = [f.rule for f in row.result.failures]
        out["first_reason"] = row.result.failures[0].message(lang) if row.result.failures else None
        out["n_vocab"] = row.result.n_vocab[1]
    if row.error is not None:
        out["error"] = {"code": row.error.code, "message": row.error.message(lang)}
    return out
