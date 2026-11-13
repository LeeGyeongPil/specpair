"""Plain-text tables and JSON for check, scan and bench results (no colour, fixed-width columns)."""

from __future__ import annotations

import unicodedata
from typing import Any

from .bench import BenchResult
from .i18n import t
from .rules import CheckResult, recommended_command
from .ruleset import load_ruleset
from .scan import ScanRow, row_dict
from .vocab import Vocab

SCHEMA_VERSION = 1


def display_width(text: str) -> int:
    """Terminal columns used by ``text`` (East Asian wide characters count twice)."""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def pad(text: str, width: int, right: bool = False) -> str:
    gap = max(0, width - display_width(text))
    return (" " * gap + text) if right else (text + " " * gap)


def table(headers: list[str], rows: list[list[str]], right: set[int] | None = None) -> list[str]:
    right = right or set()
    widths = [max(display_width(h), *(display_width(r[i]) for r in rows)) if rows else display_width(h)
              for i, h in enumerate(headers)]
    def line(cells: list[str]) -> str:
        return "  ".join(pad(c, widths[i], i in right) for i, c in enumerate(cells)).rstrip()
    return [line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows]


def _yn(lang: str, value: bool) -> str:
    return t(lang, "word.yes" if value else "word.no")


def describe(v: Vocab, lang: str) -> str:
    """One-line summary of a vocabulary (type, pre-tokenizer, size, BOS/EOS)."""
    kind = "GGUF" if v.source_kind == "gguf" else "HF"
    vtype = v.vocab_type + (f" ({v.model})" if v.model else "")
    parts = [kind, vtype]
    if v.pre:
        parts.append(f"pre {v.pre}")
    parts.append(t(lang, "sum.tokens", n=f"{v.n_vocab:,}"))
    for name, ident, add, explicit in (("BOS", v.bos_id, v.add_bos, v.add_bos_explicit),
                                       ("EOS", v.eos_id, v.add_eos, v.add_eos_explicit)):
        src = t(lang, "word.from_file" if explicit else "word.implicit")
        parts.append(t(lang, "sum.special", name=name, id=ident if ident is not None else "-",
                       add=_yn(lang, add), src=src))
    return " | ".join(parts)


def source_line(runtime: str, lang: str, cmd: str = "check") -> str:
    rules = load_ruleset(runtime)
    src = rules["sources"][0]
    return t(lang, "header", cmd=cmd, runtime=runtime, url=src["url"], date=src["checked"])


def check_text(result: CheckResult, target_path: str, draft_path: str, lang: str) -> str:
    lines = [source_line(result.runtime, lang)]
    lines.append(t(lang, "check.target", path=target_path))
    lines.append("        " + describe(result.target, lang))
    lines.append(t(lang, "check.draft", path=draft_path))
    lines.append("        " + describe(result.draft, lang))
    lines.append("")
    if result.compatible:
        lines.append(t(lang, "check.verdict_ok", pct=f"{result.same_id_ratio * 100:.1f}"))
    else:
        n_failed = len(result.failures)
        lines.append(t(lang, "check.verdict_bad_one") if n_failed == 1 else t(lang, "check.verdict_bad", n=n_failed))
        for failure in result.failures:
            lines.append(f"  x {failure.rule}: {failure.message(lang)}")
    if result.notes:
        lines.append("")
        lines.append(t(lang, "check.notes"))
        for note in result.notes:
            lines.append(f"  - {note.message(lang)}")
    cmd = recommended_command(result, target_path, draft_path)
    lines.append("")
    if cmd:
        lines.append(t(lang, "check.run"))
        lines.append(f"  {cmd}")
        lines.append(t(lang, "check.run_warn"))
    elif not result.compatible:
        lines.append(t(lang, "check.rejected_hint"))
    elif result.target.source_kind != "gguf" or result.draft.source_kind != "gguf":
        lines.append(t(lang, "check.hf_hint"))
    return "\n".join(lines)


def _summary(v: Vocab) -> dict[str, Any]:
    return {
        "path": v.path, "source": v.source_kind, "vocab_type": v.vocab_type, "model": v.model, "pre": v.pre,
        "arch": v.arch, "name": v.name, "n_tokens": v.n_vocab, "bos_id": v.bos_id, "eos_id": v.eos_id,
        "add_bos": v.add_bos, "add_bos_explicit": v.add_bos_explicit,
        "add_eos": v.add_eos, "add_eos_explicit": v.add_eos_explicit,
    }


def check_json(result: CheckResult, target_path: str, draft_path: str, lang: str) -> dict[str, Any]:
    """Stable ``check --json`` structure; keys never depend on the UI language."""
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime": result.runtime,
        "verdict": result.verdict,
        "failures": [{"rule": f.rule, "message": f.message(lang), "evidence": f.evidence}
                     for f in result.failures],
        "notes": [{"code": n.code, "message": n.message(lang), "params": n.params} for n in result.notes],
        "target": _summary(result.target),
        "draft": _summary(result.draft),
        "same_id_ratio": result.same_id_ratio,
        "recommended_command": recommended_command(result, target_path, draft_path),
    }


def scan_text(rows: list[ScanRow], target_path: str, directory: str, runtime: str, lang: str) -> str:
    lines = [source_line(runtime, lang, "scan"), t(lang, "scan.target", path=target_path),
             t(lang, "scan.folder", path=directory), ""]
    if not rows:
        lines.append(t(lang, "scan.empty"))
        return "\n".join(lines)
    body = []
    for row in rows:
        if row.status == "error":
            detail = row.error.message(lang).split(". ")[0] if row.error else ""
        elif row.result and row.result.failures:
            detail = "; ".join(f"{f.rule}: {f.short(lang)}" for f in row.result.failures)
        else:
            detail = t(lang, "scan.same_ids", pct=f"{row.result.same_id_ratio * 100:.1f}") if row.result else ""
        body.append([row.path, row.status, detail])
    lines += table([t(lang, "scan.col.file"), t(lang, "scan.col.status"), t(lang, "scan.col.detail")], body)
    ok = sum(1 for r in rows if r.status == "compatible")
    lines += ["", t(lang, "scan.summary", ok=ok, total=len(rows))]
    return "\n".join(lines)


def scan_json(rows: list[ScanRow], target_path: str, directory: str, runtime: str, lang: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "runtime": runtime,
        "target": target_path,
        "folder": directory,
        "results": [row_dict(r, lang) for r in rows],
        "compatible_count": sum(1 for r in rows if r.status == "compatible"),
    }


def _pct(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.1f}%" if signed else f"{value * 100:.1f}%"


def bench_text(result: BenchResult, lang: str, banner: bool = True) -> str:
    lines = []
    if result.mock and banner:
        lines.append(t(lang, "mock.banner"))
    lines.append(t(lang, "bench.title", n=len(result.configs), prompts=result.n_prompts,
                   n_predict=result.n_predict, repeat=result.repeat))
    lines.append("")
    headers = [t(lang, f"bench.col.{k}") for k in ("config", "tokens", "tps", "drafted", "accepted", "acceptance",
                                                   "speedup")]
    body = []
    for c in result.configs:
        spec = c.role == "spec"
        body.append([
            c.label, str(c.tokens), f"{c.tps:.1f}",
            str(c.draft_n) if spec and c.draft_used else "-",
            str(c.draft_accepted) if spec and c.draft_used else "-",
            _pct(c.acceptance) if spec else "-",
            _pct(c.speedup, signed=True) if spec else t(lang, "bench.baseline"),
        ])
    lines += table(headers, body, right={1, 2, 3, 4, 5, 6})
    lines += ["", t(lang, "bench.per_prompt")]
    p_headers = [t(lang, "bench.col.config"), t(lang, "bench.col.prompt"), t(lang, "bench.col.tokens"),
                 t(lang, "bench.col.tps"), t(lang, "bench.col.acceptance")]
    p_rows = []
    for c in result.configs:
        for r in c.rows:
            p_rows.append([c.label, f"#{r.prompt_index + 1}", str(r.tokens), f"{r.tps:.1f}",
                           _pct(r.acceptance) if c.role == "spec" and c.draft_used else "-"])
    lines += table(p_headers, p_rows, right={1, 2, 3, 4})
    lines.append("")
    if result.warnings:
        lines.append(t(lang, "bench.warnings"))
        for w in result.warnings:
            lines.append(f"  ! {w.message(lang)}")
    else:
        lines.append(t(lang, "bench.no_warnings"))
    lines.append(t(lang, "bench.noise"))
    return "\n".join(lines)
