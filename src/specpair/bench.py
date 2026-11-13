"""Send the same prompts to a baseline llama-server and to draft-enabled servers and compare timings."""

from __future__ import annotations

import http.client
import json
import re
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .errors import BenchError

DEFAULT_PROMPTS = [
    "Write a Python function that returns the first n Fibonacci numbers as a list.",
    "List the planets of the solar system in order from the sun, one per line.",
    "Explain in three sentences what a hash map is.",
]
SLOWER_THRESHOLD = -0.02  # a speedup below -2% counts as slower than baseline (single-run noise is smaller)


@dataclass
class PromptRow:
    prompt_index: int
    tokens: int
    ms: float
    draft_n: int
    draft_accepted: int

    @property
    def tps(self) -> float:
        return self.tokens / self.ms * 1000.0 if self.ms else 0.0

    @property
    def acceptance(self) -> float | None:
        return self.draft_accepted / self.draft_n if self.draft_n else None


@dataclass
class ConfigResult:
    label: str
    url: str
    role: str  # "base" | "spec"
    rows: list[PromptRow] = field(default_factory=list)
    contents: list[str] = field(default_factory=list)
    speedup: float | None = None

    @property
    def tokens(self) -> int:
        return sum(r.tokens for r in self.rows)

    @property
    def ms(self) -> float:
        return sum(r.ms for r in self.rows)

    @property
    def tps(self) -> float:
        return self.tokens / self.ms * 1000.0 if self.ms else 0.0

    @property
    def draft_n(self) -> int:
        return sum(r.draft_n for r in self.rows)

    @property
    def draft_accepted(self) -> int:
        return sum(r.draft_accepted for r in self.rows)

    @property
    def acceptance(self) -> float | None:
        return self.draft_accepted / self.draft_n if self.draft_n else None

    @property
    def draft_used(self) -> bool:
        return self.draft_n > 0


@dataclass(frozen=True)
class BenchWarning:
    code: str  # draft_unused | slower | output_differs
    label: str
    params: dict[str, Any] = field(default_factory=dict)

    def message(self, lang: str = "en") -> str:
        from .i18n import t

        return t(lang, f"warn.{self.code}", label=self.label, **self.params)


@dataclass
class BenchResult:
    configs: list[ConfigResult]
    warnings: list[BenchWarning]
    n_predict: int
    repeat: int
    n_prompts: int
    mock: bool = False

    @property
    def exit_code(self) -> int:
        """1 when a draft was not used or a setting is slower than baseline, else 0."""
        return 1 if any(w.code in ("draft_unused", "slower") for w in self.warnings) else 0


def parse_spec(arg: str, index: int) -> tuple[str, str]:
    """Split ``[LABEL=]URL`` into ``(label, url)``; unlabeled entries become ``spec<index>``."""
    m = re.match(r"^([^=/:\s]+)=(.+)$", arg)
    if m:
        return m.group(1), m.group(2)
    return f"spec{index}", arg


def load_prompts(path: str) -> list[str]:
    """Read prompts from a text file (one per line) or a ``.jsonl`` file (string or ``{"prompt": ...}`` per line)."""
    try:
        with open(path, "rb") as fh:
            text = fh.read().decode("utf-8")
    except OSError as exc:
        raise BenchError("prompts_unreadable", path=path, reason=exc.strerror or str(exc)) from exc
    except UnicodeDecodeError as exc:
        raise BenchError("prompts_utf8", path=path, offset=exc.start) from None
    prompts: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        if path.lower().endswith(".jsonl"):
            try:
                item = json.loads(line)
            except ValueError:
                raise BenchError("prompts_json", path=path, line=number) from None
            if isinstance(item, dict):
                item = item.get("prompt")
            if not isinstance(item, str) or not item.strip():
                raise BenchError("prompts_json", path=path, line=number)
            prompts.append(item)
        else:
            prompts.append(line.strip())
    if not prompts:
        raise BenchError("prompts_empty", path=path)
    return prompts


def _normalise_url(url: str) -> str:
    if not re.match(r"^https?://[^/\s]+", url):
        raise BenchError("bad_url", url=url)
    return url.rstrip("/")


def _request(url: str, timeout: float, body: dict[str, Any] | None = None) -> tuple[int, bytes]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (user-chosen http(s) URL)
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()
    except (socket.timeout, TimeoutError):
        raise BenchError("bench_timeout", url=url, timeout=timeout) from None
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            raise BenchError("bench_timeout", url=url, timeout=timeout) from None
        raise BenchError("bench_connect", url=url, reason=str(exc.reason)) from None
    except (OSError, ValueError, http.client.HTTPException) as exc:  # connection reset, bad status line, ...
        raise BenchError("bench_connect", url=url, reason=str(exc)) from None


def check_health(url: str, timeout: float) -> None:
    status, _ = _request(url + "/health", timeout)
    if status != 200:
        raise BenchError("bench_health", url=url, status=status)


def _complete(url: str, prompt: str, n_predict: int, timeout: float) -> tuple[PromptRow, str]:
    body = {"prompt": prompt, "n_predict": n_predict, "temperature": 0, "seed": 1, "cache_prompt": False}
    status, raw = _request(url + "/completion", timeout, body)
    if status != 200:
        raise BenchError("bench_http", url=url, status=status)
    try:
        data = json.loads(raw.decode("utf-8"))
        timings = data["timings"]
        tokens = int(timings["predicted_n"])
        ms = float(timings["predicted_ms"])
    except (ValueError, KeyError, TypeError):
        raise BenchError("bench_no_timings", url=url) from None
    row = PromptRow(
        prompt_index=0,
        tokens=tokens,
        ms=ms,
        draft_n=int(timings.get("draft_n") or 0),
        draft_accepted=int(timings.get("draft_n_accepted") or 0),
    )
    return row, str(data.get("content", ""))


def run_bench(
    base: tuple[str, str],
    specs: list[tuple[str, str]],
    prompts: list[str] | None = None,
    n_predict: int = 128,
    repeat: int = 1,
    timeout: float = 120.0,
    mock: bool = False,
) -> BenchResult:
    """Measure ``base`` (no draft) and every ``spec`` server with identical prompts.

    Each server gets one unmeasured warm-up request, then every prompt ``repeat`` times.
    Speed comes from the timings the server reports, not from wall-clock time.
    Raises :class:`BenchError` for unreachable servers or unusable answers.
    """
    prompts = prompts or DEFAULT_PROMPTS
    if n_predict < 1 or repeat < 1 or timeout <= 0:
        raise BenchError("bad_numbers")
    entries = [(base[0], _normalise_url(base[1]), "base")] + [(lab, _normalise_url(u), "spec") for lab, u in specs]
    labels = [e[0] for e in entries]
    if len(set(labels)) != len(labels):
        raise BenchError("duplicate_label")
    for _, url, _ in entries:  # fail fast, before any slow request
        check_health(url, timeout)

    results: list[ConfigResult] = []
    for label, url, role in entries:
        cfg = ConfigResult(label=label, url=url, role=role)
        _complete(url, prompts[0], n_predict, timeout)  # warm-up (first request is slower)
        for _ in range(repeat):
            for index, prompt in enumerate(prompts):
                row, content = _complete(url, prompt, n_predict, timeout)
                row.prompt_index = index
                cfg.rows.append(row)
                cfg.contents.append(content)
        results.append(cfg)

    base_cfg = results[0]
    warnings: list[BenchWarning] = []
    for cfg in results[1:]:
        cfg.speedup = (cfg.tps / base_cfg.tps - 1.0) if base_cfg.tps else None
        if not cfg.draft_used:
            warnings.append(BenchWarning("draft_unused", cfg.label))
        elif cfg.speedup is not None and cfg.speedup < SLOWER_THRESHOLD:
            warnings.append(BenchWarning("slower", cfg.label, {"percent": f"{cfg.speedup * 100:+.1f}"}))
        if cfg.contents != base_cfg.contents:
            warnings.append(BenchWarning("output_differs", cfg.label))
    return BenchResult(results, warnings, n_predict, repeat, len(prompts), mock)


def result_dict(result: BenchResult, lang: str) -> dict[str, Any]:
    """Language-independent JSON structure (only ``message`` follows ``lang``)."""

    def pct(x: float | None) -> float | None:
        return None if x is None else round(x * 100, 1)

    configs = []
    for c in result.configs:
        configs.append({
            "label": c.label,
            "url": None if result.mock else c.url,
            "role": c.role,
            "tokens": c.tokens,
            "predicted_ms": round(c.ms, 3),
            "tokens_per_second": round(c.tps, 2),
            "draft_n": c.draft_n,
            "draft_n_accepted": c.draft_accepted,
            "acceptance_percent": pct(c.acceptance),
            "speedup_percent": pct(c.speedup),
            "draft_used": c.draft_used if c.role == "spec" else None,
            "prompts": [
                {"index": r.prompt_index, "tokens": r.tokens, "tokens_per_second": round(r.tps, 2),
                 "draft_n": r.draft_n, "draft_n_accepted": r.draft_accepted,
                 "acceptance_percent": pct(r.acceptance)} for r in c.rows
            ],
        })
    return {
        "schema_version": 1,
        "mock": result.mock,
        "n_predict": result.n_predict,
        "repeat": result.repeat,
        "n_prompts": result.n_prompts,
        "configs": configs,
        "warnings": [{"code": w.code, "label": w.label, "message": w.message(lang)} for w in result.warnings],
        "exit_code": result.exit_code,
    }
