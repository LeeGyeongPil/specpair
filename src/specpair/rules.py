"""Apply a runtime's draft/target compatibility rules and collect *every* failing rule."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .ruleset import DEFAULT_RUNTIME, load_ruleset
from .vocab import Vocab

RULE_IDS = ("vocab_type", "bos", "eos", "vocab_size", "token_text")
_PAD_RE = re.compile(r"^\[PAD\d*\]$")


def show_token(text: str) -> str:
    """Quote a token for display; undecodable bytes are shown as escapes."""
    safe = text.encode("utf-8", errors="backslashreplace").decode("utf-8")
    return repr(safe)


def _display_params(params: dict[str, Any], lang: str = "en") -> dict[str, Any]:
    from .i18n import t

    out = {}
    for key, value in params.items():
        if value is None:
            out[key] = "-"
        elif key in ("side", "longer") and value in ("target", "draft"):
            out[key] = t(lang, f"word.{value}")
        elif isinstance(value, bool):
            out[key] = t(lang, "word.yes" if value else "word.no")
        else:
            out[key] = value
    for key in ("target_token", "draft_token"):
        if key in out and isinstance(out[key], str):
            out[key] = show_token(out[key])
    if "examples" in out:
        out["examples_text"] = "; ".join(
            f"id {e['id']}: {show_token(e['target'])} vs {show_token(e['draft'])}" for e in out["examples"]
        )
    return out


@dataclass(frozen=True)
class RuleFailure:
    """One failed rule. ``evidence`` holds language-independent facts (ids, sizes, token strings)."""

    rule: str
    evidence: dict[str, Any]

    def message(self, lang: str = "en") -> str:
        from .i18n import t

        return t(lang, f"fail.{self.rule}", **_display_params(self.evidence, lang))

    def short(self, lang: str = "en") -> str:
        """Compact one-line form used in scan tables."""
        from .i18n import t

        return t(lang, f"short.{self.rule}", **_display_params(self.evidence, lang))


@dataclass(frozen=True)
class Note:
    """Informational finding that does not change the verdict."""

    code: str
    params: dict[str, Any] = field(default_factory=dict)

    def message(self, lang: str = "en") -> str:
        from .i18n import t

        return t(lang, f"note.{self.code}", **_display_params(self.params, lang))


@dataclass
class CheckResult:
    """Outcome of :func:`check_pair`."""

    runtime: str
    verdict: str  # "compatible" | "rejected"
    failures: list[RuleFailure]
    notes: list[Note]
    same_id_ratio: float
    n_vocab: tuple[int, int]
    target: Vocab = field(repr=False, default=None)  # type: ignore[assignment]
    draft: Vocab = field(repr=False, default=None)  # type: ignore[assignment]

    @property
    def compatible(self) -> bool:
        return self.verdict == "compatible"


def _first_mismatches(t: Vocab, d: Vocab, start: int, limit: int = 3) -> tuple[int, list[dict[str, Any]]]:
    end = min(t.n_vocab, d.n_vocab)
    count = 0
    examples: list[dict[str, Any]] = []
    for i in range(start, end):
        if t.tokens[i] != d.tokens[i]:
            count += 1
            if len(examples) < limit:
                examples.append({"id": i, "target": t.tokens[i], "draft": d.tokens[i]})
    return count, examples


def check_pair(target: Vocab, draft: Vocab, runtime: str = DEFAULT_RUNTIME) -> CheckResult:
    """Check whether ``draft`` may speculate for ``target`` under ``runtime``'s rules.

    Unlike the runtime, which stops at the first failing rule, all five rules are evaluated.
    """
    rules = load_ruleset(runtime)
    max_diff = rules["max_size_diff"]
    start = rules["check_start_id"]
    failures: list[RuleFailure] = []
    notes: list[Note] = []

    # 1) vocabulary type
    if target.vocab_type != draft.vocab_type:
        failures.append(RuleFailure("vocab_type", {
            "target_type": target.vocab_type, "draft_type": draft.vocab_type,
            "target_model": target.model, "draft_model": draft.model,
            "target_pre": target.pre, "draft_pre": draft.pre,
        }))

    # 2) and 3) BOS / EOS: ids only matter while the token is added
    for rule, attr_add, attr_id, attr_explicit in (
        ("bos", "add_bos", "bos_id", "add_bos_explicit"),
        ("eos", "add_eos", "eos_id", "add_eos_explicit"),
    ):
        t_add, d_add = getattr(target, attr_add), getattr(draft, attr_add)
        t_id, d_id = getattr(target, attr_id), getattr(draft, attr_id)
        if t_add != d_add or (t_add and t_id != d_id):
            failures.append(RuleFailure(rule, {
                "target_add": t_add, "draft_add": d_add, "target_id": t_id, "draft_id": d_id,
            }))
            for side, vocab in (("target", target), ("draft", draft)):
                if not getattr(vocab, attr_explicit):
                    notes.append(Note(f"implicit_{rule}", {
                        "side": side, "value": getattr(vocab, attr_add),
                        "vocab_type": vocab.vocab_type, "pre": vocab.pre,
                    }))
        elif not t_add and t_id != d_id and t_id is not None and d_id is not None:
            notes.append(Note(f"{rule}_id_differs", {"target_id": t_id, "draft_id": d_id}))

    # 4) vocabulary size
    diff = abs(target.n_vocab - draft.n_vocab)
    if diff > max_diff:
        failures.append(RuleFailure("vocab_size", {
            "target": target.n_vocab, "draft": draft.n_vocab, "diff": diff, "max": max_diff,
        }))
    elif diff == max_diff:
        notes.append(Note("size_boundary", {"diff": diff, "max": max_diff}))
    elif diff > 0:
        notes.append(Note("size_diff_allowed", {"diff": diff, "max": max_diff, "unchecked_from": min(target.n_vocab, draft.n_vocab)}))

    # 5) token strings from id `start` up to the shorter vocabulary
    count, examples = _first_mismatches(target, draft, start)
    if count:
        first = examples[0]
        failures.append(RuleFailure("token_text", {
            "start_id": start, "count": count, "first_id": first["id"],
            "target_token": first["target"], "draft_token": first["draft"], "examples": examples,
        }))

    # --- notes that never change the verdict ---------------------------------------------------
    if failures and target.vocab_type == draft.vocab_type:
        notes.append(Note("misleading_vocab_type", {"vocab_type": target.vocab_type}))
    if (target.chat_template or "") != (draft.chat_template or ""):
        notes.append(Note("template_differs"))
    shorter, longer = sorted((target, draft), key=lambda v: v.n_vocab)
    extra = longer.tokens[shorter.n_vocab:]
    if extra and all(_PAD_RE.match(tok) for tok in extra):
        notes.append(Note("gguf_padding", {"extra": len(extra), "longer": "target" if longer is target else "draft"}))
    if "hf" in (target.source_kind, draft.source_kind):
        notes.append(Note("hf_input"))
    for side, vocab in (("target", target), ("draft", draft)):
        arch = (vocab.arch or "").lower()
        if any(h in arch for h in rules["dedicated_head_arch_hints"]):
            notes.append(Note("dedicated_head", {"side": side, "arch": vocab.arch}))

    same = sum(1 for a, b in zip(target.tokens, draft.tokens) if a == b)
    ratio = round(same / max(target.n_vocab, draft.n_vocab, 1), 6)
    return CheckResult(
        runtime=runtime,
        verdict="rejected" if failures else "compatible",
        failures=failures,
        notes=notes,
        same_id_ratio=ratio,
        n_vocab=(target.n_vocab, draft.n_vocab),
        target=target,
        draft=draft,
    )


def recommended_command(result: CheckResult, target_path: str, draft_path: str) -> str | None:
    """Current-flag llama-server command for a compatible GGUF pair, otherwise ``None``."""
    import shlex

    if not result.compatible or result.target.source_kind != "gguf" or result.draft.source_kind != "gguf":
        return None
    rec = load_ruleset(result.runtime)["recommended"]
    return (
        f"llama-server -m {shlex.quote(target_path)} -md {shlex.quote(draft_path)} "
        f"--spec-type {rec['spec_type']} --spec-draft-n-max {rec['spec_draft_n_max']}"
    )
