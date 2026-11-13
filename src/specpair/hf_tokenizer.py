"""Read Hugging Face ``tokenizer.json`` (+ optional ``tokenizer_config.json``) folders."""

from __future__ import annotations

import json
import os
from typing import Any

from .errors import InputNotFoundError, TokenizerJSONError

MAX_VOCAB = 5_000_000  # sanity bound against corrupt ids


def _load_json(path: str, shown: str) -> dict[str, Any]:
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        raise InputNotFoundError("unreadable", path=shown, reason=exc.strerror or str(exc)) from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TokenizerJSONError("tok_json_utf8", path=shown, offset=exc.start) from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        # An error at the very end of the text means the file was cut off (typical for broken downloads).
        if exc.pos >= len(text.rstrip()) - 1:
            raise TokenizerJSONError("tok_json_truncated", path=shown, size=len(raw)) from None
        raise TokenizerJSONError("tok_json_invalid", path=shown, line=exc.lineno, col=exc.colno) from None
    if not isinstance(data, dict):
        raise TokenizerJSONError("tok_json_key", path=shown, key="(top-level object)")
    return data


def _token_text(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("content")
    return value if isinstance(value, str) else None


def read_hf_folder(folder: str, label: str | None = None) -> dict[str, Any]:
    """Return normalised fields: tokens, vocab_type, bos_id, eos_id, add_bos(+explicit), add_eos(+explicit),
    chat_template, pad_slots."""
    shown = label or folder
    tj_path = os.path.join(folder, "tokenizer.json")
    if not os.path.isfile(tj_path):
        raise InputNotFoundError("no_tokenizer_json", path=shown)
    tj = _load_json(tj_path, os.path.join(shown, "tokenizer.json"))
    cfg: dict[str, Any] = {}
    cfg_path = os.path.join(folder, "tokenizer_config.json")
    if os.path.isfile(cfg_path):
        cfg = _load_json(cfg_path, os.path.join(shown, "tokenizer_config.json"))

    tj_shown = os.path.join(shown, "tokenizer.json")
    model = tj.get("model")
    if not isinstance(model, dict):
        raise TokenizerJSONError("tok_json_key", path=tj_shown, key="model")
    mtype = model.get("type")
    if not isinstance(mtype, str):
        raise TokenizerJSONError("tok_json_key", path=tj_shown, key="model.type")
    raw_vocab = model.get("vocab")
    vocab: dict[str, int] = {}
    if mtype == "Unigram" and isinstance(raw_vocab, list):
        for i, entry in enumerate(raw_vocab):
            if isinstance(entry, list) and entry and isinstance(entry[0], str):
                vocab[entry[0]] = i
    elif isinstance(raw_vocab, dict):
        vocab = {k: v for k, v in raw_vocab.items() if isinstance(k, str) and isinstance(v, int)}
    else:
        raise TokenizerJSONError("tok_json_key", path=tj_shown, key="model.vocab")
    for added in tj.get("added_tokens") or []:
        text = _token_text(added)
        if text is not None and isinstance(added, dict) and isinstance(added.get("id"), int):
            vocab[text] = added["id"]
    if not vocab:
        raise TokenizerJSONError("tok_json_key", path=tj_shown, key="model.vocab")
    n = max(vocab.values()) + 1
    if n <= 0 or n > MAX_VOCAB or min(vocab.values()) < 0:
        raise TokenizerJSONError("tok_json_invalid_ids", path=tj_shown, n=n)
    slots: list[str | None] = [None] * n
    for text, ident in vocab.items():
        slots[ident] = text
    pad_slots = sum(1 for s in slots if s is None)
    tokens = [s if s is not None else f"[PAD{i}]" for i, s in enumerate(slots)]

    structure = json.dumps([tj.get("pre_tokenizer"), tj.get("decoder")])
    if mtype == "BPE":
        vocab_type = "BPE" if ("ByteLevel" in structure or not model.get("byte_fallback")) else "SPM"
    else:
        vocab_type = {"Unigram": "UGM", "WordPiece": "WPM"}.get(mtype, mtype.upper())

    def ident(key: str) -> int | None:
        text = _token_text(cfg.get(key))
        return vocab.get(text) if text is not None else None

    template = cfg.get("chat_template")
    if template is not None and not isinstance(template, str):
        template = json.dumps(template, sort_keys=True)
    return {
        "tokens": tokens,
        "vocab_type": vocab_type,
        "bos_id": ident("bos_token"),
        "eos_id": ident("eos_token"),
        "add_bos": bool(cfg["add_bos_token"]) if "add_bos_token" in cfg else vocab_type == "SPM",
        "add_bos_explicit": "add_bos_token" in cfg,
        "add_eos": bool(cfg["add_eos_token"]) if "add_eos_token" in cfg else False,
        "add_eos_explicit": "add_eos_token" in cfg,
        "chat_template": template,
        "pad_slots": pad_slots,
        "name": os.path.basename(os.path.normpath(folder)),
    }
