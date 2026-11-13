"""Normalised tokenizer view of a GGUF file or a Hugging Face tokenizer folder."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from .errors import InputNotFoundError, MissingTokenizerError
from .gguf_header import read_header, split_info
from .hf_tokenizer import read_hf_folder
from .ruleset import DEFAULT_RUNTIME, load_ruleset

K_MODEL = "tokenizer.ggml.model"
K_PRE = "tokenizer.ggml.pre"
K_TOKENS = "tokenizer.ggml.tokens"


@dataclass
class Vocab:
    """What the speculative-decoding compatibility rules need to know about one model."""

    source_kind: str  # "gguf" or "hf"
    vocab_type: str  # SPM, BPE, WPM, UGM, RWKV (or UNKNOWN(<model>))
    tokens: list[str]
    bos_id: int | None
    eos_id: int | None
    add_bos: bool
    add_eos: bool
    add_bos_explicit: bool = False
    add_eos_explicit: bool = False
    pre: str | None = None
    model: str | None = None  # raw tokenizer.ggml.model value
    chat_template: str | None = None
    arch: str | None = None
    name: str | None = None
    header_bytes: int = 0
    path: str = ""
    pad_slots: int = 0  # HF ids without a token, filled with [PAD<i>]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_vocab(self) -> int:
        return len(self.tokens)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def load_vocab(path: str | os.PathLike[str], runtime: str = DEFAULT_RUNTIME, label: str | None = None) -> Vocab:
    """Load ``path`` (a ``.gguf`` file or a folder with ``tokenizer.json``) into a :class:`Vocab`.

    ``runtime`` picks the rule set that supplies llama.cpp's implicit defaults for ``add_bos``/``add_eos``.
    ``label`` replaces the path in error messages and summaries.
    Raises a :class:`specpair.SpecpairError` subclass for every bad-input case.
    """
    rules = load_ruleset(runtime)
    shown = label or os.fspath(path)
    p = os.fspath(path)
    if os.path.isdir(p):
        hf = read_hf_folder(p, shown)
        return Vocab(source_kind="hf", path=shown, **hf)
    if not os.path.exists(p):
        raise InputNotFoundError("not_found", path=shown)
    header = read_header(p, shown)
    kv = header.kv
    model = kv.get(K_MODEL)
    tokens = kv.get(K_TOKENS)
    if not isinstance(model, str) or not isinstance(tokens, list):
        missing = K_MODEL if not isinstance(model, str) else K_TOKENS
        info = split_info(p)
        if info and info[0] > 1:
            raise MissingTokenizerError("no_tokenizer_split", path=shown, first=info[2])
        raise MissingTokenizerError("no_tokenizer", path=shown, key=missing)
    vtype = rules["vocab_types"].get(model, f"UNKNOWN({model})")
    pre = kv.get(K_PRE) if isinstance(kv.get(K_PRE), str) else None

    implicit_bos = vtype in ("SPM", "WPM") or (vtype == "BPE" and pre in rules["bpe_add_bos_pre"])
    implicit_eos = vtype == "UGM"
    bos_explicit = "tokenizer.ggml.add_bos_token" in kv
    eos_explicit = "tokenizer.ggml.add_eos_token" in kv
    add_bos = bool(kv["tokenizer.ggml.add_bos_token"]) if bos_explicit else implicit_bos
    add_eos = bool(kv["tokenizer.ggml.add_eos_token"]) if eos_explicit else implicit_eos
    if vtype == "BPE" and pre in rules["force_add_bos_pre"]:
        add_bos = True

    defaults = rules["default_special_ids"].get(vtype, {"bos": None, "eos": None})
    bos_id = _int_or_none(kv.get("tokenizer.ggml.bos_token_id"))
    eos_id = _int_or_none(kv.get("tokenizer.ggml.eos_token_id"))
    template = kv.get("tokenizer.chat_template")
    return Vocab(
        source_kind="gguf",
        vocab_type=vtype,
        tokens=[t if isinstance(t, str) else str(t) for t in tokens],
        bos_id=bos_id if bos_id is not None else defaults["bos"],
        eos_id=eos_id if eos_id is not None else defaults["eos"],
        add_bos=add_bos,
        add_eos=add_eos,
        add_bos_explicit=bos_explicit,
        add_eos_explicit=eos_explicit,
        pre=pre,
        model=model,
        chat_template=template if isinstance(template, str) else None,
        arch=kv.get("general.architecture") if isinstance(kv.get("general.architecture"), str) else None,
        name=kv.get("general.name") if isinstance(kv.get("general.name"), str) else None,
        header_bytes=header.header_bytes,
        path=shown,
    )
