"""Regenerate the bundled demo files deterministically (header-only synthetic GGUF files).

    python tools/make_fixtures.py            # rewrites src/specpair/demo_files
    python tools/make_fixtures.py OUT_DIR    # writes somewhere else (the test suite compares the bytes)

The same helpers (``build_gguf``) are used by the tests to write synthetic files, so no real model
is ever needed. Nothing here depends on specpair itself.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

# --- minimal GGUF v3 writer ---------------------------------------------------------------------

_SCALAR = {"u8": (0, "<B"), "i8": (1, "<b"), "u16": (2, "<H"), "i16": (3, "<h"), "u32": (4, "<I"),
           "i32": (5, "<i"), "f32": (6, "<f"), "bool": (7, "<?"), "u64": (10, "<Q"), "i64": (11, "<q"),
           "f64": (12, "<d")}


def _str(text: str | bytes) -> bytes:
    raw = text if isinstance(text, bytes) else text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(kind: str, value: Any) -> tuple[int, bytes]:
    if kind == "str":
        return 8, _str(value)
    if kind.startswith("arr_"):
        inner = kind[4:]
        if inner == "str":
            body = b"".join(_str(v) for v in value)
            return 9, struct.pack("<IQ", 8, len(value)) + body
        type_id, fmt = _SCALAR[inner]
        return 9, struct.pack("<IQ", type_id, len(value)) + b"".join(struct.pack(fmt, v) for v in value)
    type_id, fmt = _SCALAR[kind]
    return type_id, struct.pack(fmt, value)


def build_gguf(kvs: list[tuple[str, str, Any]], n_tensors: int = 0, version: int = 3) -> bytes:
    """Return the bytes of a header-only GGUF file. ``kvs`` is a list of ``(key, kind, value)``."""
    out = [b"GGUF", struct.pack("<I", version), struct.pack("<QQ", n_tensors, len(kvs))]
    for key, kind, value in kvs:
        type_id, body = _value(kind, value)
        out.append(_str(key) + struct.pack("<I", type_id) + body)
    return b"".join(out)


def tokenizer_kvs(tokens: list[str], *, model: str = "gpt2", pre: str | None = "qwen2", bos: int | None = 0,
                  eos: int | None = 2, add_bos: bool | None = None, add_eos: bool | None = None,
                  template: str | None = "{{ messages }}", arch: str = "qwen2", name: str = "synthetic",
                  extra_arrays: bool = True) -> list[tuple[str, str, Any]]:
    """Typical tokenizer key/value block (optional keys are left out when ``None``)."""
    kvs: list[tuple[str, str, Any]] = [("general.architecture", "str", arch), ("general.name", "str", name)]
    kvs.append(("tokenizer.ggml.model", "str", model))
    if pre is not None:
        kvs.append(("tokenizer.ggml.pre", "str", pre))
    kvs.append(("tokenizer.ggml.tokens", "arr_str", tokens))
    if extra_arrays:  # arrays the reader has to skip
        kvs.append(("tokenizer.ggml.token_type", "arr_i32", [1] * len(tokens)))
        kvs.append(("tokenizer.ggml.merges", "arr_str", ["a b", "ab c"]))
    if bos is not None:
        kvs.append(("tokenizer.ggml.bos_token_id", "u32", bos))
    if eos is not None:
        kvs.append(("tokenizer.ggml.eos_token_id", "u32", eos))
    if add_bos is not None:
        kvs.append(("tokenizer.ggml.add_bos_token", "bool", add_bos))
    if add_eos is not None:
        kvs.append(("tokenizer.ggml.add_eos_token", "bool", add_eos))
    if template is not None:
        kvs.append(("tokenizer.chat_template", "str", template))
    return kvs


# --- demo vocabularies --------------------------------------------------------------------------

N_BASE = 1000
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<unk>", "<pad>"]


def base_tokens(n: int = N_BASE) -> list[str]:
    """Deterministic byte-level-looking vocabulary: 5 specials, then w5, Gw6, w7, ..."""
    return SPECIALS + [("Ġw" if i % 2 else "w") + str(i) for i in range(5, n)]


def padded(tokens: list[str], extra: int) -> list[str]:
    """Append ``[PAD<i>]`` slots the way the GGUF converter pads a vocabulary to the embedding size."""
    return tokens + [f"[PAD{i}]" for i in range(len(tokens), len(tokens) + extra)]


def demo_gguf_files() -> dict[str, bytes]:
    base = base_tokens()
    other = ["x" + str(i) if 5 <= i < 25 else t for i, t in enumerate(base[:800])]
    files = {
        "target.gguf": build_gguf(tokenizer_kvs(base, name="synthetic-target", template="TARGET {{ messages }}")),
        "draft-ok.gguf": build_gguf(tokenizer_kvs(padded(base, 128), name="synthetic-draft-ok",
                                                  template="TARGET {{ messages }}")),
        "draft-wide.gguf": build_gguf(tokenizer_kvs(padded(base, 129), name="synthetic-draft-wide",
                                                    template="TARGET {{ messages }}")),
        # no add_bos key + pre llama-bpe -> llama.cpp turns BOS on; other BOS id; smaller vocab; other strings
        "draft-other.gguf": build_gguf(tokenizer_kvs(other, pre="llama-bpe", bos=7, eos=2, arch="llama",
                                                     name="synthetic-draft-other", template="OTHER {{ messages }}")),
        "draft-spm.gguf": build_gguf(tokenizer_kvs(base, model="llama", pre=None, bos=1, eos=2, add_bos=False,
                                                   add_eos=False, arch="llama", name="synthetic-draft-spm",
                                                   template="TARGET {{ messages }}")),
    }
    ok = files["draft-ok.gguf"]
    marker = _str("tokenizer.ggml.tokens")
    cut = ok.index(marker) + len(marker) + 12 + 200  # inside the token array
    files["broken.gguf"] = ok[:cut]
    return files


def hf_draft_files() -> dict[str, bytes]:
    base = base_tokens()
    vocab = {tok: i for i, tok in enumerate(base) if i >= 5}
    added = [{"id": i, "content": SPECIALS[i], "special": True} for i in range(5)]
    tokenizer = {
        "version": "1.0",
        "added_tokens": added,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False},
        "model": {"type": "BPE", "vocab": vocab, "merges": []},
    }
    config = {"bos_token": "<|endoftext|>", "eos_token": "<|im_end|>", "chat_template": "TARGET {{ messages }}"}
    return {
        "hf-draft/tokenizer.json": _json(tokenizer),
        "hf-draft/tokenizer_config.json": _json(config),
    }


# --- recorded timings (llama.cpp b12360, CPU 4 threads, Qwen2.5-1.5B-Instruct q4_0 target,
#     Qwen2.5-0.5B-Instruct q4_0 draft, 128 tokens max, temperature 0, recorded 2026-11-13) -------

PROMPTS = [
    "Write a Python function that returns the first n Fibonacci numbers as a list.",
    "List the planets of the solar system in order from the sun, one per line.",
    "Explain in three sentences what a hash map is.",
]

RECORDS = {
    "baseline": [
        {"prompt_n": 15, "prompt_ms": 78.878, "predicted_n": 128, "predicted_ms": 1972.694},
        {"prompt_n": 17, "prompt_ms": 49.108, "predicted_n": 35, "predicted_ms": 560.073},
        {"prompt_n": 11, "prompt_ms": 46.255, "predicted_n": 77, "predicted_ms": 1138.177},
    ],
    "n3": [
        {"prompt_n": 15, "prompt_ms": 140.556, "predicted_n": 128, "predicted_ms": 1694.032, "draft_n": 107, "draft_n_accepted": 91},
        {"prompt_n": 17, "prompt_ms": 64.782, "predicted_n": 35, "predicted_ms": 459.28, "draft_n": 30, "draft_n_accepted": 24},
        {"prompt_n": 11, "prompt_ms": 55.477, "predicted_n": 77, "predicted_ms": 1342.977, "draft_n": 93, "draft_n_accepted": 45},
    ],
    "n8": [
        {"prompt_n": 15, "prompt_ms": 97.307, "predicted_n": 128, "predicted_ms": 2362.558, "draft_n": 165, "draft_n_accepted": 105},
        {"prompt_n": 17, "prompt_ms": 92.902, "predicted_n": 35, "predicted_ms": 528.974, "draft_n": 40, "draft_n_accepted": 29},
        {"prompt_n": 11, "prompt_ms": 53.509, "predicted_n": 77, "predicted_ms": 2327.042, "draft_n": 192, "draft_n_accepted": 52},
    ],
}


def replay_json() -> bytes:
    data = {
        "format": 1,
        "recorded_with": "llama.cpp b12360, CPU 4 threads, Qwen2.5-1.5B-Instruct q4_0 + Qwen2.5-0.5B-Instruct q4_0 draft, "
                         "n_predict 128, temperature 0, 2026-11-13",
        "content_placeholder": "(placeholder text: replayed servers return no real model output)",
        "prompts": PROMPTS,
        "configs": [
            {"label": "baseline", "role": "base", "records": RECORDS["baseline"]},
            {"label": "n3", "role": "spec", "records": RECORDS["n3"]},
            {"label": "n8", "role": "spec", "records": RECORDS["n8"]},
            # A rejected draft leaves llama-server answering like the baseline, without draft_n
            # (observed in the recording session). The records below reuse the baseline timings.
            {"label": "rejected", "role": "spec", "derived_from": "baseline", "records": RECORDS["baseline"]},
        ],
    }
    return _json(data)


def _json(data: Any) -> bytes:
    return (json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def all_files() -> dict[str, bytes]:
    files = demo_gguf_files()
    files.update(hf_draft_files())
    files["bench_replay.json"] = replay_json()
    return files


def write_all(out_dir: Path) -> list[str]:
    names = []
    for name, data in sorted(all_files().items()):
        path = out_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        names.append(name)
    return names


def main(argv: list[str]) -> int:
    default = Path(__file__).resolve().parent.parent / "src" / "specpair" / "demo_files"
    if argv and argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv and argv[0].startswith("-"):
        print(f"unknown option {argv[0]!r}; usage: python tools/make_fixtures.py [OUT_DIR]", file=sys.stderr)
        return 2
    out = Path(argv[0]) if argv else default
    for name in write_all(out):
        print(f"wrote {out / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
