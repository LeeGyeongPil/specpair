"""AC-4: bad input ends on defined error paths: exit code 2, a cause message, no traceback."""

from __future__ import annotations

import json

import pytest

from conftest import clean_env, run_module
from specpair import (
    InputNotFoundError,
    MissingTokenizerError,
    NotGGUFError,
    TokenizerJSONError,
    TruncatedHeaderError,
    load_vocab,
)


def make_hf(folder, content: bytes):
    folder.mkdir()
    (folder / "tokenizer.json").write_bytes(content)
    return folder


@pytest.fixture
def cases(tmp_path, demo_dir, mk):
    """name -> (args, message fragment)."""
    target = str(demo_dir / "target.gguf")
    not_gguf = tmp_path / "not-gguf.gguf"
    not_gguf.write_bytes(b"this is plain text, not a model file at all")
    good_json = json.dumps({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}, "added_tokens": []}).encode()
    truncated = make_hf(tmp_path / "truncated", good_json[: len(good_json) // 2])
    non_utf8 = make_hf(tmp_path / "non-utf8", b'{"model": {"type": "BPE", "vocab": {"\xff\xfe": 0}}}')
    no_model = make_hf(tmp_path / "no-model", b'{"added_tokens": []}')
    split = tmp_path / "qwen-00002-of-00002.gguf"
    split.write_bytes(mk.build_gguf([("general.architecture", "str", "qwen2")]))
    return {
        "missing path": (["check", "/home/me/models/missing.gguf", target], "does not exist"),
        "not gguf": (["check", target, str(not_gguf)], "not a GGUF file"),
        "header truncated": (["check", target, str(demo_dir / "broken.gguf")], "tokenizer.ggml.tokens"),
        "truncated json": (["check", target, str(truncated)], "truncated"),
        "non utf-8 json": (["check", target, str(non_utf8)], "not valid UTF-8"),
        "missing model key": (["check", target, str(no_model)], "required key 'model'"),
        "later split part": (["check", target, str(split)], "qwen-00001-of-00002.gguf"),
        "unsupported runtime": (["check", target, target, "--runtime", "vllm@0.0"], "unsupported runtime"),
        "scan missing folder": (["scan", target, "/home/me/models/nowhere"], "is not a folder"),
    }


CASE_NAMES = ["missing path", "not gguf", "header truncated", "truncated json", "non utf-8 json",
              "missing model key", "later split part", "unsupported runtime", "scan missing folder"]


@pytest.mark.parametrize("name", CASE_NAMES)
def test_defined_error_path(cases, name):
    args, fragment = cases[name]
    proc = run_module(*args)
    assert proc.returncode == 2, proc.stderr
    assert fragment in proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""


@pytest.mark.parametrize("name", CASE_NAMES)
def test_defined_error_path_in_korean(cases, name):
    args, _ = cases[name]
    proc = run_module("--lang", "ko", *args)
    assert proc.returncode == 2
    assert "specpair: 오류:" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_truncation_reports_key_and_byte_offset(demo_dir):
    with pytest.raises(TruncatedHeaderError) as info:
        load_vocab(demo_dir / "broken.gguf")
    assert info.value.params["key"] == "tokenizer.ggml.tokens"
    assert isinstance(info.value.params["offset"], int) and info.value.params["offset"] > 0
    assert str(info.value.params["offset"]) in str(info.value)


def test_library_exception_types(tmp_path, mk):
    with pytest.raises(InputNotFoundError):
        load_vocab("/home/me/models/missing.gguf")
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"GGUX" + b"\0" * 40)
    with pytest.raises(NotGGUFError):
        load_vocab(bad)
    empty = tmp_path / "empty.gguf"
    empty.write_bytes(b"")
    with pytest.raises(TruncatedHeaderError):
        load_vocab(empty)
    nokeys = tmp_path / "nokeys.gguf"
    nokeys.write_bytes(mk.build_gguf([("general.architecture", "str", "llama")]))
    with pytest.raises(MissingTokenizerError):
        load_vocab(nokeys)
    folder = make_hf(tmp_path / "f", b"[1, 2]")
    with pytest.raises(TokenizerJSONError):
        load_vocab(folder)


def test_unsupported_gguf_version(tmp_path, mk):
    path = tmp_path / "v9.gguf"
    path.write_bytes(mk.build_gguf([], version=9))
    proc = run_module("check", str(path), str(path))
    assert proc.returncode == 2 and "GGUF version 9" in proc.stderr and "Traceback" not in proc.stderr


def test_corrupt_value_type_and_huge_counts_do_not_crash(tmp_path):
    import struct

    huge = b"GGUF" + struct.pack("<IQQ", 3, 0, 2**62) + b"\x01" * 16
    path = tmp_path / "huge.gguf"
    path.write_bytes(huge)
    proc = run_module("check", str(path), str(path))
    assert proc.returncode == 2 and "Traceback" not in proc.stderr


def test_folder_without_tokenizer_json(tmp_path, demo_dir):
    (tmp_path / "empty-folder").mkdir()
    proc = run_module("check", str(demo_dir / "target.gguf"), str(tmp_path / "empty-folder"))
    assert proc.returncode == 2 and "no tokenizer.json" in proc.stderr


def test_error_text_never_depends_on_traceback_even_with_ascii_locale(demo_dir):
    env = clean_env()
    env["LANG"] = "C"
    env["PYTHONIOENCODING"] = "ascii"
    proc = run_module("--lang", "ko", "check", "/home/me/x.gguf", str(demo_dir / "target.gguf"), env=env)
    assert proc.returncode == 2 and "Traceback" not in proc.stderr


def test_empty_and_tiny_files_get_specific_messages(tmp_path, demo_dir):
    target = str(demo_dir / "target.gguf")
    empty = tmp_path / "empty.gguf"
    empty.write_bytes(b"")
    tiny = tmp_path / "tiny.gguf"
    tiny.write_bytes(b"GGUF\x03\x00\x00\x00")
    for lang, word in (("en", "is empty"), ("ko", "비어 있습니다")):
        proc = run_module("--lang", lang, "check", target, str(empty))
        err = proc.stderr
        assert proc.returncode == 2 and word in err and "(header)" not in err
    proc = run_module("check", target, str(tiny))
    err = proc.stderr
    assert proc.returncode == 2 and "fixed GGUF header at byte" in err and "(header)" not in err and "Traceback" not in err
