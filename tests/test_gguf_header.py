"""AC-2: the stdlib header parser agrees with the official gguf package and applies llama.cpp defaults."""

from __future__ import annotations

import numpy as np
import pytest
from gguf import GGUFReader, GGUFWriter

from specpair import load_vocab
from specpair.gguf_header import read_header

TOKENS = ["<s>", "</s>", "<unk>", "a", "b", "Ġthe", "café", "你好", "<0xE5>", "x"] * 20


def write_official(path, *, model="gpt2", pre="qwen2", add_bos=None, with_tensor=True):
    writer = GGUFWriter(str(path), "qwen2")
    writer.add_tokenizer_model(model)
    if pre is not None:
        writer.add_tokenizer_pre(pre)
    writer.add_token_list(TOKENS)
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(1)
    if add_bos is not None:
        writer.add_add_bos_token(add_bos)
    if with_tensor:
        writer.add_tensor("w", np.zeros((4, 4), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    if with_tensor:
        writer.write_tensors_to_file()
    writer.close()


def official_str(reader, key):
    return reader.fields[key].contents()


def test_matches_official_reader(tmp_path):
    path = tmp_path / "official.gguf"
    write_official(path)
    reader = GGUFReader(str(path))
    vocab = load_vocab(path)
    assert vocab.tokens == official_str(reader, "tokenizer.ggml.tokens")
    assert vocab.bos_id == official_str(reader, "tokenizer.ggml.bos_token_id") == 0
    assert vocab.eos_id == official_str(reader, "tokenizer.ggml.eos_token_id") == 1
    assert vocab.model == official_str(reader, "tokenizer.ggml.model") == "gpt2"
    assert vocab.pre == official_str(reader, "tokenizer.ggml.pre") == "qwen2"
    assert vocab.vocab_type == "BPE"
    assert vocab.arch == "qwen2"


def test_header_only_file_with_cut_off_tensor_data(tmp_path):
    full = tmp_path / "full.gguf"
    write_official(full)
    expected = load_vocab(full)
    header = read_header(full)
    assert header.n_tensors == 1
    cut = tmp_path / "cut.gguf"
    cut.write_bytes(full.read_bytes()[: header.header_bytes])  # tensor info and data are gone
    cut_vocab = load_vocab(cut)
    assert cut_vocab.tokens == expected.tokens
    assert cut_vocab.header_bytes == header.header_bytes


def test_large_non_token_arrays_are_skipped(write_gguf, mk):
    path = write_gguf(mk.tokenizer_kvs(mk.base_tokens()))
    header = read_header(path)
    assert header.array_lengths["tokenizer.ggml.token_type"] == 1000
    assert header.array_lengths["tokenizer.ggml.merges"] == 2
    assert "tokenizer.ggml.merges" not in header.kv


@pytest.mark.parametrize(
    ("model", "pre", "expected_bos", "expected_eos", "vocab_type"),
    [
        ("gpt2", "llama-bpe", True, False, "BPE"),
        ("gpt2", "llama3", True, False, "BPE"),
        ("gpt2", "qwen2", False, False, "BPE"),
        ("gpt2", None, False, False, "BPE"),
        ("llama", None, True, False, "SPM"),
        ("bert", None, True, False, "WPM"),
        ("t5", None, False, True, "UGM"),
        ("rwkv", None, False, False, "RWKV"),
    ],
)
def test_implicit_add_bos_add_eos_defaults(write_gguf, mk, model, pre, expected_bos, expected_eos, vocab_type):
    path = write_gguf(mk.tokenizer_kvs(mk.base_tokens(), model=model, pre=pre))
    vocab = load_vocab(path)
    assert vocab.vocab_type == vocab_type
    assert (vocab.add_bos, vocab.add_eos) == (expected_bos, expected_eos)
    assert not vocab.add_bos_explicit and not vocab.add_eos_explicit


def test_explicit_keys_override_implicit_defaults(write_gguf, mk):
    path = write_gguf(mk.tokenizer_kvs(mk.base_tokens(), model="llama", pre=None, add_bos=False, add_eos=True))
    vocab = load_vocab(path)
    assert (vocab.add_bos, vocab.add_eos) == (False, True)
    assert vocab.add_bos_explicit and vocab.add_eos_explicit


def test_missing_special_ids_use_llama_cpp_defaults(write_gguf, mk):
    spm = load_vocab(write_gguf(mk.tokenizer_kvs(mk.base_tokens(), model="llama", pre=None, bos=None, eos=None)))
    assert (spm.bos_id, spm.eos_id) == (1, 2)
    bpe = load_vocab(write_gguf(mk.tokenizer_kvs(mk.base_tokens(), bos=None, eos=None)))
    assert (bpe.bos_id, bpe.eos_id) == (11, 11)


def test_gguf_v2_is_supported(write_gguf, mk):
    path = write_gguf(mk.tokenizer_kvs(mk.base_tokens()), version=2)
    assert load_vocab(path).n_vocab == 1000


def test_utf8_tokens_round_trip(write_gguf, mk):
    tokens = mk.base_tokens()
    tokens[10] = "café 你好"
    assert load_vocab(write_gguf(mk.tokenizer_kvs(tokens))).tokens[10] == "café 你好"
