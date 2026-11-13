"""AC-3: Hugging Face tokenizer folders are normalised and comparable with GGUF files."""

from __future__ import annotations

import json
import shutil

import pytest

from specpair import check_pair, load_vocab


def write_hf(folder, mk, *, tokens=None, config=True, model_type="BPE"):
    folder.mkdir(parents=True, exist_ok=True)
    tokens = tokens or mk.base_tokens()
    vocab = {tok: i for i, tok in enumerate(tokens) if i >= 5}
    added = [{"id": i, "content": tokens[i], "special": True} for i in range(5)]
    tokenizer = {
        "added_tokens": added,
        "pre_tokenizer": {"type": "ByteLevel"},
        "model": {"type": model_type, "vocab": vocab, "merges": []},
    }
    (folder / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")
    if config:
        cfg = {"bos_token": {"content": "<|endoftext|>"}, "eos_token": "<|im_end|>", "chat_template": "T"}
        (folder / "tokenizer_config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return folder


def test_bpe_folder_is_normalised(tmp_path, mk):
    vocab = load_vocab(write_hf(tmp_path / "hf", mk))
    assert vocab.source_kind == "hf"
    assert vocab.vocab_type == "BPE"
    assert vocab.n_vocab == 1000
    assert vocab.tokens[:5] == mk.SPECIALS
    assert (vocab.bos_id, vocab.eos_id) == (0, 2)  # dict-style and string-style special tokens
    assert vocab.chat_template == "T"
    assert vocab.add_bos is False and not vocab.add_bos_explicit
    assert vocab.pad_slots == 0


def test_works_without_tokenizer_config(tmp_path, mk):
    vocab = load_vocab(write_hf(tmp_path / "hf", mk, config=False))
    assert vocab.n_vocab == 1000
    assert vocab.bos_id is None and vocab.eos_id is None
    assert vocab.chat_template is None


def test_ids_without_a_token_become_pad_slots(tmp_path, mk):
    folder = write_hf(tmp_path / "hf", mk)
    data = json.loads((folder / "tokenizer.json").read_text(encoding="utf-8"))
    data["model"]["vocab"].pop(mk.base_tokens()[10])  # leave a hole at id 10
    (folder / "tokenizer.json").write_text(json.dumps(data), encoding="utf-8")
    vocab = load_vocab(folder)
    assert vocab.tokens[10] == "[PAD10]"
    assert vocab.pad_slots == 1


def test_wordpiece_and_unigram_types(tmp_path, mk):
    wp = tmp_path / "wp"
    wp.mkdir()
    (wp / "tokenizer.json").write_text(json.dumps(
        {"model": {"type": "WordPiece", "vocab": {"[PAD]": 0, "a": 1}}}), encoding="utf-8")
    assert load_vocab(wp).vocab_type == "WPM"
    uni = tmp_path / "uni"
    uni.mkdir()
    (uni / "tokenizer.json").write_text(json.dumps(
        {"model": {"type": "Unigram", "vocab": [["<unk>", 0.0], ["a", -1.0]]}}), encoding="utf-8")
    vocab = load_vocab(uni)
    assert vocab.vocab_type == "UGM" and vocab.tokens == ["<unk>", "a"]


def test_gguf_target_versus_hf_draft_is_compatible(demo_dir):
    result = check_pair(load_vocab(demo_dir / "target.gguf"), load_vocab(demo_dir / "hf-draft"))
    assert result.verdict == "compatible"
    assert [n.code for n in result.notes].count("hf_input") == 1


def test_hf_versus_hf_comparison(tmp_path, mk):
    a = load_vocab(write_hf(tmp_path / "a", mk))
    b = load_vocab(write_hf(tmp_path / "b", mk, tokens=mk.base_tokens(800)))
    result = check_pair(a, b)
    assert [f.rule for f in result.failures] == ["vocab_size"]


def test_gguf_padding_note_when_sizes_differ_only_by_pad_slots(demo_dir):
    # the GGUF side is padded with [PAD<i>] slots (like a converted model); the HF folder is not
    result = check_pair(load_vocab(demo_dir / "hf-draft"), load_vocab(demo_dir / "draft-wide.gguf"))
    assert result.verdict == "rejected"
    codes = [n.code for n in result.notes]
    assert "gguf_padding" in codes and "hf_input" in codes
    pad = next(n for n in result.notes if n.code == "gguf_padding")
    assert pad.params == {"extra": 129, "longer": "draft"}
    assert "129 extra [PAD] slots" in pad.message("en")


def test_no_padding_note_when_extra_tokens_are_real(tmp_path, mk, write_gguf):
    base = mk.base_tokens()
    gguf = write_gguf(mk.tokenizer_kvs(base + [f"real{i}" for i in range(10)]))
    result = check_pair(load_vocab(write_hf(tmp_path / "hf", mk)), load_vocab(gguf))
    assert "gguf_padding" not in [n.code for n in result.notes]


def test_copy_of_bundled_hf_folder_loads(demo_dir, tmp_path):
    copy = tmp_path / "elsewhere"
    shutil.copytree(demo_dir / "hf-draft", copy)
    assert load_vocab(copy).n_vocab == 1000


@pytest.mark.parametrize("shown", ["hf-folder"])
def test_label_replaces_the_path_in_summaries(demo_dir, shown):
    assert load_vocab(demo_dir / "hf-draft", label=shown).path == shown
