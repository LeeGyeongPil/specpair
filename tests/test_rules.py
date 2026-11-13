"""AC-1: llama.cpp@b12360 verdicts at the boundaries and in combination."""

from __future__ import annotations

import pytest

from specpair import check_pair, load_vocab


def rules_of(result):
    return [f.rule for f in result.failures]


def note_codes(result):
    return [n.code for n in result.notes]


def check_demo(demo_dir, draft):
    return check_pair(load_vocab(demo_dir / "target.gguf"), load_vocab(demo_dir / draft))


def test_boundary_128_is_compatible_with_note(demo_dir):
    result = check_demo(demo_dir, "draft-ok.gguf")
    assert result.verdict == "compatible"
    assert result.failures == []
    assert "size_boundary" in note_codes(result)
    assert result.n_vocab == (1000, 1128)


def test_129_is_rejected_with_only_the_size_rule(demo_dir):
    result = check_demo(demo_dir, "draft-wide.gguf")
    assert result.verdict == "rejected"
    assert rules_of(result) == ["vocab_size"]
    assert result.failures[0].evidence == {"target": 1000, "draft": 1129, "diff": 129, "max": 128}


def test_other_draft_reports_all_three_failures_with_evidence(demo_dir):
    result = check_demo(demo_dir, "draft-other.gguf")
    assert result.verdict == "rejected"
    assert rules_of(result) == ["bos", "vocab_size", "token_text"]
    bos, size, text = result.failures
    assert bos.evidence["target_add"] is False and bos.evidence["draft_add"] is True
    assert (bos.evidence["target_id"], bos.evidence["draft_id"]) == (0, 7)
    assert size.evidence["diff"] == 200
    assert text.evidence["first_id"] == 5
    assert text.evidence["target_token"] == "Ġw5" and text.evidence["draft_token"] == "x5"
    assert text.evidence["count"] == 20
    assert [e["id"] for e in text.evidence["examples"]] == [5, 6, 7]
    # both strings appear in the human-readable message
    message = text.message("en")
    assert "'Ġw5'" in message and "'x5'" in message
    # the "vocab type must match" wording is flagged as misleading because both are BPE
    assert "misleading_vocab_type" in note_codes(result)


def test_spm_draft_fails_the_vocab_type_rule(demo_dir):
    result = check_demo(demo_dir, "draft-spm.gguf")
    assert rules_of(result) == ["vocab_type"]
    assert result.failures[0].evidence["target_type"] == "BPE"
    assert result.failures[0].evidence["draft_type"] == "SPM"


def test_different_special_ids_pass_when_add_flags_are_off_on_both_sides(write_gguf, mk):
    tokens = mk.base_tokens()
    target = write_gguf(mk.tokenizer_kvs(tokens, bos=0, eos=2, add_bos=False, add_eos=False))
    draft = write_gguf(mk.tokenizer_kvs(tokens, bos=3, eos=4, add_bos=False, add_eos=False))
    result = check_pair(load_vocab(target), load_vocab(draft))
    assert result.verdict == "compatible"
    codes = note_codes(result)
    assert "bos_id_differs" in codes and "eos_id_differs" in codes


def test_different_special_ids_are_rejected_when_the_token_is_added(write_gguf, mk):
    tokens = mk.base_tokens()
    target = write_gguf(mk.tokenizer_kvs(tokens, bos=0, eos=2, add_bos=True, add_eos=True))
    draft = write_gguf(mk.tokenizer_kvs(tokens, bos=3, eos=4, add_bos=True, add_eos=True))
    result = check_pair(load_vocab(target), load_vocab(draft))
    assert rules_of(result) == ["bos", "eos"]


def test_add_flag_mismatch_alone_is_a_failure(write_gguf, mk):
    tokens = mk.base_tokens()
    target = write_gguf(mk.tokenizer_kvs(tokens, add_eos=False))
    draft = write_gguf(mk.tokenizer_kvs(tokens, add_eos=True))
    assert rules_of(check_pair(load_vocab(target), load_vocab(draft))) == ["eos"]


@pytest.mark.parametrize(
    ("extra", "verdict", "rules"),
    [(0, "compatible", []), (127, "compatible", []), (128, "compatible", []), (129, "rejected", ["vocab_size"])],
)
def test_size_limit_boundaries(write_gguf, mk, extra, verdict, rules):
    base = mk.base_tokens()
    target = write_gguf(mk.tokenizer_kvs(base))
    draft = write_gguf(mk.tokenizer_kvs(mk.padded(base, extra)))
    result = check_pair(load_vocab(target), load_vocab(draft))
    assert (result.verdict, rules_of(result)) == (verdict, rules)


def test_size_is_symmetric(write_gguf, mk):
    base = mk.base_tokens()
    big = write_gguf(mk.tokenizer_kvs(mk.padded(base, 129)))
    small = write_gguf(mk.tokenizer_kvs(base))
    assert rules_of(check_pair(load_vocab(big), load_vocab(small))) == ["vocab_size"]


def test_token_strings_are_compared_from_id_5_only(write_gguf, mk):
    base = mk.base_tokens()
    changed_low = list(base)
    changed_low[4] = "<other-pad>"
    changed_high = list(base)
    changed_high[5] = "<other>"
    target = write_gguf(mk.tokenizer_kvs(base))
    ok = write_gguf(mk.tokenizer_kvs(changed_low))
    bad = write_gguf(mk.tokenizer_kvs(changed_high))
    assert check_pair(load_vocab(target), load_vocab(ok)).verdict == "compatible"
    result = check_pair(load_vocab(target), load_vocab(bad))
    assert rules_of(result) == ["token_text"]


def test_only_the_shorter_vocabulary_range_is_compared(write_gguf, mk):
    base = mk.base_tokens()
    longer = base + [f"extra{i}" for i in range(50)]
    target = write_gguf(mk.tokenizer_kvs(base))
    draft = write_gguf(mk.tokenizer_kvs(longer))
    result = check_pair(load_vocab(target), load_vocab(draft))
    assert result.verdict == "compatible"
    assert "size_diff_allowed" in note_codes(result)


def test_implicit_default_note_names_the_pre_tokenizer(demo_dir):
    result = check_demo(demo_dir, "draft-other.gguf")
    implicit = [n for n in result.notes if n.code == "implicit_bos"]
    assert {n.params["side"] for n in implicit} == {"target", "draft"}
    draft_note = next(n for n in implicit if n.params["side"] == "draft")
    assert draft_note.params["pre"] == "llama-bpe" and draft_note.params["value"] is True


def test_dedicated_draft_heads_are_flagged_not_judged(write_gguf, mk):
    base = mk.base_tokens()
    target = write_gguf(mk.tokenizer_kvs(base))
    draft = write_gguf(mk.tokenizer_kvs(base, arch="eagle3"))
    result = check_pair(load_vocab(target), load_vocab(draft))
    assert "dedicated_head" in note_codes(result)


def test_unsupported_runtime_is_rejected(demo_dir):
    from specpair import UnsupportedRuntimeError

    target = load_vocab(demo_dir / "target.gguf")
    with pytest.raises(UnsupportedRuntimeError):
        check_pair(target, target, runtime="vllm@0.0")
