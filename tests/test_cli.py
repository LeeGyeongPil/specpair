"""AC-9: check --json has a stable, language-independent schema; recommended command uses current flags."""

from __future__ import annotations

import json

import pytest

TOP_KEYS = {"schema_version", "runtime", "verdict", "failures", "notes", "target", "draft", "same_id_ratio",
            "recommended_command"}
SUMMARY_KEYS = {"path", "source", "vocab_type", "model", "pre", "arch", "name", "n_tokens", "bos_id", "eos_id",
                "add_bos", "add_bos_explicit", "add_eos", "add_eos_explicit"}


def run_json(cli, demo_dir, draft, *extra):
    code, out, err = cli("check", str(demo_dir / "target.gguf"), str(demo_dir / draft), "--json", *extra)
    assert err == ""
    return code, json.loads(out)


def test_compatible_pair_schema_and_command(cli, demo_dir):
    code, data = run_json(cli, demo_dir, "draft-ok.gguf")
    assert code == 0
    assert set(data) == TOP_KEYS
    assert data["schema_version"] == 1
    assert data["runtime"] == "llama.cpp@b12360"
    assert data["verdict"] == "compatible" and data["failures"] == []
    assert set(data["target"]) == SUMMARY_KEYS == set(data["draft"])
    assert data["target"]["path"] == str(demo_dir / "target.gguf")  # exactly what the user typed
    command = data["recommended_command"]
    assert "--spec-type draft-simple" in command and "--spec-draft-n-max 3" in command
    assert "--draft-max" not in command
    assert command.startswith("llama-server -m ") and " -md " in command
    assert {n["code"] for n in data["notes"]} >= {"size_boundary", "gguf_padding"}


def test_rejected_pair_failures_have_rule_message_evidence(cli, demo_dir):
    code, data = run_json(cli, demo_dir, "draft-other.gguf")
    assert code == 1
    assert data["verdict"] == "rejected" and data["recommended_command"] is None
    assert [f["rule"] for f in data["failures"]] == ["bos", "vocab_size", "token_text"]
    for failure in data["failures"]:
        assert set(failure) == {"rule", "message", "evidence"}
        assert isinstance(failure["message"], str) and isinstance(failure["evidence"], dict)
    assert data["failures"][2]["evidence"]["first_id"] == 5


@pytest.mark.parametrize("draft", ["draft-ok.gguf", "draft-other.gguf", "draft-spm.gguf"])
def test_json_keys_do_not_depend_on_language(cli, demo_dir, draft):
    def shape(value):
        if isinstance(value, dict):
            return {k: shape(v) for k, v in value.items() if k != "message"}
        if isinstance(value, list):
            return [shape(v) for v in value]
        return value

    _, en = run_json(cli, demo_dir, draft, "--lang", "en")
    _, ko = run_json(cli, demo_dir, draft, "--lang", "ko")
    assert shape(en) == shape(ko)
    assert [f["message"] for f in en["failures"]] != [f["message"] for f in ko["failures"]] or not en["failures"]


def test_hf_pair_has_no_recommended_command(cli, demo_dir):
    code, data = run_json(cli, demo_dir, "hf-draft")
    assert code == 0 and data["verdict"] == "compatible"
    assert data["recommended_command"] is None
    assert data["draft"]["source"] == "hf"


def test_paths_with_spaces_are_quoted_in_the_command(cli, demo_dir, tmp_path):
    spaced = tmp_path / "my models"
    spaced.mkdir()
    (spaced / "t.gguf").write_bytes((demo_dir / "target.gguf").read_bytes())
    (spaced / "d.gguf").write_bytes((demo_dir / "draft-ok.gguf").read_bytes())
    code, out, _ = cli("check", str(spaced / "t.gguf"), str(spaced / "d.gguf"), "--json")
    assert code == 0
    assert f"'{spaced / 't.gguf'}'" in json.loads(out)["recommended_command"]


def test_text_output_for_compatible_pair(cli, demo_dir):
    code, out, _ = cli("check", str(demo_dir / "target.gguf"), str(demo_dir / "draft-ok.gguf"))
    assert code == 0
    assert "rule set llama.cpp@b12360" in out
    assert "verdict: compatible" in out
    assert "--spec-type draft-simple --spec-draft-n-max 3" in out
    assert "--draft-max were removed" in out  # explains the old flag, never recommends it


def test_text_output_for_rejected_pair_lists_every_rule(cli, demo_dir):
    code, out, _ = cli("check", str(demo_dir / "target.gguf"), str(demo_dir / "draft-other.gguf"))
    assert code == 1
    assert "verdict: rejected (3 rules failed)" in out
    for rule in ("x bos:", "x vocab_size:", "x token_text:"):
        assert rule in out
    assert "'Ġw5'" in out and "'x5'" in out
    assert "-md" not in out  # no command for a rejected pair


def test_no_command_prints_help_and_exits_2(cli):
    code, out, _ = cli()
    assert code == 2 and "check" in out and "bench" in out


def test_version_flag(cli):
    from specpair import __version__

    code, out, _ = cli("--version")
    assert code == 0 and __version__ in out


def test_usage_errors_are_exit_2_without_traceback(cli):
    code, _, err = cli("check", "only-one-argument")
    assert code == 2 and "error" in err and "Traceback" not in err


def test_public_api_round_trip(demo_dir):
    import specpair

    result = specpair.check_pair(specpair.load_vocab(demo_dir / "target.gguf"),
                                 specpair.load_vocab(demo_dir / "draft-ok.gguf"))
    assert result.compatible
    command = specpair.recommended_command(result, "t.gguf", "d.gguf")
    assert command == "llama-server -m t.gguf -md d.gguf --spec-type draft-simple --spec-draft-n-max 3"
    assert specpair.resolve_lang("ko-KR") == "ko"
