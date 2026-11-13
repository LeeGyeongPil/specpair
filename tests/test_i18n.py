"""AC-8: --lang / SPECPAIR_LANG / locale precedence, identical en/ko message keys, visible switching."""

from __future__ import annotations

import re
import string

import pytest

from specpair.i18n import MESSAGES, normalize_lang, resolve_lang, t


def placeholders(text):
    return {name for _, name, _, _ in string.Formatter().parse(text) if name}


def test_en_and_ko_have_identical_keys():
    assert set(MESSAGES["en"]) == set(MESSAGES["ko"])


def test_en_and_ko_use_the_same_placeholders():
    for key in MESSAGES["en"]:
        assert placeholders(MESSAGES["en"][key]) == placeholders(MESSAGES["ko"][key]), key


def test_no_message_is_empty():
    for lang, table in MESSAGES.items():
        assert all(text.strip() for text in table.values()), lang


def test_korean_messages_contain_hangul_where_english_has_prose():
    hangul = re.compile("[가-힣]")
    assert hangul.search(MESSAGES["ko"]["check.verdict_bad"])
    assert not hangul.search(MESSAGES["en"]["check.verdict_bad"])


@pytest.mark.parametrize(
    ("cli_value", "env", "expected"),
    [
        (None, {}, "en"),
        (None, {"LANG": "C.UTF-8"}, "en"),
        (None, {"LANG": "en_US.UTF-8"}, "en"),
        (None, {"LANG": "ko_KR.UTF-8"}, "ko"),
        (None, {"LC_MESSAGES": "ko_KR.UTF-8", "LANG": "en_US.UTF-8"}, "ko"),
        (None, {"LC_ALL": "en_US.UTF-8", "LC_MESSAGES": "ko_KR.UTF-8", "LANG": "ko_KR.UTF-8"}, "en"),
        (None, {"SPECPAIR_LANG": "ko", "LC_ALL": "en_US.UTF-8"}, "ko"),
        (None, {"SPECPAIR_LANG": "en", "LANG": "ko_KR.UTF-8"}, "en"),
        ("en", {"SPECPAIR_LANG": "ko", "LANG": "ko_KR.UTF-8"}, "en"),
        ("ko", {"SPECPAIR_LANG": "en", "LANG": "C"}, "ko"),
        ("ko-KR", {}, "ko"),
        ("ko_KR", {}, "ko"),
        ("KO", {}, "ko"),
        ("ko_KR.UTF-8", {}, "ko"),
        ("en-US", {"LANG": "ko_KR.UTF-8"}, "en"),
        ("", {"LANG": "ko_KR.UTF-8"}, "ko"),
    ],
)
def test_resolve_lang_precedence(cli_value, env, expected):
    assert resolve_lang(cli_value, env) == expected


@pytest.mark.parametrize("bad", ["fr", "korean", "k"])
def test_resolve_lang_rejects_unknown_explicit_values(bad):
    with pytest.raises(ValueError):
        resolve_lang(bad, {})


def test_normalize_lang():
    assert normalize_lang("ko-KR") == "ko" and normalize_lang("zh") is None


def test_unknown_language_falls_back_to_english_text():
    assert t("fr", "bench.baseline") == "baseline"


def test_help_switches_language(cli):
    _, en_out, _ = cli("--lang", "en", "--help")
    _, ko_out, _ = cli("--lang", "ko", "--help")
    assert "Check whether a draft model" in en_out and "check whether" in en_out
    assert "드래프트 모델이" in ko_out and "Check whether a draft model" not in ko_out


def test_subcommand_help_switches_language(cli):
    _, ko_out, _ = cli("check", "--lang", "ko", "--help")
    assert "대상 모델" in ko_out
    _, en_out, _ = cli("check", "--lang", "en", "--help")
    assert "target model" in en_out


def test_check_output_switches_language(cli, demo_dir):
    args = ("check", str(demo_dir / "target.gguf"), str(demo_dir / "draft-wide.gguf"))
    _, en_out, _ = cli("--lang", "en", *args)
    _, ko_out, _ = cli("--lang", "ko", *args)
    assert "vocabulary sizes differ by 129" in en_out
    assert "어휘 크기 차이가 129" in ko_out
    assert "verdict: rejected" in en_out and "판정: rejected" in ko_out


def test_environment_and_flag_precedence_in_the_cli(cli, demo_dir, monkeypatch):
    args = ("check", str(demo_dir / "target.gguf"), str(demo_dir / "draft-wide.gguf"))
    monkeypatch.setenv("SPECPAIR_LANG", "ko")
    assert "판정" in cli(*args)[1]
    assert "verdict:" in cli("--lang", "en", *args)[1]
    monkeypatch.delenv("SPECPAIR_LANG")
    monkeypatch.setenv("LC_ALL", "ko_KR.UTF-8")
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    assert "판정" in cli(*args)[1]
    assert "판정" in cli("--lang", "ko-KR", *args)[1]


def test_unsupported_lang_flag_is_exit_2(cli, demo_dir):
    code, out, err = cli("--lang", "fr", "check", "a", "b")
    assert code == 2 and out == "" and "unsupported language 'fr'" in err


def test_scan_bench_and_demo_follow_the_language(cli, demo_dir):
    _, scan_out, _ = cli("--lang", "ko", "scan", str(demo_dir / "target.gguf"), str(demo_dir))
    assert "상태" in scan_out
    _, bench_out, _ = cli("--lang", "ko", "bench", "--mock")
    assert bench_out.splitlines()[0].startswith("연습용 목(mock)")
    assert "수락률" in bench_out and "slower than baseline" in bench_out


@pytest.mark.parametrize("argv", [["--help"], ["check", "--help"], ["scan", "--help"], ["bench", "--help"],
                                  ["demo", "--help"]])
def test_korean_help_has_no_english_argparse_labels(cli, argv):
    _, out, _ = cli("--lang", "ko", *argv)
    for label in ("usage:", "options:", "positional arguments:", "show this help message"):
        assert label not in out
    assert "사용법:" in out and "옵션:" in out and "이 도움말을 보여 주고 끝냅니다" in out


def test_english_help_keeps_argparse_labels(cli):
    _, out, _ = cli("--lang", "en", "check", "--help")
    assert "usage:" in out and "options:" in out and "positional arguments:" in out
    assert "show this help message and exit" in out


def test_korean_usage_error_uses_korean_usage_prefix(cli):
    code, _, err = cli("--lang", "ko", "check")
    assert code == 2 and "사용법:" in err and "usage:" not in err
