"""AC-5: scan lists every GGUF, survives corrupt files, skips later split parts, sets the exit code."""

from __future__ import annotations

import json
import shutil

from specpair import load_vocab
from specpair.scan import scan_exit_code, scan_folder


def status_by_name(rows):
    return {r.path.rsplit("/", 1)[-1]: r.status for r in rows}


def test_verdict_for_every_file_and_error_row_for_corrupt_one(demo_dir):
    target = load_vocab(demo_dir / "target.gguf")
    rows = scan_folder(target, str(demo_dir / "target.gguf"), str(demo_dir))
    assert status_by_name(rows) == {
        "broken.gguf": "error",
        "draft-ok.gguf": "compatible",
        "draft-other.gguf": "rejected",
        "draft-spm.gguf": "rejected",
        "draft-wide.gguf": "rejected",
    }  # the target itself is not listed
    broken = next(r for r in rows if r.status == "error")
    assert broken.error is not None and broken.error.code == "header_truncated"
    assert scan_exit_code(rows) == 0


def test_later_split_parts_are_skipped(demo_dir, tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    ok = demo_dir / "draft-ok.gguf"
    shutil.copy(ok, folder / "model-00001-of-00003.gguf")
    shutil.copy(ok, folder / "model-00002-of-00003.gguf")
    shutil.copy(ok, folder / "model-00003-of-00003.gguf")
    target = load_vocab(demo_dir / "target.gguf")
    rows = scan_folder(target, str(demo_dir / "target.gguf"), str(folder))
    assert [r.path.rsplit("/", 1)[-1] for r in rows] == ["model-00001-of-00003.gguf"]


def test_direct_subfolders_are_scanned_and_hidden_entries_ignored(demo_dir, tmp_path):
    folder = tmp_path / "models"
    (folder / "publisher-x").mkdir(parents=True)
    (folder / ".hidden").mkdir()
    shutil.copy(demo_dir / "draft-ok.gguf", folder / "publisher-x" / "draft.gguf")
    shutil.copy(demo_dir / "draft-ok.gguf", folder / ".hidden" / "draft.gguf")
    target = load_vocab(demo_dir / "target.gguf")
    rows = scan_folder(target, str(demo_dir / "target.gguf"), str(folder), display_dir="models")
    assert [r.path for r in rows] == ["models/publisher-x/draft.gguf"]


def test_cli_exit_0_with_compatible_draft(demo_dir, cli):
    code, out, _ = cli("scan", str(demo_dir / "target.gguf"), str(demo_dir))
    assert code == 0
    assert "compatible" in out and "rejected" in out and "error" in out
    assert "1 of 5 files can be used as a draft" in out


def test_cli_exit_1_when_nothing_fits(demo_dir, tmp_path, cli):
    folder = tmp_path / "only-bad"
    folder.mkdir()
    shutil.copy(demo_dir / "draft-wide.gguf", folder / "wide.gguf")
    shutil.copy(demo_dir / "broken.gguf", folder / "broken.gguf")
    code, out, _ = cli("scan", str(demo_dir / "target.gguf"), str(folder))
    assert code == 1
    assert "0 of 2 files" in out


def test_cli_exit_1_for_empty_folder(demo_dir, tmp_path, cli):
    (tmp_path / "empty").mkdir()
    code, out, _ = cli("scan", str(demo_dir / "target.gguf"), str(tmp_path / "empty"))
    assert code == 1 and "no .gguf files found" in out


def test_cli_missing_folder_is_exit_2(demo_dir, cli):
    code, _, err = cli("scan", str(demo_dir / "target.gguf"), "/home/me/models/none")
    assert code == 2 and "not a folder" in err


def test_json_output(demo_dir, cli):
    code, out, _ = cli("scan", str(demo_dir / "target.gguf"), str(demo_dir), "--json")
    data = json.loads(out)
    assert code == 0
    assert data["schema_version"] == 1 and data["compatible_count"] == 1
    by_status = {r["status"] for r in data["results"]}
    assert by_status == {"compatible", "rejected", "error"}
    error_row = next(r for r in data["results"] if r["status"] == "error")
    assert error_row["error"]["code"] == "header_truncated"
    wide = next(r for r in data["results"] if r["path"].endswith("draft-wide.gguf"))
    assert wide["failed_rules"] == ["vocab_size"]
