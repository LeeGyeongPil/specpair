"""AC-10: bundled demo files are produced byte-for-byte by tools/make_fixtures.py."""

from __future__ import annotations

from pathlib import Path

from conftest import MK, ROOT

BUNDLED = ROOT / "src" / "specpair" / "demo_files"


def test_bundled_files_are_regenerated_byte_for_byte(tmp_path):
    names = MK.write_all(tmp_path)
    assert names, "no files generated"
    for name in names:
        assert (BUNDLED / name).read_bytes() == (tmp_path / name).read_bytes(), name


def test_no_extra_files_in_the_bundle(tmp_path):
    expected = set(MK.write_all(tmp_path))
    found = {p.relative_to(BUNDLED).as_posix() for p in BUNDLED.rglob("*") if p.is_file()}
    assert found == expected


def test_generation_is_deterministic(tmp_path):
    first = MK.all_files()
    second = MK.all_files()
    assert first == second


def test_bundled_files_are_small():
    assert sum(p.stat().st_size for p in BUNDLED.rglob("*") if p.is_file()) < 200_000


def test_fixture_sizes_hit_the_designed_boundaries(demo_dir):
    from specpair import load_vocab

    sizes = {p.name: load_vocab(p).n_vocab for p in demo_dir.glob("draft-*.gguf")}
    assert sizes == {"draft-ok.gguf": 1128, "draft-wide.gguf": 1129, "draft-other.gguf": 800,
                     "draft-spm.gguf": 1000}
    assert load_vocab(demo_dir / "target.gguf").n_vocab == 1000
    assert Path(demo_dir / "broken.gguf").stat().st_size < Path(demo_dir / "draft-ok.gguf").stat().st_size
