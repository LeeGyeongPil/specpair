"""`specpair demo`: an offline tour with synthetic GGUF files and replayed timings (a practice mock)."""

from __future__ import annotations

import os
import tempfile
from importlib import resources
from typing import TextIO

from .bench import run_bench
from .i18n import t
from .replay import ReplayServers
from .report import bench_text, check_text
from .rules import check_pair
from .scan import scan_folder
from .report import scan_text
from .ruleset import DEFAULT_RUNTIME
from .vocab import load_vocab

SHOWN_DIR = "demo-files"
_DEMO_FILES = ["target.gguf", "draft-ok.gguf", "draft-wide.gguf", "draft-other.gguf", "draft-spm.gguf",
               "broken.gguf", "hf-draft/tokenizer.json", "hf-draft/tokenizer_config.json", "bench_replay.json"]


def extract_files(dest: str) -> list[str]:
    """Copy the bundled synthetic files into ``dest`` (created if needed) and return their relative names."""
    root = resources.files("specpair") / "demo_files"
    names = []
    for name in _DEMO_FILES:
        node = root
        for part in name.split("/"):
            node = node / part
        out = os.path.join(dest, *name.split("/"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "wb") as fh:
            fh.write(node.read_bytes())
        names.append(name)
    return names


def run_extract(dest: str, lang: str, out: TextIO) -> int:
    names = extract_files(dest)
    print(t(lang, "mock.banner"), file=out)
    print(t(lang, "demo.extracted", n=len(names), dir=dest), file=out)
    for line in ("check {d}/target.gguf {d}/draft-other.gguf", "check {d}/target.gguf {d}/draft-ok.gguf",
                 "scan {d}/target.gguf {d}", "bench --mock"):
        print("  specpair " + line.format(d=dest), file=out)
    return 0


def run_demo(lang: str, out: TextIO, runtime: str = DEFAULT_RUNTIME) -> int:
    """Run the whole tour; always exits 0 (the rejections shown are the point of the demo)."""
    print(t(lang, "mock.banner"), file=out)
    print(t(lang, "demo.intro"), file=out)
    with tempfile.TemporaryDirectory(prefix="specpair-demo-") as tmp:
        extract_files(tmp)
        target_real = os.path.join(tmp, "target.gguf")
        target_shown = f"{SHOWN_DIR}/target.gguf"
        target = load_vocab(target_real, runtime, label=target_shown)

        for step, name in ((1, "draft-other.gguf"), (2, "draft-ok.gguf")):
            print("\n" + t(lang, f"demo.step{step}", name=name), file=out)
            shown = f"{SHOWN_DIR}/{name}"
            result = check_pair(target, load_vocab(os.path.join(tmp, name), runtime, label=shown), runtime)
            print(check_text(result, target_shown, shown, lang), file=out)

        print("\n" + t(lang, "demo.step3"), file=out)
        rows = scan_folder(target, target_real, tmp, runtime, display_dir=SHOWN_DIR)
        print(scan_text(rows, target_shown, SHOWN_DIR, runtime, lang), file=out)

    print("\n" + t(lang, "demo.step4"), file=out)
    with ReplayServers() as servers:
        result = run_bench(servers.base, servers.specs, prompts=servers.prompts, mock=True)
    print(bench_text(result, lang, banner=False), file=out)
    print("\n" + t(lang, "demo.outro"), file=out)
    return 0
