"""Command line interface: argument parsing, language choice, exit codes (0 ok, 1 verdict, 2 error)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Sequence, TextIO

from . import __version__
from .bench import BenchResult, load_prompts, parse_spec, result_dict, run_bench
from .demo import run_demo, run_extract
from .errors import BenchError, SpecpairError
from .i18n import normalize_lang, resolve_lang, t
from .replay import ReplayServers
from .report import bench_text, check_json, check_text, scan_json, scan_text
from .rules import check_pair
from .ruleset import DEFAULT_RUNTIME, load_ruleset
from .scan import scan_exit_code, scan_folder
from .vocab import load_vocab


def _formatter_for(lang: str) -> Any:
    """HelpFormatter factory whose ``usage:`` prefix is in ``lang`` (English output stays as argparse's)."""
    prefix = t(lang, "help.usage") + " "

    class _Formatter(argparse.HelpFormatter):
        def add_usage(self, usage: Any, actions: Any, groups: Any, prefix_: Any = None) -> None:
            super().add_usage(usage, actions, groups, prefix if prefix_ is None else prefix_)

    return _Formatter


class _Parser(argparse.ArgumentParser):
    """ArgumentParser whose usage errors use the chosen language prefix and exit with code 2."""

    lang = "en"

    def error(self, message: str) -> "Any":  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: {t(self.lang, 'cli.error_prefix')}: {message}\n")


def _find_lang_arg(argv: Sequence[str]) -> str | None:
    """Find ``--lang X`` / ``--lang=X`` anywhere so help texts can already be localised."""
    for i, arg in enumerate(argv):
        if arg == "--lang" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--lang="):
            return arg.split("=", 1)[1]
    return None


def build_parser(lang: str) -> argparse.ArgumentParser:
    """Build the argparse tree with help text in ``lang``."""
    def h(key: str) -> str:
        return t(lang, f"help.{key}")

    lang_kwargs: dict[str, Any] = {"metavar": "en|ko", "help": h("lang")}
    fmt = _formatter_for(lang)

    def localise(p: argparse.ArgumentParser) -> None:
        """Replace argparse's built-in English labels (-h text and group titles) with ``lang`` text."""
        p.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help=h("help"))
        p._positionals.title = h("positional")
        p._optionals.title = h("options")

    parser = _Parser(prog="specpair", description=h("description"), epilog=h("epilog"), add_help=False,
                     formatter_class=fmt)
    parser.lang = lang
    localise(parser)
    parser.add_argument("--lang", default=None, **lang_kwargs)
    parser.add_argument("--version", action="version", version=f"specpair {__version__}", help=h("version"))
    sub = parser.add_subparsers(dest="command", metavar="{check,scan,bench,demo}", title=h("commands"),
                                parser_class=_Parser)

    def add(name: str, help_key: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=h(help_key), description=h(help_key), add_help=False,
                           formatter_class=fmt)
        p.lang = lang  # type: ignore[attr-defined]
        localise(p)
        p.add_argument("--lang", default=argparse.SUPPRESS, **lang_kwargs)
        return p

    check = add("check", "check")
    check.add_argument("target", help=h("check.target"))
    check.add_argument("draft", help=h("check.draft"))
    check.add_argument("--runtime", default=DEFAULT_RUNTIME, help=h("runtime"))
    check.add_argument("--json", action="store_true", help=h("json"))

    scan = add("scan", "scan")
    scan.add_argument("target", help=h("scan.target"))
    scan.add_argument("dir", help=h("scan.dir"))
    scan.add_argument("--runtime", default=DEFAULT_RUNTIME, help=h("runtime"))
    scan.add_argument("--json", action="store_true", help=h("json"))

    bench = add("bench", "bench")
    bench.add_argument("--base", metavar="URL", help=h("bench.base"))
    bench.add_argument("--spec", metavar="[LABEL=]URL", action="append", default=[], help=h("bench.spec"))
    bench.add_argument("--prompts", metavar="FILE", help=h("bench.prompts"))
    bench.add_argument("--n-predict", type=int, default=128, metavar="N", help=h("bench.n_predict"))
    bench.add_argument("--repeat", type=int, default=1, metavar="N", help=h("bench.repeat"))
    bench.add_argument("--timeout", type=float, default=120.0, metavar="SEC", help=h("bench.timeout"))
    bench.add_argument("--mock", action="store_true", help=h("bench.mock"))
    bench.add_argument("--json", action="store_true", help=h("json"))

    demo = add("demo", "demo")
    demo.add_argument("--extract", metavar="DIR", help=h("demo.extract"))
    return parser


def _emit_json(data: dict[str, Any], out: TextIO) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False), file=out)


def _cmd_check(args: argparse.Namespace, lang: str, out: TextIO) -> int:
    load_ruleset(args.runtime)  # unsupported runtime fails before any file is read
    target = load_vocab(args.target, args.runtime)
    draft = load_vocab(args.draft, args.runtime)
    result = check_pair(target, draft, args.runtime)
    if args.json:
        _emit_json(check_json(result, args.target, args.draft, lang), out)
    else:
        print(check_text(result, args.target, args.draft, lang), file=out)
    return 0 if result.compatible else 1


def _cmd_scan(args: argparse.Namespace, lang: str, out: TextIO) -> int:
    load_ruleset(args.runtime)
    target = load_vocab(args.target, args.runtime)
    rows = scan_folder(target, args.target, args.dir, args.runtime)
    if args.json:
        _emit_json(scan_json(rows, args.target, args.dir, args.runtime, lang), out)
    else:
        print(scan_text(rows, args.target, args.dir, args.runtime, lang), file=out)
    return scan_exit_code(rows)


def _cmd_bench(args: argparse.Namespace, lang: str, out: TextIO) -> int:
    result: BenchResult
    if args.mock:
        if args.base or args.spec or args.prompts:
            raise BenchError("mock_conflict")
        with ReplayServers() as servers:
            result = run_bench(servers.base, servers.specs, prompts=servers.prompts, n_predict=args.n_predict,
                               repeat=args.repeat, timeout=args.timeout, mock=True)
    else:
        if not args.base or not args.spec:
            raise BenchError("bench_args")
        specs = [parse_spec(s, i) for i, s in enumerate(args.spec, start=1)]
        prompts = load_prompts(args.prompts) if args.prompts else None
        result = run_bench(("baseline", args.base), specs, prompts=prompts, n_predict=args.n_predict,
                           repeat=args.repeat, timeout=args.timeout)
    if args.json:
        _emit_json(result_dict(result, lang), out)
    else:
        print(bench_text(result, lang), file=out)
    return result.exit_code


def _cmd_demo(args: argparse.Namespace, lang: str, out: TextIO) -> int:
    if args.extract:
        return run_extract(args.extract, lang, out)
    return run_demo(lang, out)


_COMMANDS = {"check": _cmd_check, "scan": _cmd_scan, "bench": _cmd_bench, "demo": _cmd_demo}


def _make_streams_safe() -> None:
    """Never crash on characters the terminal encoding cannot show."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    _make_streams_safe()
    raw_lang = _find_lang_arg(args_list)
    try:
        lang = resolve_lang(raw_lang, os.environ)
    except ValueError:
        lang = resolve_lang(None, os.environ)
        print(f"specpair: {t(lang, 'cli.error_prefix')}: "
              f"{t(lang, 'err.lang_unsupported', value=raw_lang)}", file=sys.stderr)
        return 2
    parser = build_parser(lang)
    args = parser.parse_args(args_list)
    if getattr(args, "lang", None) is not None:
        lang = normalize_lang(args.lang) or lang
    if not args.command:
        parser.print_help()
        return 2
    try:
        return _COMMANDS[args.command](args, lang, sys.stdout)
    except SpecpairError as exc:
        print(f"specpair: {t(lang, 'cli.error_prefix')}: {exc.message(lang)}", file=sys.stderr)
        return 2
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        print(f"specpair: {t(lang, 'cli.interrupted')}", file=sys.stderr)
        return 130
    except OSError as exc:
        print(f"specpair: {t(lang, 'cli.error_prefix')}: {exc.strerror or exc}", file=sys.stderr)
        return 2
