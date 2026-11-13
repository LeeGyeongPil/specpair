"""specpair - check draft/target model pairs for speculative decoding and measure the real speedup.

Public API::

    from specpair import load_vocab, check_pair
    result = check_pair(load_vocab("target.gguf"), load_vocab("draft.gguf"))
    print(result.verdict, [f.rule for f in result.failures])
"""

from __future__ import annotations

from .errors import (
    BenchError,
    InputNotFoundError,
    MissingTokenizerError,
    NotGGUFError,
    SpecpairError,
    TokenizerJSONError,
    TruncatedHeaderError,
    UnsupportedRuntimeError,
)
from .i18n import resolve_lang
from .rules import CheckResult, Note, RuleFailure, check_pair, recommended_command
from .ruleset import DEFAULT_RUNTIME, SUPPORTED_RUNTIMES
from .vocab import Vocab, load_vocab

__version__ = "0.1.0"

__all__ = [
    "BenchError",
    "CheckResult",
    "DEFAULT_RUNTIME",
    "InputNotFoundError",
    "MissingTokenizerError",
    "NotGGUFError",
    "Note",
    "RuleFailure",
    "SUPPORTED_RUNTIMES",
    "SpecpairError",
    "TokenizerJSONError",
    "TruncatedHeaderError",
    "UnsupportedRuntimeError",
    "Vocab",
    "__version__",
    "check_pair",
    "load_vocab",
    "recommended_command",
    "resolve_lang",
]
