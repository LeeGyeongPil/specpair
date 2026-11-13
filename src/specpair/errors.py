"""Exception types. Every error carries a message code and parameters so the CLI can render
it in the user's language; ``str(exc)`` is always the English text."""

from __future__ import annotations

from typing import Any


class SpecpairError(Exception):
    """Base class for every error specpair raises on purpose (bad input, bad server, bad option)."""

    default_code = "generic"

    def __init__(self, code: str | None = None, **params: Any) -> None:
        self.code = code or self.default_code
        self.params = params
        super().__init__(self.message("en"))

    def message(self, lang: str = "en") -> str:
        """Return the error text in ``lang`` ("en" or "ko")."""
        from .i18n import t

        return t(lang, f"err.{self.code}", **self.params)


class InputNotFoundError(SpecpairError):
    """A path given by the user does not exist or is not the expected kind of path."""

    default_code = "not_found"


class NotGGUFError(SpecpairError):
    """The file does not start with the GGUF magic or uses an unsupported GGUF version."""

    default_code = "not_gguf"


class TruncatedHeaderError(SpecpairError):
    """The GGUF header ends before all key/value pairs were read."""

    default_code = "header_truncated"


class MissingTokenizerError(SpecpairError):
    """The GGUF file carries no tokenizer metadata (for example a later split part)."""

    default_code = "no_tokenizer"


class TokenizerJSONError(SpecpairError):
    """tokenizer.json / tokenizer_config.json is truncated, not UTF-8, or lacks required keys."""

    default_code = "tok_json_invalid"


class UnsupportedRuntimeError(SpecpairError):
    """The requested rule set is not bundled."""

    default_code = "runtime_unsupported"


class BenchError(SpecpairError):
    """A llama-server could not be reached or answered with something unusable."""

    default_code = "bench_connect"
