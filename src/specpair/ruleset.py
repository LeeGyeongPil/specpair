"""Bundled rule sets (data files under ``specpair/rules``)."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from .errors import UnsupportedRuntimeError

DEFAULT_RUNTIME = "llama.cpp@b12360"
SUPPORTED_RUNTIMES = (DEFAULT_RUNTIME,)


@lru_cache(maxsize=None)
def load_ruleset(runtime: str = DEFAULT_RUNTIME) -> dict[str, Any]:
    """Return the rule data for ``runtime`` or raise :class:`UnsupportedRuntimeError`."""
    if runtime not in SUPPORTED_RUNTIMES:
        raise UnsupportedRuntimeError("runtime_unsupported", runtime=runtime,
                                      supported=", ".join(SUPPORTED_RUNTIMES))
    name = runtime.replace("@", "-") + ".json"
    text = (resources.files("specpair") / "rules" / name).read_text(encoding="utf-8")
    return json.loads(text)
