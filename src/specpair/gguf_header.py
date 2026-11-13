"""Standard-library GGUF (v2/v3) header reader.

Only the key/value section is parsed (tensor data is never touched), so files whose tensor data
was cut off, or that were only partly downloaded, still work as long as the header is complete.
Large arrays other than ``tokenizer.ggml.tokens`` are skipped without being materialised.
"""

from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, BinaryIO

from .errors import InputNotFoundError, NotGGUFError, SpecpairError, TruncatedHeaderError

MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)
KEEP_ARRAYS = {"tokenizer.ggml.tokens"}

_T_STRING = 8
_T_ARRAY = 9
# GGUF value type id -> struct format (little endian)
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.IGNORECASE)


@dataclass
class GGUFHeader:
    """Result of :func:`read_header`."""

    version: int
    n_tensors: int
    kv: dict[str, Any] = field(default_factory=dict)
    array_lengths: dict[str, int] = field(default_factory=dict)
    header_bytes: int = 0


class _Eof(Exception):
    """Internal: ran past the end of the file at ``offset``."""

    def __init__(self, offset: int) -> None:
        super().__init__(offset)
        self.offset = offset


class _Reader:
    def __init__(self, f: BinaryIO, size: int) -> None:
        self.f = f
        self.size = size

    def tell(self) -> int:
        return self.f.tell()

    def read(self, n: int) -> bytes:
        pos = self.f.tell()
        if n < 0 or pos + n > self.size:
            raise _Eof(pos)
        data = self.f.read(n)
        if len(data) != n:
            raise _Eof(pos)
        return data

    def skip(self, n: int) -> None:
        pos = self.f.tell()
        if n < 0 or pos + n > self.size:
            raise _Eof(pos)
        self.f.seek(n, os.SEEK_CUR)

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.read(n).decode("utf-8", errors="surrogateescape")

    def scalar(self, type_id: int) -> Any:
        fmt = _SCALARS[type_id]
        return struct.unpack(fmt, self.read(struct.calcsize(fmt)))[0]


class _BadType(Exception):
    def __init__(self, type_id: int, offset: int) -> None:
        super().__init__(type_id)
        self.type_id = type_id
        self.offset = offset


def _read_value(r: _Reader, type_id: int, keep: bool) -> Any:
    if type_id == _T_STRING:
        return r.string()
    if type_id == _T_ARRAY:
        item_type = r.u32()
        count = r.u64()
        if item_type == _T_ARRAY:
            raise _BadType(item_type, r.tell())
        if item_type == _T_STRING:
            if keep:
                return [r.string() for _ in range(count)]
            for _ in range(count):
                r.skip(r.u64())
            return count
        if item_type not in _SCALARS:
            raise _BadType(item_type, r.tell())
        width = struct.calcsize(_SCALARS[item_type])
        if keep:
            return [r.scalar(item_type) for _ in range(count)]
        r.skip(count * width)
        return count
    if type_id not in _SCALARS:
        raise _BadType(type_id, r.tell())
    return r.scalar(type_id)


def read_header(path: str | os.PathLike[str], label: str | None = None) -> GGUFHeader:
    """Parse the GGUF key/value header of ``path``.

    Raises :class:`InputNotFoundError`, :class:`NotGGUFError` or :class:`TruncatedHeaderError`
    (with the key being read and the byte offset) instead of low-level exceptions.
    """
    shown = label or os.fspath(path)
    if not os.path.isfile(path):
        raise InputNotFoundError("not_found", path=shown)
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            return _parse(f, size, shown)
    except OSError as exc:  # permission problems, I/O errors
        raise InputNotFoundError("unreadable", path=shown, reason=exc.strerror or str(exc)) from exc


def _parse(f: BinaryIO, size: int, shown: str) -> GGUFHeader:
    if size == 0:
        raise TruncatedHeaderError("file_empty", path=shown)
    r = _Reader(f, size)
    current = "(header)"
    try:
        if r.read(4) != MAGIC:
            raise NotGGUFError("not_gguf", path=shown)
        version = r.u32()
        if version not in SUPPORTED_VERSIONS:
            raise NotGGUFError("gguf_version", path=shown, version=version)
        n_tensors = r.u64()
        n_kv = r.u64()
        header = GGUFHeader(version=version, n_tensors=n_tensors)
        for index in range(n_kv):
            current = f"(key #{index})"
            key = r.string()
            current = key
            type_id = r.u32()
            keep = key in KEEP_ARRAYS
            value = _read_value(r, type_id, keep)
            if type_id == _T_ARRAY and not keep:
                header.array_lengths[key] = value
            else:
                header.kv[key] = value
                if type_id == _T_ARRAY:
                    header.array_lengths[key] = len(value)
        header.header_bytes = r.tell()
        return header
    except _Eof as exc:
        if current == "(header)":
            raise TruncatedHeaderError("header_cut_fixed", path=shown, offset=exc.offset) from None
        raise TruncatedHeaderError("header_truncated", path=shown, key=current, offset=exc.offset) from None
    except _BadType as exc:
        raise NotGGUFError("gguf_value_type", path=shown, key=current, type_id=exc.type_id,
                           offset=exc.offset) from None
    except SpecpairError:
        raise
    except (struct.error, MemoryError, OverflowError) as exc:  # corrupt counts
        raise TruncatedHeaderError("header_truncated", path=shown, key=current, offset=f.tell()) from exc


def split_info(path: str | os.PathLike[str]) -> tuple[int, int, str] | None:
    """Return ``(index, total, first_part_name)`` when the file name looks like ``x-00002-of-00003.gguf``."""
    name = os.path.basename(os.fspath(path))
    m = SPLIT_RE.match(name)
    if not m:
        return None
    first = f"{m['stem']}-{1:05d}-of-{m['total']}.gguf"
    return int(m["index"]), int(m["total"]), first
