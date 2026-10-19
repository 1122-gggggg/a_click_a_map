"""Immutable input fingerprints and concurrent-write guards."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path


class InputChangedError(RuntimeError):
    """Raised when an input no longer matches its frozen fingerprint."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    size: int
    mtime_ns: int
    sha256: str | None

    def as_dict(self) -> dict[str, str | int | None]:
        return asdict(self)

    def assert_unchanged(self) -> None:
        current = fingerprint_file(Path(self.path), sha256=self.sha256 is not None)
        if current != self:
            raise InputChangedError(f"input changed after it was frozen: {self.path}")


def fingerprint_file(path: Path, *, sha256: bool) -> FileFingerprint:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    digest = _sha256(resolved) if sha256 else None
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise InputChangedError(f"input changed while it was being fingerprinted: {resolved}")
    return FileFingerprint(
        path=str(resolved),
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        sha256=digest,
    )
