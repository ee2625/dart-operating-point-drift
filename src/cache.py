"""Caching layer for derived tables.

Recomputation is the expensive part of this project: the Drive mount is slow,
Colab sessions are short, and the same panels get rebuilt across experiments.
Every derived table therefore goes through :func:`cached_frame`, which keys on a
fingerprint of its upstream files plus an explicit version string.

Cached files live in ``dart_dev/cache/`` and are disposable. Deleting the
directory must only ever cost time.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .paths import DEV, MASTER, assert_writable

__all__ = [
    "cached_frame",
    "clear_cache",
    "fingerprint",
    "read_master_parquet",
    "read_master_csv",
    "read_master_json",
]


# ---------------------------------------------------------------------
# Fingerprinting
# ---------------------------------------------------------------------

def fingerprint(dependencies: Iterable[str | Path]) -> str:
    """Build a short digest describing the state of ``dependencies``.

    Uses each file's size and modification time rather than its contents. The
    upstream parquet files are hundreds of megabytes in aggregate and live on a
    network-backed mount, so hashing their bytes would defeat the purpose of
    caching. Size and mtime are sufficient here because the upstream tree is
    read-only and only changes when the collaborator regenerates it.

    Missing dependencies are recorded as such rather than raising, so a builder
    can declare optional inputs.
    """
    parts: list[str] = []

    for dependency in sorted(str(item) for item in dependencies):
        path = Path(dependency)
        if path.exists():
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}")
        else:
            parts.append(f"{path.name}:missing")

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


# ---------------------------------------------------------------------
# Read-only master accessors
# ---------------------------------------------------------------------

@lru_cache(maxsize=32)
def _read_parquet_cached(path_text: str, columns: tuple[str, ...] | None) -> pd.DataFrame:
    return pd.read_parquet(path_text, columns=list(columns) if columns else None)


def read_master_parquet(
    path: str | Path,
    columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Read a parquet file from the collaborator's tree, memoised in process.

    Returns a defensive copy so callers cannot mutate the shared cache entry.
    Pass ``columns`` to avoid materialising the full panel when only a few
    fields are needed.
    """
    column_key = tuple(columns) if columns is not None else None
    return _read_parquet_cached(str(path), column_key).copy()


@lru_cache(maxsize=32)
def _read_csv_cached(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text)


def read_master_csv(path: str | Path) -> pd.DataFrame:
    """Read a CSV manifest from the collaborator's tree, memoised in process."""
    return _read_csv_cached(str(path)).copy()


@lru_cache(maxsize=32)
def _read_json_cached(path_text: str) -> str:
    return Path(path_text).read_text(encoding="utf-8")


def read_master_json(path: str | Path) -> dict:
    """Read a JSON manifest from the collaborator's tree, memoised in process."""
    return json.loads(_read_json_cached(str(path)))


# ---------------------------------------------------------------------
# Derived-table cache
# ---------------------------------------------------------------------

def cached_frame(
    name: str,
    builder: Callable[[], pd.DataFrame],
    dependencies: Iterable[str | Path] = (),
    version: str = "v1",
    verbose: bool = True,
) -> pd.DataFrame:
    """Return ``builder()``'s result, reusing a cached parquet when possible.

    The cache key combines ``name``, ``version``, and a
    :func:`fingerprint` of ``dependencies``. Changing the builder's logic
    requires bumping ``version`` — the fingerprint only tracks inputs, not code.

    Args:
        name: Stable identifier for the table, used as the filename stem.
        builder: Zero-argument callable producing the table.
        dependencies: Upstream files whose state invalidates the cache.
        version: Bump when the builder's logic changes.
        verbose: Print cache hits, misses, and build times.

    Returns:
        The cached or freshly built table.
    """
    DEV.ensure()

    key = fingerprint(dependencies)
    cache_path = assert_writable(DEV.cache / f"{name}__{version}__{key}.parquet")

    if cache_path.exists():
        if verbose:
            print(f"[cache] hit  {name} ({version}/{key})")
        return pd.read_parquet(cache_path)

    if verbose:
        print(f"[cache] miss {name} ({version}/{key}) — building")

    started = time.perf_counter()
    frame = builder()
    elapsed = time.perf_counter() - started

    if not isinstance(frame, pd.DataFrame):
        raise TypeError(
            f"Builder for {name!r} returned {type(frame).__name__}, expected DataFrame."
        )

    frame.to_parquet(cache_path, index=False)

    # Retire older versions of the same table so the cache does not grow without
    # bound as versions and fingerprints change.
    for stale in DEV.cache.glob(f"{name}__*.parquet"):
        if stale != cache_path:
            stale.unlink()

    if verbose:
        print(
            f"[cache] built {name} in {elapsed:.1f}s "
            f"({len(frame):,} rows, {len(frame.columns)} columns)"
        )

    return frame


def clear_cache(name: str | None = None) -> int:
    """Delete cached tables and return how many files were removed.

    Args:
        name: Clear only this table when given, otherwise the whole cache.
    """
    pattern = f"{name}__*.parquet" if name else "*.parquet"

    removed = 0
    for path in DEV.cache.glob(pattern):
        assert_writable(path).unlink()
        removed += 1

    return removed


# ---------------------------------------------------------------------
# Convenience loaders for the certified inputs
# ---------------------------------------------------------------------

def load_enriched_panel(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the certified hourly DA/RT panel with calendar and LMP components."""
    return read_master_parquet(MASTER.enriched_panel, columns=columns)


def load_feature_panel(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the leakage-certified G1/G6 feature panel."""
    return read_master_parquet(MASTER.feature_panel, columns=columns)


def load_signed_tail_labels(columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Load the existing signed-tail development labels.

    This is the collaborator's baseline target. It is preserved as-is; the
    magnitude and direction targets in :mod:`src.labels` are built alongside it,
    never in place of it.
    """
    return read_master_parquet(MASTER.signed_tail_labels, columns=columns)


def load_fold_registry() -> pd.DataFrame:
    """Load the expanding walk-forward fold definitions (F1–F4)."""
    return read_master_csv(MASTER.fold_registry)


def load_information_policy() -> dict:
    """Load the pre-bid information policy that every feature must satisfy."""
    return read_master_json(MASTER.information_policy)
