"""Optional Hugging Face dataset backing for hosted deployments.

A hosted server's storage is disposable, and the screening table takes a ~1.4 GB
SEC download plus minutes of parsing to rebuild. So `sec_refresh` can upload the
finished table to a private Hugging Face dataset, and this module downloads it back
on the first call that needs it after a cold start. The dataset mirrors the data
directory, so a file lands exactly where it would have been built.

Active only when ``HF_DATA_REPO`` names the dataset; otherwise every tool runs from
the live sources and the local cache.
"""
from __future__ import annotations

import os
import threading

from .config import DATA

_LOCK = threading.Lock()
_READY: set[str] = set()

# Tools that read the screening table, and the files that make it up.
_TABLE_TOOLS = {"screen_stocks", "refresh_screening_data"}
_TABLE_FILES = ("live_cache/universe/sec_universe.parquet",
                "live_cache/universe/sec_universe.meta.json")


def repo_id() -> str:
    return os.getenv("HF_DATA_REPO", "").strip()


def enabled() -> bool:
    return bool(repo_id())


def _revision() -> str:
    # Pinning is supported for reproducible deployments; main tracks the latest
    # uploaded refresh.
    return os.getenv("HF_DATA_REVISION", "main")


def _download_file(path_in_repo: str, *, optional: bool = False) -> None:
    target = DATA / path_in_repo
    if target.exists() or path_in_repo in _READY:
        return
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        hf_hub_download(
            repo_id=repo_id(),
            filename=path_in_repo,
            repo_type="dataset",
            revision=_revision(),
            token=os.getenv("HF_TOKEN") or None,
            local_dir=DATA,
        )
    except EntryNotFoundError:
        if not optional:
            raise
    _READY.add(path_in_repo)


def prepare(tool_name: str, arguments: dict | None) -> None:
    """Fetch what a tool reads from the dataset before its handler runs."""
    if not enabled() or tool_name not in _TABLE_TOOLS:
        return
    with _LOCK:
        for path in _TABLE_FILES:
            # Absent until the first refresh has been uploaded; the tools say so.
            _download_file(path, optional=True)


def status() -> dict:
    return {
        "mode": "huggingface" if enabled() else "local",
        "repo": repo_id() or None,
        "revision": _revision(),
        "data_dir": str(DATA),
    }
