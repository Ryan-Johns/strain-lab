"""Keep the data store in a private Hugging Face dataset, so the hosted app keeps its data across restarts.

Turned on by two environment variables (environment variables or Streamlit secrets):
    STRAIN_LAB_DATASET   e.g. "your-name/strain-lab-data"
    HF_TOKEN             a token with write access to that dataset

How it works
- At start-up the catalog (catalog.json) and statistics (stats.parquet) are downloaded. Raw and filtered files
  are downloaded the first time someone opens them, so start-up stays quick however much data there is.
- Every change made through DataStore is queued and pushed a few seconds later as one commit, so a burst of
  changes (importing a ZIP, filtering 17 channels) becomes a single commit.
- If the start-up download fails, the app refuses to start rather than risk overwriting the online copy
  with an empty store.
Settings can also come from Streamlit secrets (Streamlit Community Cloud).
"""
import atexit
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import store as store_mod

SMALL_FILES = ("catalog.json", "stats.parquet")   # rewritten in place; uploaded from a snapshot of their bytes


def setting(name: str, default: str = "") -> str:
    """A setting from the environment, or from Streamlit secrets (Streamlit Community Cloud's "Secrets" box)."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        import streamlit as st                    # only imported when the env var is missing
        v = st.secrets.get(name)
    except Exception:
        v = None
    return str(v).strip() if v else default


def configured() -> bool:
    return bool(setting("STRAIN_LAB_DATASET")) and bool(setting("HF_TOKEN"))


class CloudMirror:
    def __init__(self, root, repo_id: str, token: str, api=None, delay_s: float = 5.0):
        self.root = Path(root)
        self.repo_id = repo_id
        if api is None:
            from huggingface_hub import HfApi
            api = HfApi(token=token)
        self.api = api
        self.delay_s = float(delay_s)
        self._lock = threading.Lock()
        self._fetch_lock = threading.Lock()
        self._dirty = set()
        self._last_change = 0.0
        self._remote = set()
        self._wake = threading.Event()
        self._stop = False
        self.last_push: Optional[str] = None
        self.last_error: Optional[str] = None
        self.commits = 0

    # ------------------------------------------------------------------ start-up
    def restore(self):
        """Download the catalog and statistics. Raises if the dataset can't be reached."""
        self.root.mkdir(parents=True, exist_ok=True)
        self._remote = set(self.api.list_repo_files(self.repo_id, repo_type="dataset"))
        for rel in SMALL_FILES:
            if rel in self._remote:
                self._download(rel)

    def _download(self, rel: str):
        self.api.hf_hub_download(self.repo_id, rel, repo_type="dataset", local_dir=str(self.root))

    # ------------------------------------------------------------------ hooks used by DataStore
    def fetch(self, rel: str):
        if (self.root / rel).exists():
            return
        with self._fetch_lock:
            if not (self.root / rel).exists() and rel in self._remote:
                self._download(rel)

    def on_write(self, rel: str):
        self._mark(rel)

    def on_delete(self, rel: str):
        self._mark(rel)

    def _mark(self, rel: str):
        with self._lock:
            self._dirty.add(rel)
            self._last_change = time.time()
        self._wake.set()

    # ------------------------------------------------------------------ pushing
    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._dirty)

    def flush(self) -> bool:
        """Push every queued change as one commit. Returns True on success (or nothing to do)."""
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        with self._lock:
            paths, self._dirty = set(self._dirty), set()
        if not paths:
            return True
        ops, added, deleted = [], set(), set()
        for rel in sorted(paths):
            p = self.root / rel
            if p.exists():
                data = p.read_bytes() if rel in SMALL_FILES else str(p)
                ops.append(CommitOperationAdd(path_in_repo=rel, path_or_fileobj=data))
                added.add(rel)
            elif rel in self._remote:
                ops.append(CommitOperationDelete(path_in_repo=rel))
                deleted.add(rel)
        if not ops:
            return True
        try:
            self.api.create_commit(self.repo_id, operations=ops, repo_type="dataset",
                                   commit_message=f"Strain Lab: {len(added)} updated, {len(deleted)} deleted")
        except Exception as e:                       # keep the changes queued and try again later
            with self._lock:
                self._dirty |= paths
            self.last_error = f"{type(e).__name__}: {e}"[:300]
            return False
        self._remote = (self._remote | added) - deleted
        self.last_push = datetime.now().strftime("%H:%M:%S")
        self.last_error = None
        self.commits += 1
        return True

    def _worker(self):
        backoff = 0.0
        while not self._stop:
            self._wake.wait(timeout=2.0)
            self._wake.clear()
            if not self.pending:
                continue
            wait = self.delay_s + backoff - (time.time() - self._last_change)
            if wait > 0:
                time.sleep(min(wait, 2.0))
                self._wake.set()
                continue
            backoff = 0.0 if self.flush() else min(max(backoff * 2, 10.0), 300.0)

    def start(self):
        threading.Thread(target=self._worker, name="strain-lab-cloud-sync", daemon=True).start()
        atexit.register(self.flush)

    def compact_history(self):
        """Squash the dataset's commit history into one commit (keeps the Hub fast after thousands of saves)."""
        self.flush()
        self.api.super_squash_history(self.repo_id, repo_type="dataset")


_MIRROR: Optional[CloudMirror] = None
_MIRROR_LOCK = threading.Lock()


def get_mirror(root, api=None) -> Optional[CloudMirror]:
    """The process-wide mirror (None when not configured). Installs the DataStore hooks on first use."""
    global _MIRROR
    if not configured() and api is None:
        return None
    with _MIRROR_LOCK:
        if _MIRROR is None:
            m = CloudMirror(root, setting("STRAIN_LAB_DATASET"), setting("HF_TOKEN"), api=api)
            m.restore()                              # raises -> app shows an error and stops
            store_mod.HOOKS.update(on_write=m.on_write, on_delete=m.on_delete, fetch=m.fetch)
            m.start()
            _MIRROR = m
        return _MIRROR
