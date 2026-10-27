"""Bucket reads and writes (fenced), the run manifest and the audit log."""
from __future__ import annotations

import datetime as dt
import json
import os
import threading

from .fences import Fences


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


class Store:
    """Thin wrapper over a google.cloud.storage Bucket. Every name goes through the fences."""

    def __init__(self, bucket, fences: Fences, run_id: str | None = None):
        self.bucket, self.f, self.run_id = bucket, fences, run_id

    # reads: anywhere under the prefix
    def read_text(self, key: str) -> str:
        return self.bucket.blob(self._inside(key)).download_as_text()

    def read_bytes(self, key: str) -> bytes:
        return self.bucket.blob(self._inside(key)).download_as_bytes()

    def exists(self, key: str) -> bool:
        return self.bucket.blob(self._inside(key)).exists()

    def list(self, key_prefix: str) -> list[str]:
        clean = self._inside(key_prefix) + ('/' if key_prefix.endswith('/') else '')
        return [b.name for b in self.bucket.list_blobs(prefix=clean)]

    def size(self, key: str) -> int:
        blob = self.bucket.get_blob(self._inside(key))
        return int(blob.size or 0) if blob else 0

    # writes: only this run's folder and exports/
    def write_text(self, key: str, text: str, content_type: str = 'text/plain') -> str:
        clean = self._inside(key)
        if clean != key:
            raise PermissionError(f'Refused bucket path that is not in plain form: {key!r}')
        self.f.check_write(clean, self.run_id)
        self.bucket.blob(clean).upload_from_string(text, content_type=content_type)
        return clean

    def write_json(self, key: str, obj) -> str:
        return self.write_text(key, json.dumps(obj, indent=1, default=str), 'application/json')

    def read_json(self, key: str):
        return json.loads(self.read_text(key))

    def _inside(self, key: str) -> str:
        if not isinstance(key, str) or not key.startswith(self.f.prefix):
            raise PermissionError(f'Refused bucket path outside {self.f.prefix}: {key!r}')
        return self.f.gcs_key(key[len(self.f.prefix):])


class Manifest:
    """The run record: one JSON file in the run folder, saved after every change."""

    def __init__(self, store: Store, run_id: str, data: dict | None = None):
        self.store, self.run_id = store, run_id
        self.key = store.f.run_key(run_id, 'manifest.json')
        self.data = data if data is not None else store.read_json(self.key)
        self._lock = threading.Lock()

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, k, v):
        self.data[k] = v

    def get(self, k, default=None):
        return self.data.get(k, default)

    def __contains__(self, k):
        return k in self.data

    def setdefault(self, k, default):
        return self.data.setdefault(k, default)

    def save(self):
        with self._lock:
            self.data['updated_utc'] = now_utc()
            self.store.write_json(self.key, self.data)

    # per-pass detail files live next to the manifest so the manifest stays small
    def file_key(self, name: str) -> str:
        return self.store.f.run_key(self.run_id, name)

    def save_file(self, name: str, obj) -> str:
        return self.store.write_json(self.file_key(name), obj)

    def load_file(self, name: str, default=None):
        key = self.file_key(name)
        if not self.store.exists(key):
            return default
        return self.store.read_json(key)


class Audit:
    """Append-only log of every tool call, refusal and agent session (runs/RUN_ID/audit.jsonl).

    Lines are appended to a local file in the scratch folder and the whole file is
    uploaded after each event, so the bucket copy is never more than one event behind.
    """

    def __init__(self, store: Store, run_id: str, scratch_dir: str):
        self.store, self.run_id = store, run_id
        self.key = store.f.run_key(run_id, 'audit.jsonl')
        os.makedirs(scratch_dir, exist_ok=True)
        self.path = os.path.join(scratch_dir, f'{run_id}-audit.jsonl')
        self._lock = threading.Lock()
        if not os.path.exists(self.path) and store.exists(self.key):
            with open(self.path, 'w') as fh:
                fh.write(store.read_text(self.key))

    def log(self, event: str, **fields):
        rec = {'at': now_utc(), 'event': event, **fields}
        line = json.dumps(rec, default=str)
        with self._lock:
            with open(self.path, 'a') as fh:
                fh.write(line + '\n')
            with open(self.path) as fh:
                self.store.write_text(self.key, fh.read(), 'application/x-ndjson')
        return rec

    def tail(self, n: int = 20) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as fh:
            lines = fh.readlines()[-n:]
        return [json.loads(x) for x in lines if x.strip()]


def clip(obj, n: int = 400) -> str:
    """Short, single-line text for logs and tool replies."""
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    s = ' '.join(s.split())
    return s if len(s) <= n else s[: n - 1] + '…'
