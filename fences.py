"""The fences: every bucket object name and Earth Engine asset ID is built here.

Nothing else in the package assembles a path by hand. Writes are allowed only
inside the current run's folder (and exports/); asset IDs only inside
projects/<project>/assets/lst/. Names are letters, digits, - and _ only, so
nothing can climb out with '..' or a slash.
"""
from __future__ import annotations

import posixpath
import re

NAME = re.compile(r'[A-Za-z0-9_-]{1,100}')
DATE = re.compile(r'\d{4}-\d{2}-\d{2}')


class FenceError(PermissionError):
    """Raised when a path or ID would leave the project's fenced area."""


class Fences:
    def __init__(self, project: str, bucket: str, prefix: str = 'lst-agents/'):
        if not re.fullmatch(r'[a-z][a-z0-9-]{4,61}[a-z0-9]', project):
            raise FenceError(f'Not a valid Cloud project ID: {project!r}')
        if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]', bucket):
            raise FenceError(f'Not a valid bucket name: {bucket!r}')
        if not re.fullmatch(r'[A-Za-z0-9_-]+/', prefix):
            raise FenceError(f'The prefix must be one folder name ending in /: {prefix!r}')
        self.project, self.bucket, self.prefix = project, bucket, prefix
        self.asset_root = f'projects/{project}/assets/lst'

    # ── Cloud Storage ──────────────────────────────────────────────────────
    def gcs_key(self, rel: str) -> str:
        """Object name under the prefix. Refuses anything that could climb out of it."""
        raw = str(rel).strip()
        if not raw or raw.startswith('/') or '\\' in raw or '://' in raw:
            raise FenceError(f'Refused bucket path outside {self.prefix}: {rel!r}')
        norm = posixpath.normpath(raw)
        if norm in ('.', '..') or norm.startswith('../') or '/../' in f'/{norm}/':
            raise FenceError(f'Refused bucket path outside {self.prefix}: {rel!r}')
        key = self.prefix + norm
        if not key.startswith(self.prefix):
            raise FenceError(f'Refused bucket path outside {self.prefix}: {rel!r}')
        return key

    def run_key(self, run_id: str, rel: str) -> str:
        """Object name inside one run's folder."""
        self.check_name(run_id)
        return self.gcs_key(f'runs/{run_id}/{rel}')

    def check_write(self, key: str, run_id: str) -> str:
        """A full object name may be written only inside this run's folder or exports/."""
        allowed = (self.prefix + f'runs/{run_id}/', self.prefix + 'exports/')
        if posixpath.normpath(key) != key.rstrip('/') or not key.startswith(allowed):
            raise FenceError(f'Refused write outside this run: {key!r}')
        return key

    def gs_uri(self, key: str) -> str:
        if not key.startswith(self.prefix):
            raise FenceError(f'Refused bucket path outside {self.prefix}: {key!r}')
        return f'gs://{self.bucket}/{key}'

    # ── Earth Engine assets ────────────────────────────────────────────────
    @staticmethod
    def check_name(name: str) -> str:
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise FenceError(f'Refused name (letters, digits, - and _ only): {name!r}')
        return name

    def asset_id(self, *parts: str) -> str:
        """Asset ID inside projects/<project>/assets/lst/; each part is one plain name."""
        if not parts:
            return self.asset_root
        return '/'.join([self.asset_root, *(self.check_name(p) for p in parts)])

    def check_asset(self, aid: str) -> str:
        """Accepts only IDs that asset_id() could have made."""
        if aid == self.asset_root:
            return aid
        head = self.asset_root + '/'
        if not isinstance(aid, str) or not aid.startswith(head):
            raise FenceError(f'Refused asset outside {self.asset_root}: {aid!r}')
        for part in aid[len(head):].split('/'):
            self.check_name(part)
        return aid

    # ── Values agents send in ──────────────────────────────────────────────
    @staticmethod
    def check_date(d: str) -> str:
        if not isinstance(d, str) or not DATE.fullmatch(d):
            raise FenceError(f'Refused date (use YYYY-MM-DD): {d!r}')
        return d


def pass_name(date: str) -> str:
    """Asset-safe name for a pass date: 2023-07-14 → 20230714."""
    return Fences.check_date(date).replace('-', '')
