"""Test doubles: an in-memory bucket, a stand-in for Earth Engine, and a ready-made run."""
from __future__ import annotations

import json
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lst_agents import config, pipeline  # noqa: E402
from lst_agents.fences import Fences  # noqa: E402

JOB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'jobs', 'san_antonio_p27.yaml')


class FakeBlob:
    def __init__(self, bucket, name):
        self.b, self.name = bucket, name

    def upload_from_string(self, text, content_type=None):
        self.b.data[self.name] = text if isinstance(text, str) else text.decode()

    def download_as_text(self):
        return self.b.data[self.name]

    def download_as_bytes(self):
        return self.b.data[self.name].encode()

    def exists(self):
        return self.name in self.b.data

    @property
    def size(self):
        return len(self.b.data.get(self.name, ''))


class FakeBucket:
    def __init__(self):
        self.data = {}

    def blob(self, name):
        return FakeBlob(self, name)

    def get_blob(self, name):
        return FakeBlob(self, name) if name in self.data else None

    def list_blobs(self, prefix=''):
        return [FakeBlob(self, n) for n in sorted(self.data) if n.startswith(prefix)]


class FakeEE:
    """Accepts any chain of Earth Engine calls and returns more of itself. Catches Python mistakes
    (typos, wrong arity in our own helpers), not Earth Engine semantics."""
    EEException = Exception

    def __init__(self, name='ee', log=None):
        self._n, self._log = name, log if log is not None else []

    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        return FakeEE(f'{self._n}.{k}', self._log)

    def __call__(self, *a, **kw):
        self._log.append(self._n)
        return FakeEE(f'{self._n}()', self._log)

    def getInfo(self):
        return [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

    def _op(self, *a):
        return FakeEE(f'{self._n}<op>', self._log)

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = _op

    def __iter__(self):
        raise TypeError('an Earth Engine object was iterated in Python')


class Session:
    def __init__(self, tmp):
        self.project = 'project-ec9a437d-ade6-43c0-b02'
        self.bucket = FakeBucket()
        self.fences = Fences(self.project, 'my-lst-project-lst')
        self.sa_email = 'lst-agents-runner@example.iam.gserviceaccount.com'
        self.anthropic_key = 'sk-test'
        self.scratch_dir = str(tmp)
        self.versions = {}


class FakeGeo:
    """Stands in for gee.Geo inside pipeline tests: every builder returns a label."""

    def __init__(self, job, grid):
        self.region = 'region'

    def embedding_years(self):
        return types.SimpleNamespace(getInfo=lambda: [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024])

    def landsat(self, path):
        return types.SimpleNamespace(size=lambda: types.SimpleNamespace(getInfo=lambda: 40))

    def __getattr__(self, k):
        return lambda *a, **kw: f'{k}'


@pytest.fixture
def job():
    return config.load_job(open(JOB_PATH).read())


@pytest.fixture
def fake_ee(monkeypatch):
    """Replaces Earth Engine for the pipeline: exports get fake IDs, tasks finish when told to."""
    from lst_agents import gee
    state = {'n': 0, 'status': {}, 'started': []}

    def export(*a, **kw):
        state['n'] += 1
        tid = f'T{state["n"]:04d}'
        state['status'][tid] = {'id': tid, 'state': 'RUNNING'}
        state['started'].append((tid, a[1] if len(a) > 1 else kw.get('description')))
        return tid

    monkeypatch.setattr(gee, 'export_image', export)
    monkeypatch.setattr(gee, 'export_table', export)
    monkeypatch.setattr(gee, 'asset_exists', lambda aid: False)
    monkeypatch.setattr(gee, 'asset_size', lambda aid: 50_000_000)
    monkeypatch.setattr(gee, 'ensure_container', lambda aid, kind='FOLDER': None)
    monkeypatch.setattr(gee, 'task_states', lambda ids: {i: dict(state['status'][i]) for i in ids})
    monkeypatch.setattr(gee, 'Geo', FakeGeo)
    fake = FakeEE()
    monkeypatch.setitem(sys.modules, 'ee', fake)

    def finish_all(eecu=1800.0):
        for t in state['status'].values():
            if t['state'] == 'RUNNING':
                t.update(state='COMPLETED', batch_eecu_usage_seconds=eecu,
                         start_timestamp_ms=1_000_000, update_timestamp_ms=1_600_000)

    state['finish_all'] = finish_all
    return state


@pytest.fixture
def session(tmp_path):
    return Session(tmp_path)


def write_geojson(bucket, key_prefix, rows):
    bucket.data[key_prefix + '.geojson'] = json.dumps(
        {'type': 'FeatureCollection', 'features': [{'type': 'Feature', 'geometry': None, 'properties': r}
                                                   for r in rows]})
