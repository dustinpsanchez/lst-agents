import pytest

from lst_agents import config
from lst_agents.fences import FenceError, Fences, pass_name
from lst_agents.store import Store
from conftest import FakeBucket, JOB_PATH


@pytest.fixture
def f():
    return Fences('project-ec9a437d-ade6-43c0-b02', 'my-lst-project-lst')


@pytest.mark.parametrize('bad', ['../x', '/etc/passwd', 'runs/../../x', 'gs://other/x', '..', '', 'a\\b',
                                 'runs/a/../../../x'])
def test_gcs_key_refuses_escapes(f, bad):
    with pytest.raises(FenceError):
        f.gcs_key(bad)


def test_gcs_key_ok(f):
    assert f.gcs_key('runs/r1/manifest.json') == 'lst-agents/runs/r1/manifest.json'
    assert f.run_key('r1', 'tables/wald/20230714') == 'lst-agents/runs/r1/tables/wald/20230714'


@pytest.mark.parametrize('bad', ['a/b', '../x', 'x y', 'projects/other/assets/x', ''])
def test_asset_names(f, bad):
    with pytest.raises(FenceError):
        f.asset_id('run', bad)


def test_asset_ids(f):
    assert f.asset_id('r1', 'stack90', 's_20230714') == \
        'projects/project-ec9a437d-ade6-43c0-b02/assets/lst/r1/stack90/s_20230714'
    f.check_asset(f.asset_id('r1', 'detail10'))
    with pytest.raises(FenceError):
        f.check_asset('projects/other/assets/lst/x')
    with pytest.raises(FenceError):
        f.check_asset('projects/project-ec9a437d-ade6-43c0-b02/assets/lst/../x')


def test_writes_only_in_run_folder(f):
    s = Store(FakeBucket(), f, 'r1')
    s.write_text('lst-agents/runs/r1/notes.txt', 'ok')
    s.write_text('lst-agents/exports/x.tif', 'ok')
    for bad in ('lst-agents/runs/r2/notes.txt', 'lst-agents/jobs/a.yaml', 'other/x',
                'lst-agents/runs/r1/../r2/x'):
        with pytest.raises(PermissionError):
            s.write_text(bad, 'no')


def test_pass_name():
    assert pass_name('2023-07-14') == '20230714'
    with pytest.raises(FenceError):
        pass_name('2023/07/14')


def test_job_defaults_and_errors():
    job = config.load_job(open(JOB_PATH).read())
    assert job['landsat']['path'] == 27 and job['grid']['wald_factor'] == 9
    assert job['validation']['crosscheck_path'] == 28
    with pytest.raises(config.JobError) as e:
        config.load_job('job: BAD NAME\nyears: [2020, 2010]\nlandsat: {path: 27, typo: 1}\n')
    assert 'Unknown setting' in str(e.value)
    with pytest.raises(config.JobError) as e:
        config.load_job('job: ok-name\nyears: [2020, 2010]\n')
    assert 'years' in str(e.value)
    job = config.load_job('job: tiny\nvalidation: {station: null, crosscheck_path: null}\n')
    assert job['validation']['station'] is None
