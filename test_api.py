"""The advance loop: stops at approvals and waits, falls back to code when an agent doesn't act."""
import asyncio

from lst_agents import api, pipeline, runner, tools
from conftest import write_geojson
from test_pipeline import inventory_rows


def test_advance_loop(session, job, fake_ee, monkeypatch, capsys):
    calls = []

    async def fake_session(ctx, stage, say=print):
        calls.append(stage)
        _, T = tools.build_server(ctx)
        if stage == 'plan':
            await T['select_passes'].handler({'drop': [], 'note': 'all'})
            await T['request_approval'].handler({'gate': 'plan', 'summary': 'Seven passes; pilot two. ' * 2})
        # every other stage: the agent does nothing, so the code must fall back after two tries
        ctx.m['agent_runs'][stage] = ctx.m['agent_runs'].get(stage, 0) + 1
        ctx.save()
        return {}

    monkeypatch.setattr(runner, 'run_stage_session', fake_session)
    run_id = api.start_run(session, open('jobs/san_antonio_p27.yaml').read())
    assert run_id.startswith('sa-p27-')
    assert asyncio.run(api.advance(session, run_id, wait_minutes=0)) == 'waiting'
    ctx = pipeline.load_run(session, run_id)
    fake_ee['finish_all']()
    write_geojson(session.bucket, ctx.m['tasks']['inventory_all']['out'], inventory_rows())
    assert asyncio.run(api.advance(session, run_id, wait_minutes=0)) == 'approval'
    assert calls == ['plan']
    api.approve(session, run_id, 'plan')
    assert asyncio.run(api.advance(session, run_id, wait_minutes=0)) == 'waiting'
    assert calls == ['plan', 'pilot_stacks', 'pilot_stacks']          # two tries, then the code fallback
    ctx = pipeline.load_run(session, run_id)
    assert any(r['by'] == 'code' for r in ctx.m['requests'])
    assert len(pipeline.stage_tasks(ctx.m, 'pilot_stacks')) >= 3
    out = capsys.readouterr().out
    assert 'WAITING FOR YOUR APPROVAL' in out and 'disconnect' in out
    api.status(session, run_id)


def test_start_run_resumes_unfinished(session, job, fake_ee):
    text = open('jobs/san_antonio_p27.yaml').read()
    first = api.start_run(session, text)
    assert api.start_run(session, text) == first
    assert api.list_runs(session) == [first]


def test_bad_job_prints_problems(session, capsys):
    assert api.start_run(session, 'job: x\nyears: [2030, 2020]\n') == ''
    assert 'years' in capsys.readouterr().out
