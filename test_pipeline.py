"""Walks a whole run through the stage machine with Earth Engine and the agents stubbed out."""
import asyncio

import pytest

from lst_agents import pipeline, tools
from lst_agents.analysis import cand_name
from lst_agents.fences import pass_name
from conftest import write_geojson

DATES = ['2018-06-01', '2018-07-03', '2019-06-20', '2022-07-14', '2022-08-15', '2023-06-09', '2023-07-27',
         '2025-08-10']


def inventory_rows():
    rows = []
    for i, d in enumerate(DATES):
        rows.append({'date': d, 'spacecraft': 'LANDSAT_9' if d >= '2022' and i % 2 else 'LANDSAT_8',
                     'time_utc': '17:03:11.1234', 'seen': 1.0, 'clear': [0.97, 0.6, 0.93, 0.99, 0.4, 0.95, 0.15,
                                                                         0.92][i],
                     's2_n_w': 2, 's2_cover_w': 0.97, 's2_n_fb': 4, 's2_cover_fb': 0.99, 'scene_cloud': 10})
    return rows


def wald_table(date, alphas, good=True):
    rows = []
    for half in ('west', 'east'):
        r = {'date': date, 'half': half, 'n': 800, 'n_train': 900, 'mse_nofix': 1.5}
        for a in alphas:
            r[f'mse_{cand_name(a)}'] = (1.0 - 0.5 * a + 0.5 * a * a) if good else (1.0 + a)
        rows.append(r)
    return rows


def call(tool_map, name, args):
    """Validates against the tool's schema the way the MCP server does, then runs the handler."""
    import jsonschema
    t = tool_map[name]
    try:
        jsonschema.validate(args, t.input_schema)
    except jsonschema.ValidationError as err:
        return {'content': [{'type': 'text', 'text': err.message}], 'is_error': True}
    return asyncio.run(t.handler(args))


def finish_tasks(ctx, fake_ee, stage):
    fake_ee['finish_all']()
    pipeline.refresh(ctx)
    for t in pipeline.stage_tasks(ctx.m, stage):
        if t['kind'] == 'wald':
            write_geojson(ctx.s.bucket, t['out'], wald_table(t['key'], ctx.job['alphas']))
        if t['kind'] == 'qa':
            rows = [{'date': d, 'n_fine_total': 5670 ** 2, 'energy': 0.004, 'lst_p001': 24.0, 'lst_p50': 41.0,
                     'lst_p999': 63.0, 'lst_count': 2e7, 'det2': 1.2, 'station': 44.0, 'T_mean': 314.0,
                     'T_count': 3e5} for d in t['meta']['dates']]
            rows[0]['energy'] = 0.5                     # one pass fails the energy check
            write_geojson(ctx.s.bucket, t['out'], rows)
        if t['kind'] == 'crosscheck':
            write_geojson(ctx.s.bucket, t['out'], [{'other_path': 28, 'r': 0.81, 'msd': 0.25, 'mad': 0.4,
                                                    'n_pixels': 30000, 'passes_main': 6, 'passes_other': 9}])
        if t['kind'] == 'offsets':
            write_geojson(ctx.s.bucket, t['out'], [{'date': d, 'offset': 312.0, 'area_mean': 312.0, 'n_valid': 1e5,
                                                    'reference': 1, 'method': 'reference pattern'}
                                                   for d in t['meta']['dates']])


def test_full_run(session, job, fake_ee, monkeypatch):
    job['gates']['eecu_hours_over'] = 10000
    job['gates']['storage_gb_over'] = 10000
    ctx = pipeline.create_run(session, job, 'job: sa-p27\n')
    assert ctx.m['stage'] == 'inventory' and ctx.m['emb_years'][-1] == 2024
    assert pipeline.next_action(ctx)[0] == 'wait'
    fake_ee['finish_all']()
    pipeline.refresh(ctx)
    write_geojson(session.bucket, ctx.m['tasks']['inventory_all']['out'], inventory_rows())
    assert pipeline.next_action(ctx) == ('advanced', 'plan')
    sel = ctx.m['selection']
    assert '2023-07-27' in sel['rejected']                       # 15% clear < 20%
    assert sel['passes']['2025-08-10']['emb_year'] == 2024
    assert pipeline.next_action(ctx) == ('agent', 'plan')

    _, T = tools.build_server(ctx)
    r = call(T, 'start_stacks', {'scope': 'pilot'})
    assert r.get('is_error')                                     # wrong stage: refused
    r = call(T, 'request_approval', {'gate': 'plan', 'summary': 'x' * 30})
    assert r.get('is_error')                                     # nothing selected yet
    r = call(T, 'select_passes', {'drop': [{'date': '2022-08-15', 'reason': 'wide window in green-up'}],
                                  'note': 'kept the rest'})
    assert not r.get('is_error')
    call(T, 'request_approval', {'gate': 'plan', 'summary': 'Six passes, two pilots. ' * 3})
    assert pipeline.next_action(ctx) == ('approval', 'plan')
    with pytest.raises(pipeline.StopRun):
        pipeline.approve(ctx, 'spend')
    pipeline.approve(ctx, 'plan', 'go')
    assert pipeline.next_action(ctx) == ('advanced', 'pilot_stacks')
    assert pipeline.next_action(ctx) == ('agent', 'pilot_stacks')

    call(T, 'start_stacks', {'scope': 'pilot'})
    assert call(T, 'start_stacks', {'scope': 'pilot'}).get('is_error')   # only once
    assert pipeline.next_action(ctx)[0] == 'execute'
    pipeline.execute_requests(ctx, say=lambda *a: None)
    kinds = sorted(t['kind'] for t in pipeline.stage_tasks(ctx.m, 'pilot_stacks'))
    assert kinds.count('stack') == 2 and kinds.count('static') >= 1
    assert pipeline.next_action(ctx)[0] == 'wait'
    finish_tasks(ctx, fake_ee, 'pilot_stacks')
    assert pipeline.next_action(ctx) == ('advanced', 'pilot_models')

    call(T, 'start_wald', {'scope': 'pilot'})
    assert pipeline.next_action(ctx)[0] == 'execute'
    pipeline.execute_requests(ctx, say=lambda *a: None)
    call(T, 'start_maps', {'scope': 'pilot'})
    pipeline.execute_requests(ctx, say=lambda *a: None)
    finish_tasks(ctx, fake_ee, 'pilot_models')
    assert pipeline.next_action(ctx) == ('advanced', 'spend')
    assert ctx.m['pilot_wald']['passed'] and ctx.m['spend']['projected_eecu_hours'] is not None
    assert pipeline.next_action(ctx) == ('advanced', 'stacks')

    for stage, tool, scope in (('stacks', 'start_stacks', 'rest'), ('wald', 'start_wald', 'rest')):
        assert pipeline.next_action(ctx) == ('agent', stage)
        call(T, tool, {'scope': scope})
        pipeline.execute_requests(ctx, say=lambda *a: None)
        finish_tasks(ctx, fake_ee, stage)
        pipeline.next_action(ctx)
    assert ctx.m['stage'] == 'model' and ctx.m['wald']['passed'] and ctx.m['wald']['alpha'] == 0.5
    assert pipeline.next_action(ctx) == ('agent', 'model')
    call(T, 'record_review', {'note': 'Alpha 0.5 holds on both halves.'})
    ctx.m['agent_runs']['model'] = 1
    assert pipeline.next_action(ctx) == ('advanced', 'maps')

    call(T, 'start_maps', {'scope': 'rest'})
    pipeline.execute_requests(ctx, say=lambda *a: None)
    finish_tasks(ctx, fake_ee, 'maps')
    assert pipeline.next_action(ctx) == ('advanced', 'qa')
    call(T, 'start_qa', {})
    pipeline.execute_requests(ctx, say=lambda *a: None)
    assert 'crosscheck_all' in ctx.m['tasks']
    finish_tasks(ctx, fake_ee, 'qa')
    assert pipeline.next_action(ctx) == ('advanced', 'qa_review')
    qa = ctx.m['qa']
    assert len(qa['flagged']) == 1 and qa['crosscheck']['r'] == 0.81 and abs(qa['crosscheck']['rmsd_K'] - 0.5) < 1e-9
    flagged = next(iter(qa['flagged']))
    bad = call(T, 'record_qa_verdict', {'verdict': 'pass', 'findings': 'one pass fails energy',
                                        'exclude': [{'date': '2019-06-20', 'reason': 'not flagged'}]})
    assert bad.get('is_error')
    call(T, 'record_qa_verdict', {'verdict': 'pass', 'findings': 'one pass fails the energy check; excluded',
                                  'exclude': [{'date': flagged, 'reason': 'energy check'}]})
    assert pipeline.next_action(ctx) == ('advanced', 'stats')
    assert flagged in ctx.m['excluded']

    call(T, 'start_stats', {})
    pipeline.execute_requests(ctx, say=lambda *a: None)
    finish_tasks(ctx, fake_ee, 'stats')
    assert pipeline.next_action(ctx)[0] == 'wait'                # offsets collected, stats image started
    assert 'stats_all' in ctx.m['tasks'] and flagged not in ctx.m['tasks']['stats_all']['meta']['dates']

    def fake_finish_stats(c):
        c.m['stats'] = {'summary': {'mean_cK': {'p05': -2.1, 'p50': 0.0, 'p95': 2.4}}, 'bands': ['mean_cK']}
        pipeline.set_stage(c, 'report', 'stats ready')
    monkeypatch.setattr(pipeline, 'finish_stats', fake_finish_stats)
    finish_tasks(ctx, fake_ee, 'stats')
    assert pipeline.next_action(ctx) == ('advanced', 'report')
    assert call(T, 'write_report', {'markdown': 'short'}).get('is_error')   # too short for the schema check
    call(T, 'write_report', {'markdown': '# Report\n' + 'All good. ' * 40})
    assert pipeline.next_action(ctx) == ('done', None)
    text = session.bucket.data[ctx.m['report']['key']]
    assert 'Appendix' in text and '| west | east |' in text
    assert all(k.startswith('lst-agents/') for k in session.bucket.data)


def test_spend_gate_and_fallbacks(session, job, fake_ee):
    ctx = pipeline.create_run(session, job, 'job: sa-p27\n')
    fake_ee['finish_all']()
    pipeline.refresh(ctx)
    write_geojson(session.bucket, ctx.m['tasks']['inventory_all']['out'], inventory_rows())
    pipeline.next_action(ctx)
    pipeline.fallback(ctx, 'plan', say=lambda *a: None)          # the scout never chose: code keeps all eligible
    assert len(ctx.m['selection']['selected']) == len(ctx.m['selection']['eligible'])
    assert pipeline.next_action(ctx) == ('approval', 'plan')     # code filed the request itself
    assert ctx.m['pending_approval']['by'] == 'code'
    pipeline.approve(ctx, 'plan')
    pipeline.next_action(ctx)
    pipeline.fallback(ctx, 'pilot_stacks', say=lambda *a: None)
    pipeline.execute_requests(ctx, say=lambda *a: None)
    finish_tasks(ctx, fake_ee, 'pilot_stacks')
    pipeline.next_action(ctx)
    pipeline.fallback(ctx, 'pilot_models', say=lambda *a: None)
    pipeline.execute_requests(ctx, say=lambda *a: None)
    fake_ee['finish_all'](eecu=36000.0)                          # 10 EECU-hours per task: over the 50 h limit
    pipeline.refresh(ctx)
    for t in pipeline.stage_tasks(ctx.m, 'pilot_models'):
        if t['kind'] == 'wald':
            write_geojson(session.bucket, t['out'], wald_table(t['key'], job['alphas']))
    assert pipeline.next_action(ctx) == ('advanced', 'spend')
    assert ctx.m['spend']['approval_required']
    assert pipeline.next_action(ctx) == ('agent', 'spend')
    ctx.m['agent_runs']['spend'] = 1
    assert pipeline.next_action(ctx) == ('approval', 'spend')
    _, T = tools.build_server(ctx)
    assert call(T, 'start_stacks', {'scope': 'rest'}).get('is_error')
    pipeline.reject(ctx, 'spend', 'too expensive')
    assert pipeline.next_action(ctx)[0] == 'stopped'


def test_failed_map_drops_only_that_pass(session, job, fake_ee):
    ctx = pipeline.create_run(session, job, 'job: sa-p27\n')
    fake_ee['finish_all']()
    pipeline.refresh(ctx)
    write_geojson(session.bucket, ctx.m['tasks']['inventory_all']['out'], inventory_rows())
    pipeline.next_action(ctx)
    pipeline.fallback(ctx, 'plan', say=lambda *a: None)
    assert pipeline.next_action(ctx) == ('approval', 'plan')
    pipeline.approve(ctx, 'plan')
    pipeline.next_action(ctx)
    pipeline.fallback(ctx, 'pilot_stacks', say=lambda *a: None)
    pipeline.execute_requests(ctx, say=lambda *a: None)
    finish_tasks(ctx, fake_ee, 'pilot_stacks')
    pipeline.next_action(ctx)
    pipeline.fallback(ctx, 'pilot_models', say=lambda *a: None)
    pipeline.execute_requests(ctx, say=lambda *a: None)
    pilot = ctx.m['selection']['pilot']
    tid = ctx.m['tasks'][f'map_{pilot[0]}']['id']
    fake_ee['status'][tid].update(state='FAILED', error_message='User memory limit exceeded.')
    finish_tasks(ctx, fake_ee, 'pilot_models')
    assert ctx.m['tasks'][f'map_{pilot[0]}']['attempts'] == 1     # not a transient error: no retry
    pipeline.next_action(ctx)
    assert pilot[0] in ctx.m['excluded'] and pilot[1] not in ctx.m['excluded']


def test_transient_failure_is_retried(session, job, fake_ee):
    ctx = pipeline.create_run(session, job, 'job: sa-p27\n')
    tid = ctx.m['tasks']['inventory_all']['id']
    fake_ee['status'][tid].update(state='FAILED', error_message='An internal error has occurred (request id x).')
    pipeline.refresh(ctx)
    rec = ctx.m['tasks']['inventory_all']
    assert rec['attempts'] == 2 and rec['state'] == 'READY' and rec['id'] != tid
