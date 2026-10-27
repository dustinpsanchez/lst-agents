"""The only tools agents have. Each returns a short JSON summary, never file contents or imagery.

Tools never take a path or an asset ID: they take dates, scopes and short text,
and build every name through the fences. Starting Earth Engine work only queues
a request; plain Python starts the tasks after the agent session ends.
"""
from __future__ import annotations

import asyncio
import json

from claude_agent_sdk import create_sdk_mcp_server, tool

from . import pipeline
from .fences import FenceError, Fences
from .store import clip, now_utc

SERVER = 'lst'


def full(name: str) -> str:
    return f'mcp__{SERVER}__{name}'


def _ok(obj) -> dict:
    text = json.dumps(obj, default=str)
    if len(text) > 12000:
        text = text[:11990] + '…"}'
    return {'content': [{'type': 'text', 'text': text}]}


def _err(msg: str) -> dict:
    return {'content': [{'type': 'text', 'text': f'Refused: {msg}'}], 'is_error': True}


SCOPE = {'type': 'object', 'properties': {'scope': {'type': 'string', 'enum': ['pilot', 'rest']}},
         'required': ['scope'], 'additionalProperties': False}
NONE = {'type': 'object', 'properties': {}, 'additionalProperties': False}
DROPS = {'type': 'array', 'maxItems': 300, 'items': {
    'type': 'object', 'additionalProperties': False, 'required': ['date', 'reason'],
    'properties': {'date': {'type': 'string', 'pattern': r'^\d{4}-\d{2}-\d{2}$'},
                   'reason': {'type': 'string', 'minLength': 3, 'maxLength': 200}}}}
SECTIONS = ['job', 'selection', 'spend', 'wald', 'qa', 'qa_verdict', 'stats', 'approvals', 'excluded',
            'notes', 'outputs', 'history', 'reviews']


def run_status_data(ctx) -> dict:
    m = ctx.m
    st = m['stage']
    by_state: dict[str, int] = {}
    for t in pipeline.stage_tasks(m, st):
        by_state[t['state']] = by_state.get(t['state'], 0) + 1
    sel = m.get('selection') or {}
    return {
        'run_id': ctx.run_id, 'stage': st, 'stage_means': pipeline.STAGE_LABEL.get(st, st),
        'pending_approval': (m.get('pending_approval') or {}).get('gate'),
        'approvals': sorted(m['approvals']), 'tasks_this_stage': by_state,
        'requests_this_stage': [f"{r['kind']}({r['scope']})" for r in m['requests'] if r['stage'] == st],
        'passes': {'eligible': len(sel.get('eligible') or []), 'selected': len(pipeline.selected(m)),
                   'pilot': sel.get('pilot'), 'excluded': len(m['excluded'])},
        'eecu_hours_used': pipeline.eecu_hours_used(m), 'claude_spend_usd': round(m['agent_cost_usd'], 3),
    }


def build_server(ctx):
    """In-process MCP server whose tools are bound to this run."""
    m = ctx.m

    async def save():
        await asyncio.to_thread(ctx.save)

    def blocked(name, args):
        return pipeline.check_tool(m, name, args)

    @tool('run_status', 'Where the run stands: stage, approvals, task counts, passes, spend so far.', NONE)
    async def run_status(args):
        return _ok(run_status_data(ctx))

    @tool('read_manifest', 'One section of the run record, summarised.',
          {'type': 'object', 'properties': {'section': {'type': 'string', 'enum': SECTIONS}},
           'required': ['section'], 'additionalProperties': False})
    async def read_manifest(args):
        sec = args['section']
        val = m.get(sec)
        if sec == 'selection' and val:
            val = {k: v for k, v in val.items() if k != 'passes'}
            val['rejected'] = dict(list(val.get('rejected', {}).items())[:40])
        if sec == 'spend' and val:
            val = {k: v for k, v in val.items() if k != 'pilot'}
        if sec == 'wald' and val:
            val = {k: v for k, v in val.items() if k != 'per_pass'}
        return _ok({sec: val})

    @tool('inventory_summary', 'The pass inventory after the job rules: counts by year, rejections, every eligible '
          'pass with its clear share, Sentinel-2 window and flags.', NONE)
    async def inventory_summary(args):
        sel = m.get('selection')
        if not sel:
            return _err('the inventory is not ready yet')
        rows = [{'date': d, 'sat': p['spacecraft'], 'clear': p['clear'], 's2_days': p['s2_window'],
                 'flags': p['flags']} for d, p in sorted(sel['passes'].items())]
        return _ok({'summary': sel['summary'], 'pilot_proposed_by_code': sel['pilot'],
                    'eligible': rows, 'rejected_examples': dict(list(sel['rejected'].items())[:25]),
                    'rules': {'landsat': ctx.job['landsat'], 'sentinel2': ctx.job['sentinel2']}})

    @tool('select_passes', 'Choose the passes to use: every eligible pass except the ones you drop, each with a '
          'reason. Call once. Dropped pilot passes are replaced automatically.',
          {'type': 'object', 'additionalProperties': False, 'required': ['drop', 'note'],
           'properties': {'drop': DROPS, 'note': {'type': 'string', 'maxLength': 1500}}})
    async def select_passes(args):
        why = blocked('select_passes', args)
        if why:
            return _err(why)
        sel = m['selection']
        drops = {}
        for d in args['drop']:
            Fences.check_date(d['date'])
            if d['date'] not in sel['passes']:
                return _err(f"{d['date']} is not an eligible pass")
            drops[d['date']] = d['reason']
        keep = [d for d in sel['eligible'] if d not in drops]
        if len(keep) < max(3, ctx.job['pilot_passes']):
            return _err(f'that would leave only {len(keep)} passes; keep at least 3')
        sel['selected'], sel['dropped'], sel['note'] = keep, drops, args['note']
        if any(p in drops for p in sel['pilot']):
            from .analysis import pick_pilot
            sel['pilot'] = pick_pilot([sel['passes'][d] for d in keep], ctx.job['pilot_passes'],
                                      ctx.job['sentinel2']['window_days'])
        m['pending_approval'] = None
        await save()
        years = {}
        for d in keep:
            years[d[:4]] = years.get(d[:4], 0) + 1
        return _ok({'selected': len(keep), 'dropped': len(drops), 'pilot': sel['pilot'], 'by_year': years})

    def starter(kind, schema, desc):
        @tool(f'start_{kind}', desc, schema)
        async def start(args):
            scope = args.get('scope', 'all')
            why = blocked(f'start_{kind}', {'scope': scope})
            if why:
                return _err(why)
            res = await asyncio.to_thread(pipeline.add_request, ctx, kind, scope, 'agent')
            return _ok(res)
        return start

    start_stacks = starter('stacks', SCOPE, 'Queue the 90 m training stacks (Sentinel-2 predictors, observed LST, '
                           'coverage) for the pilot passes or the rest, plus the static stacks they need.')
    start_wald = starter('wald', SCOPE, 'Queue one Wald test per pass (90 m → 810 m → back to 90 m, every alpha).')
    start_maps = starter('maps', SCOPE, 'Queue the 10 m model-detail maps, one per pass.')
    start_qa = starter('qa', NONE, 'Queue the quality checks for every selected pass, and the cross-check with the '
                       'other Landsat path.')
    start_stats = starter('stats', NONE, 'Queue the anomaly offsets and then the per-pixel time-series statistics.')

    @tool('wald_results', 'Wald test pooled over passes: chosen alpha, the gate, held-out gains on each half, '
          'and passes where model detail hurts. Before the full run, shows the pilot preview.', NONE)
    async def wald_results(args):
        w, which = (m.get('wald'), 'all passes') if m.get('wald') else (m.get('pilot_wald'), 'pilot only')
        if not w:
            return _err('no Wald results yet')
        tbl = w.get('table', {})
        from .analysis import cand_name
        keep = {'nofix', cand_name(0.0), cand_name(w['alpha']), cand_name(max(ctx.job['alphas']))}
        worst = sorted(w.get('per_pass', {}).items(), key=lambda kv: kv[1]['gain'])[:6]
        return _ok({'scope': which, 'alpha': w['alpha'], 'passed': w['passed'], 'reason': w.get('reason'),
                    'cross_check': w.get('cross_check'), 'gain_overall': w.get('gain_overall'),
                    'n_passes': w.get('n_passes'), 'n_eval': w.get('n_eval'),
                    'n_train_median': w.get('n_train_median'),
                    'rmse_table_K': {k: v for k, v in tbl.items() if k in keep},
                    'passes_where_detail_hurts': w.get('passes_where_detail_hurts', [])[:20],
                    'worst_passes': dict(worst), 'wald_task_failures': m.get('wald_failures', {})})

    @tool('spend_estimate', 'Projected Earth Engine compute (EECU-hours) and storage for the rest of the run, '
          'measured on the pilot, and whether it needs the person\'s approval.', NONE)
    async def spend_estimate(args):
        sp = m.get('spend')
        if not sp:
            return _err('the pilot has not finished yet')
        return _ok({**{k: v for k, v in sp.items() if k != 'pilot'},
                    'thresholds': {k: ctx.job['gates'][k] for k in ('eecu_hours_over', 'storage_gb_over',
                                                                     'ee_monthly_eecu_hours')}})

    @tool('qa_results', 'Quality-check results: passes flagged by the objective checks, the largest '
          'energy-check difference, the cross-check with the other path, and the station comparison.', NONE)
    async def qa_results(args):
        qa = m.get('qa')
        if not qa:
            return _err('QA results are not ready yet')
        flagged = dict(list(qa['flagged'].items())[:40])
        return _ok({**{k: v for k, v in qa.items() if k != 'flagged'}, 'flagged': flagged,
                    'n_flagged': len(qa['flagged']), 'thresholds': {
                        'energy_tol_k': ctx.job['gates']['energy_tol_k'], **ctx.job['validation']},
                    'notes': m.get('qa_notes', [])})

    @tool('record_qa_verdict', "Record the QA verdict. 'fail' holds the run for the person. You may exclude only "
          'passes the objective checks flagged, each with a reason.',
          {'type': 'object', 'additionalProperties': False, 'required': ['verdict', 'findings', 'exclude'],
           'properties': {'verdict': {'type': 'string', 'enum': ['pass', 'fail']},
                          'findings': {'type': 'string', 'minLength': 10, 'maxLength': 2500},
                          'exclude': DROPS}})
    async def record_qa_verdict(args):
        why = blocked('record_qa_verdict', args)
        if why:
            return _err(why)
        flagged = m['qa']['flagged']
        bad = [d['date'] for d in args['exclude'] if d['date'] not in flagged]
        if bad:
            return _err(f'only flagged passes can be excluded; not flagged: {bad[:10]}')
        for d in args['exclude']:
            m['excluded'][d['date']] = f"QA: {d['reason']}"
        m['qa_verdict'] = {'verdict': args['verdict'], 'findings': args['findings'], 'by': 'qa-validator',
                           'excluded': {d['date']: d['reason'] for d in args['exclude']}, 'at': now_utc()}
        m['pending_approval'] = None
        await save()
        return _ok({'recorded': args['verdict'], 'excluded': len(args['exclude']),
                    'passes_left': len(pipeline.selected(m))})

    @tool('record_review', 'Record your review of the pooled Wald test (a few lines).',
          {'type': 'object', 'additionalProperties': False, 'required': ['note'],
           'properties': {'note': {'type': 'string', 'minLength': 10, 'maxLength': 2000}}})
    async def record_review(args):
        why = blocked('record_review', args)
        if why:
            return _err(why)
        m.setdefault('reviews', {})[m['stage']] = {'by': 'model-trainer', 'note': args['note'], 'at': now_utc()}
        await save()
        return _ok({'recorded': True})

    @tool('stats_summary', 'Per-pixel statistics of the anomaly time series: percentiles over the area and the '
          'values at the station pixel.', NONE)
    async def stats_summary(args):
        s = m.get('stats')
        return _ok(s) if s else _err('statistics are not ready yet')

    @tool('request_approval', 'Ask the person at the notebook to approve a gate. Only they can approve; stop after '
          'this.',
          {'type': 'object', 'additionalProperties': False, 'required': ['gate', 'summary'],
           'properties': {'gate': {'type': 'string', 'enum': ['plan', 'spend', 'model', 'qa']},
                          'summary': {'type': 'string', 'minLength': 20, 'maxLength': 4000}}})
    async def request_approval(args):
        why = blocked('request_approval', args)
        if why:
            return _err(why)
        res = await asyncio.to_thread(pipeline.request_approval, ctx, args['gate'], args['summary'], 'orchestrator')
        return _ok(res)

    @tool('write_note', 'Add a short note to the run record.',
          {'type': 'object', 'additionalProperties': False, 'required': ['text'],
           'properties': {'text': {'type': 'string', 'minLength': 3, 'maxLength': 1500}}})
    async def write_note(args):
        m['notes'].append({'at': now_utc(), 'stage': m['stage'], 'text': args['text']})
        await save()
        return _ok({'saved': True})

    @tool('write_report', 'Save the final report (Markdown). The code appends exact tables from the run record.',
          {'type': 'object', 'additionalProperties': False, 'required': ['markdown'],
           'properties': {'markdown': {'type': 'string', 'minLength': 200, 'maxLength': 20000}}})
    async def write_report(args):
        why = blocked('write_report', args)
        if why:
            return _err(why)
        key = await asyncio.to_thread(pipeline.write_report, ctx, args['markdown'], 'orchestrator')
        return _ok({'saved': ctx.f.gs_uri(key)})

    all_tools = [run_status, read_manifest, inventory_summary, select_passes, start_stacks, start_wald,
                 start_maps, start_qa, start_stats, wald_results, spend_estimate, qa_results, record_qa_verdict,
                 record_review, stats_summary, request_approval, write_note, write_report]
    return create_sdk_mcp_server(SERVER, '1.0.0', all_tools), {t.name: t for t in all_tools}
