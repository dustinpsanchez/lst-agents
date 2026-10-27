"""Run state: the stage machine, Earth Engine task bookkeeping, and the steps code does on its own.

Agents decide and explain; this module enforces. A stage can only move on when
its tasks have finished, its results are collected, and any approval it needs
has been given in the notebook (never by an agent).
"""
from __future__ import annotations

import datetime as dt
import io
import json
import math

from . import analysis
from .fences import FenceError, pass_name
from .grid import make_grid, utm_point
from .store import Audit, Manifest, Store, clip, now_utc

STAGES = ['inventory', 'plan', 'pilot_stacks', 'pilot_models', 'spend', 'stacks', 'wald', 'model',
          'maps', 'qa', 'qa_review', 'stats', 'report', 'done']

STAGE_LABEL = {
    'inventory': 'Finding every pass on the path', 'plan': 'Choosing passes (needs your approval)',
    'pilot_stacks': 'Pilot: 90 m training stacks', 'pilot_models': 'Pilot: Wald tests and 10 m maps',
    'spend': 'Projecting compute and storage from the pilot', 'stacks': '90 m stacks for the remaining passes',
    'wald': 'Wald tests for the remaining passes', 'model': 'Model gate: pooled Wald test',
    'maps': '10 m maps for the remaining passes', 'qa': 'Quality checks', 'qa_review': 'QA verdict',
    'stats': 'Per-pixel time-series statistics', 'report': 'Report', 'done': 'Finished',
}

# Which specialist works each stage, and what they are allowed to start there.
STAGE_AGENT = {'plan': 'data-scout', 'pilot_stacks': 'feature-builder', 'pilot_models': 'model-trainer',
               'spend': None, 'stacks': 'feature-builder', 'wald': 'model-trainer', 'model': 'model-trainer',
               'maps': 'inference-runner', 'qa': 'qa-validator', 'qa_review': 'qa-validator',
               'stats': 'inference-runner', 'report': None}
REQUESTS = {('stacks', 'pilot'): 'pilot_stacks', ('wald', 'pilot'): 'pilot_models', ('maps', 'pilot'): 'pilot_models',
            ('stacks', 'rest'): 'stacks', ('wald', 'rest'): 'wald', ('maps', 'rest'): 'maps',
            ('qa', 'all'): 'qa', ('stats', 'all'): 'stats'}
GATES = {'plan': 'plan', 'spend': 'spend', 'model': 'model', 'qa_review': 'qa'}

TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'CANCEL_REQUESTED', 'UNKNOWN'}
RETRYABLE = ('internal error', 'service unavailable', 'backend error', 'please try again', 'too many concurrent',
             'deadline exceeded', 'temporarily')
MAX_ATTEMPTS = 3


class StopRun(Exception):
    """A problem the run can't get past on its own; the message says what to do."""


# ── context ───────────────────────────────────────────────────────────────
class Ctx:
    """Everything a run needs: the signed-in session, the manifest, the audit log and the EE objects."""

    def __init__(self, session, manifest: Manifest, audit: Audit):
        self.s, self.m, self.audit = session, manifest, audit
        self.run_id = manifest.run_id
        self.job, self.grid = manifest['job'], manifest['grid']
        self.f, self.store = session.fences, manifest.store
        self._geo = None

    @property
    def geo(self):
        if self._geo is None:
            from .gee import Geo
            self._geo = Geo(self.job, self.grid)
        return self._geo

    def key(self, rel: str) -> str:
        return self.f.run_key(self.run_id, rel)

    def asset(self, *parts: str) -> str:
        return self.f.asset_id(self.run_id, *parts)

    def save(self):
        self.m.save()


def create_run(session, job: dict, job_text: str) -> Ctx:
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M')
    run_id = f"{job['job']}-{stamp}"
    store = Store(session.bucket, session.fences, run_id)
    if store.exists(session.fences.run_key(run_id, 'manifest.json')):
        raise StopRun(f'A run called {run_id} already exists. Wait a minute and try again.')
    grid = make_grid(job)
    data = {
        'run_id': run_id, 'phase': 2, 'created_utc': now_utc(), 'stage': 'inventory',
        'project': session.project, 'service_account': session.sa_email,
        'bucket_prefix': f'gs://{session.fences.bucket}/{session.fences.prefix}',
        'job': job, 'grid': grid, 'versions': session.versions,
        'tasks': {}, 'requests': [], 'approvals': {}, 'pending_approval': None, 'excluded': {},
        'agent_runs': {}, 'agent_cost_usd': 0.0, 'sessions': [], 'notes': [], 'history': [],
        'outputs': {'asset_folder': session.fences.asset_id(run_id)},
    }
    m = Manifest(store, run_id, data)
    m.save()
    store.write_text(session.fences.run_key(run_id, 'job.yaml'), job_text, 'text/yaml')
    audit = Audit(store, run_id, session.scratch_dir)
    audit.log('run_created', job=job['job'], grid=clip(grid, 300))
    ctx = Ctx(session, m, audit)
    from . import gee
    gee.ensure_container(session.fences.asset_id())
    gee.ensure_container(ctx.asset())
    for coll in ('static90', 'stack90', 'detail10'):
        gee.ensure_container(ctx.asset(coll), 'IMAGE_COLLECTION')
    m['emb_years'] = [int(y) for y in ctx.geo.embedding_years().getInfo()] if job['model']['use_embeddings'] else []
    launch(ctx, 'inventory', 'all', stage='inventory', scope='all')
    history(ctx, 'inventory', 'run created; inventory task started')
    return ctx


def load_run(session, run_id: str) -> Ctx:
    session.fences.check_name(run_id)
    store = Store(session.bucket, session.fences, run_id)
    key = session.fences.run_key(run_id, 'manifest.json')
    if not store.exists(key):
        raise StopRun(f'No run record found for {run_id!r}. Check the ID with list_runs().')
    m = Manifest(store, run_id)
    return Ctx(session, m, Audit(store, run_id, session.scratch_dir))


def history(ctx: Ctx, stage: str, note: str):
    ctx.m['history'].append({'at': now_utc(), 'stage': stage, 'note': note})
    ctx.save()


def set_stage(ctx: Ctx, stage: str, note: str = ''):
    ctx.m['stage'] = stage
    history(ctx, stage, note or f'entered {stage}')
    ctx.audit.log('stage', stage=stage, note=note)


# ── passes and scopes ─────────────────────────────────────────────────────
def passes(m) -> dict:
    return (m.get('selection') or {}).get('passes', {})


def selected(m) -> list[str]:
    sel = (m.get('selection') or {}).get('selected') or []
    return [d for d in sel if d not in m['excluded']]


def scope_dates(m, scope: str) -> list[str]:
    sel = selected(m)
    pilot = [d for d in (m.get('selection') or {}).get('pilot', []) if d in sel]
    if scope == 'pilot':
        return pilot
    if scope == 'rest':
        return [d for d in sel if d not in pilot]
    return sel


def static_key(p: dict) -> str:
    return str(p['emb_year']) if p.get('emb_year') is not None else 'all'


def items(ctx: Ctx, dates: list[str], detail: bool = False):
    """(pass, stack, static[, detail]) Earth Engine images for the given dates."""
    import ee
    out = []
    for d in dates:
        p = passes(ctx.m)[d]
        row = [p, ee.Image(ctx.asset('stack90', f's_{pass_name(d)}')),
               ee.Image(ctx.asset('static90', f'e_{static_key(p)}'))]
        if detail:
            row.append(ee.Image(ctx.asset('detail10', f'd_{pass_name(d)}')))
        out.append(tuple(row))
    return out


# ── tasks ─────────────────────────────────────────────────────────────────
def launch(ctx: Ctx, kind: str, key: str, stage: str, scope: str, meta: dict | None = None) -> str | None:
    """Starts one Earth Engine task (or skips it if its output already exists). Returns the task name."""
    from . import gee
    m, geo, g = ctx.m, ctx.geo, ctx.grid
    name = f'{kind}_{key}'
    rec = m['tasks'].get(name)
    if rec and rec['state'] not in TERMINAL:
        return name
    desc = f'{ctx.run_id}_{name}'
    bucket = ctx.f.bucket
    out = None
    if kind == 'inventory':
        out = ctx.key('tables/inventory')
        tid = gee.export_table(geo.inventory_fc(), desc, bucket, out)
    elif kind == 'static':
        out = ctx.asset('static90', f'e_{key}')
        if gee.asset_exists(out):
            tid = None
        else:
            tid = gee.export_image(geo.static_stack(None if key == 'all' else int(key)), desc, out,
                                   geo.region, g['crs'], g['coarse_t'])
    elif kind == 'stack':
        out = ctx.asset('stack90', f's_{pass_name(key)}')
        tid = None if gee.asset_exists(out) else gee.export_image(
            geo.pass_stack(passes(m)[key]), desc, out, geo.region, g['crs'], g['coarse_t'])
    elif kind == 'wald':
        out = ctx.key(f'tables/wald/{pass_name(key)}')
        p, stack, static = items(ctx, [key])[0]
        tid = gee.export_table(geo.wald_fc(p, stack, static, m['job']['alphas']), desc, bucket, out)
    elif kind == 'map':
        out = ctx.asset('detail10', f'd_{pass_name(key)}')
        if gee.asset_exists(out):
            tid = None
        else:
            p, stack, static = items(ctx, [key])[0]
            tid = gee.export_image(geo.detail_image(p, stack, static), desc, out,
                                   geo.region, g['crs'], g['fine_t'])
    elif kind == 'qa':
        out = ctx.key(f'tables/qa/{key}')
        st = m['job']['validation']['station']
        xy = utm_point(m['job'], st['lon'], st['lat']) if st else None
        tid = gee.export_table(geo.qa_fc(items(ctx, meta['dates'], detail=True), m['wald']['alpha'], xy),
                               desc, bucket, out)
    elif kind == 'crosscheck':
        out = ctx.key('tables/crosscheck')
        p27 = [(p, stack) for p, stack, _ in items(ctx, meta['dates'])]
        tid = gee.export_table(geo.crosscheck_fc(p27, meta['other_path']), desc, bucket, out)
    elif kind == 'offsets':
        out = ctx.key('tables/offsets')
        tid = gee.export_table(geo.offsets_fc(items(ctx, meta['dates']), meta['ref_dates']), desc, bucket, out)
    elif kind == 'stats':
        out = ctx.asset('stats10')
        dates = meta['dates']
        offs = m['offsets']['by_date']
        fy = {d: analysis.frac_year(d) for d in dates}
        img = geo.stats_image(items(ctx, dates, detail=True), offs, m['wald']['alpha'], fy,
                              m['job']['stats']['early_through'])
        tid = None if gee.asset_exists(out) else gee.export_image(img, desc, out, geo.region, g['crs'], g['fine_t'])
    else:
        raise ValueError(kind)
    attempts = (rec or {}).get('attempts', 0) + 1
    m['tasks'][name] = {'kind': kind, 'key': key, 'stage': stage, 'scope': scope, 'id': tid,
                        'state': 'COMPLETED' if tid is None else 'READY', 'attempts': attempts,
                        'out': out, 'meta': meta or {}, 'started_utc': now_utc(), 'error': None,
                        'eecu_s': None, 'secs': None, 'reused': tid is None}
    ctx.audit.log('task_started' if tid else 'task_reused', name=name, task_id=tid, out=out)
    return name


def stage_tasks(m, stage: str) -> list[dict]:
    return [dict(t, name=n) for n, t in m['tasks'].items() if t['stage'] == stage]


def refresh(ctx: Ctx) -> dict:
    """Updates task states from Earth Engine, retries transient failures. Returns counts by state."""
    from . import gee
    m = ctx.m
    live = {t['id']: n for n, t in m['tasks'].items() if t['id'] and t['state'] not in TERMINAL}
    if live:
        states = gee.task_states(list(live))
        for tid, name in live.items():
            st = states.get(tid)
            if not st:
                continue
            rec = m['tasks'][name]
            rec['state'] = st.get('state', rec['state'])
            if st.get('batch_eecu_usage_seconds') is not None:
                rec['eecu_s'] = float(st['batch_eecu_usage_seconds'])
            s0, s1 = st.get('start_timestamp_ms'), st.get('update_timestamp_ms')
            if s0 and s1 and rec['state'] in TERMINAL:
                rec['secs'] = round((int(s1) - int(s0)) / 1000)
            if rec['state'] == 'FAILED':
                rec['error'] = st.get('error_message', 'no message')
                ctx.audit.log('task_failed', name=name, error=rec['error'], attempt=rec['attempts'])
                err = (rec['error'] or '').lower()
                if rec['attempts'] < MAX_ATTEMPTS and any(k in err for k in RETRYABLE):
                    launch(ctx, rec['kind'], rec['key'], rec['stage'], rec['scope'], rec['meta'])
        ctx.save()
    counts: dict[str, int] = {}
    for t in m['tasks'].values():
        counts[t['state']] = counts.get(t['state'], 0) + 1
    return counts


def eecu_hours_used(m) -> float:
    return round(sum(t.get('eecu_s') or 0 for t in m['tasks'].values()) / 3600, 2)


def read_table(ctx: Ctx, key_prefix: str) -> list[dict]:
    """Rows of a GeoJSON table Earth Engine exported to the bucket."""
    names = [n for n in ctx.store.list(key_prefix) if n.endswith('.geojson')]
    if not names:
        raise StopRun(f'The export finished but no table was found at gs://{ctx.f.bucket}/{key_prefix}*.')
    rows = []
    for n in sorted(names):
        rows += [f.get('properties') or {} for f in json.loads(ctx.store.read_text(n)).get('features', [])]
    return rows


def _num(v):
    if v is None or (isinstance(v, (int, float)) and v <= -9998):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _find(props: dict, name: str, raw_group: str | None = None):
    """A value by its flat name, else by suffix inside the raw dictionary EE returned."""
    v = _num(props.get(name))
    if v is not None or not raw_group:
        return v
    raw = props.get(f'raw_{raw_group}') or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            raw = {}
    suffix = name.split('_', 1)[-1] if '_' in name else name
    for k, val in raw.items():
        if k == name or k.endswith(suffix):
            return _num(val)
    return None


# ── stage finishers ───────────────────────────────────────────────────────
def finish_inventory(ctx: Ctx):
    m, job = ctx.m, ctx.job
    rows = []
    for r in read_table(ctx, ctx.key('tables/inventory')):
        rows.append({'date': r['date'], 'spacecraft': r.get('spacecraft'), 'time_utc': r.get('time_utc'),
                     'rows': r.get('rows'), 'scene_cloud': _num(r.get('scene_cloud')),
                     'seen': _num(r.get('seen')), 'clear': _num(r.get('clear')),
                     's2_n_w': _num(r.get('s2_n_w')), 's2_cover_w': _num(r.get('s2_cover_w')),
                     's2_n_fb': _num(r.get('s2_n_fb')), 's2_cover_fb': _num(r.get('s2_cover_fb'))})
    m.save_file('inventory.json', rows)
    eligible, rejected = analysis.eligibility(rows, job, m.get('emb_years', []))
    if not eligible:
        raise StopRun('No pass met the job rules. Lower landsat.min_clear or sentinel2.min_cover and start a new run.')
    pilot = analysis.pick_pilot(eligible, job['pilot_passes'], job['sentinel2']['window_days'])
    m['selection'] = {'passes': {p['date']: p for p in eligible}, 'eligible': [p['date'] for p in eligible],
                      'rejected': rejected, 'pilot': pilot, 'selected': None, 'dropped': {},
                      'summary': analysis.summarize_inventory(rows, eligible, rejected)}
    set_stage(ctx, 'plan', f'{len(eligible)} of {len(rows)} passes meet the rules')


def pilot_measurements(ctx: Ctx) -> dict:
    from . import gee
    m = ctx.m
    out = {'eecu_s': {}, 'bytes': {}, 'secs': {}}
    for t in m['tasks'].values():
        if t['scope'] != 'pilot' or t['state'] != 'COMPLETED' or t['kind'] not in ('static', 'stack', 'wald', 'map'):
            continue
        if not t.get('reused'):
            out['eecu_s'].setdefault(t['kind'], []).append(t.get('eecu_s'))
            out['secs'].setdefault(t['kind'], []).append(t.get('secs'))
        if t['kind'] in ('static', 'stack', 'map'):
            out['bytes'].setdefault(t['kind'], []).append(gee.asset_size(t['out']))
    return out


def collect_wald(ctx: Ctx, dates: list[str]) -> dict:
    rows = ctx.m.load_file('wald_rows.json', {})
    for d in dates:
        if d in rows:
            continue
        t = ctx.m['tasks'].get(f'wald_{d}')
        if not t or t['state'] != 'COMPLETED':
            continue
        rows[d] = [{k: (_num(v) if k.startswith('mse_') or k in ('n', 'n_train') else v) for k, v in r.items()}
                   for r in read_table(ctx, t['out'])]
    ctx.m.save_file('wald_rows.json', rows)
    return rows


def pooled_wald(ctx: Ctx, dates: list[str]) -> dict:
    rows = collect_wald(ctx, dates)
    flat = [dict(r, date=d) for d in dates for r in rows.get(d, [])]
    res = analysis.pool_wald(flat, ctx.job['alphas'])
    res['n_train_median'] = sorted(int(r.get('n_train') or 0) for r in flat)[len(flat) // 2] if flat else 0
    return res


def finish_pilot_models(ctx: Ctx):
    m = ctx.m
    pilot = scope_dates(m, 'pilot')
    if not pilot:
        raise StopRun('Every pilot pass failed. Look at the task errors with status(RUN_ID) before going further.')
    m['pilot_wald'] = pooled_wald(ctx, pilot)
    meas = pilot_measurements(ctx)
    rest = scope_dates(m, 'rest')
    static_done = {t['key'] for t in m['tasks'].values() if t['kind'] == 'static' and t['state'] == 'COMPLETED'}
    static_rest = {static_key(passes(m)[d]) for d in rest} - static_done
    m['spend'] = analysis.spend_projection(meas, len(rest), len(static_rest), len(selected(m)), ctx.grid, ctx.job)
    m['spend']['pilot'] = meas
    try:
        import ee
        q = ee.data.getAssetRootQuota(f'projects/{ctx.s.project}/assets/')
        m['spend']['asset_quota'] = {k: q.get(k) for k in ('asset_size', 'asset_count')}
    except Exception as err:                     # the service account may not be allowed to read quotas
        m['spend']['asset_quota'] = f'not readable: {clip(str(err), 120)}'
    m['spend']['eecu_hours_used_so_far'] = eecu_hours_used(m)
    set_stage(ctx, 'spend', 'pilot measured')


def finish_qa(ctx: Ctx):
    m, job = ctx.m, ctx.job
    rows = []
    for t in stage_tasks(m, 'qa'):
        if t['kind'] != 'qa' or t['state'] != 'COMPLETED':
            continue
        for r in read_table(ctx, t['out']):
            det2 = _find(r, 'det2', 'det')
            cnt = _find(r, 'lst_count', 'lst')
            rows.append({'date': r['date'],
                         'energy_max_K': _find(r, 'energy', 'energy'),
                         'lst_p001_C': _find(r, 'lst_p001', 'lst'), 'lst_p50_C': _find(r, 'lst_p50', 'lst'),
                         'lst_p999_C': _find(r, 'lst_p999', 'lst'),
                         'detail_rms_K': math.sqrt(det2) if det2 is not None else None,
                         'valid_share': (cnt / float(r.get('n_fine_total') or 1)) if cnt is not None else None,
                         'station_lst_C': _find(r, 'station', 'station'),
                         'T_mean_K': _find(r, 'T_mean', 'T')})
    m.save_file('qa_rows.json', rows)
    judged = analysis.qa_judgement(rows, job)
    missing = [d for d in selected(m) if d not in judged]
    for d in missing:                                   # never deliver a pass that wasn't checked
        m['excluded'][d] = 'QA could not be computed for this pass'
    qa = {'alpha': m['wald']['alpha'], 'passes_checked': len(judged), 'passes_missing': missing,
          'flagged': {d: v['flags'] for d, v in judged.items() if not v['ok']},
          'energy_max_K': max((v['energy_max_K'] for v in judged.values() if v['energy_max_K'] is not None),
                              default=None)}
    m.save_file('qa_by_pass.json', judged)
    cc = m['tasks'].get('crosscheck_all')
    if cc and cc['state'] == 'COMPLETED':
        r = read_table(ctx, cc['out'])[0]
        msd = _find(r, 'msd', 'diff')
        qa['crosscheck'] = {'other_path': r.get('other_path'), 'r': _find(r, 'r', 'r'),
                            'rmsd_K': math.sqrt(msd) if msd is not None else None,
                            'mean_abs_diff_K': _find(r, 'mad', 'diff'), 'pixels': _find(r, 'n_pixels', 'diff'),
                            'passes_main': r.get('passes_main'), 'passes_other': r.get('passes_other')}
    elif cc:
        qa['crosscheck'] = {'failed': cc.get('error')}
    qa['station'] = station_check(ctx, judged)
    m['qa'] = qa
    set_stage(ctx, 'qa_review', f"{len(judged)} passes checked, {len(qa['flagged'])} flagged")


def station_check(ctx: Ctx, judged: dict) -> dict | None:
    """Compares the 10 m LST at the station pixel with NOAA LCD air temperature (not a gate)."""
    st = ctx.job['validation']['station']
    if not st:
        return None
    try:
        import pandas as pd
        csvs = [n for n in ctx.store.list(ctx.f.gcs_key('stations') + '/') if n.lower().endswith('.csv')]
        if not csvs:
            return {'note': 'no station CSV in the bucket under stations/'}
        lcd = pd.read_csv(io.BytesIO(ctx.store.read_bytes(csvs[0])), low_memory=False)
        inv = {r['date']: r for r in ctx.m.load_file('inventory.json', [])}
        pairs, used = [], 0
        for d, v in judged.items():
            t = (inv.get(d) or {}).get('time_utc')
            if v.get('station_lst_C') is None or not t:
                continue
            when_local = pd.Timestamp(f'{d}T{str(t)[:8]}') - pd.Timedelta(hours=6)   # LCD: local standard time
            hit = analysis.lcd_air_temp(lcd, when_local, st.get('lcd_units', 'F'))
            if hit:
                pairs.append((v['station_lst_C'], hit[0]))
                used += 1
        out = analysis.station_compare(pairs) or {'n': 0}
        out['file'] = csvs[0].rsplit('/', 1)[-1]
        out['station'] = st['name']
        return out
    except Exception as err:
        return {'error': clip(str(err), 200)}


def finish_stats(ctx: Ctx):
    """Summary numbers from the statistics asset, read at 90 m (cheap)."""
    import ee
    geo, m = ctx.geo, ctx.m
    img = ee.Image(ctx.asset('stats10'))
    bands = img.bandNames().getInfo()
    scaled = ee.Image.cat([img.select(b).toFloat().divide(1 if b.startswith('n_') else 100).rename(b) for b in bands])
    pct = ee.Reducer.percentile([5, 50, 95], ['p05', 'p50', 'p95'])
    s = scaled.reduceRegion(pct, geo.region, **geo.kw('coarse_t')).getInfo()
    summary = {b: {k: s.get(f'{b}_{k}') for k in ('p05', 'p50', 'p95')} for b in bands}
    st = ctx.job['validation']['station']
    if st:
        x, y = utm_point(ctx.job, st['lon'], st['lat'])
        at = scaled.reduceRegion(ee.Reducer.first(), ee.Geometry.Point([x, y], ctx.grid['crs']),
                                 crs=ctx.grid['crs'], crsTransform=ctx.grid['fine_t']).getInfo()
        summary['at_station'] = {'station': st['name'], **at}
    m['stats'] = {'asset': ctx.asset('stats10'), 'bands': bands, 'units': 'K (n_* are counts)', 'summary': summary}
    m['outputs']['stats_asset'] = ctx.asset('stats10')
    m['outputs']['detail_collection'] = ctx.asset('detail10')
    m['outputs']['stack_collection'] = ctx.asset('stack90')
    set_stage(ctx, 'report', 'statistics ready')


# ── requests from agents and approvals from you ───────────────────────────
def check_tool(m, tool: str, args: dict) -> str | None:
    """Why a tool call is not allowed right now, or None if it is. Shared by the hook and the tools."""
    st = m['stage']
    pend = m.get('pending_approval')
    if tool in ('start_stacks', 'start_wald', 'start_maps', 'start_qa', 'start_stats'):
        kind = tool[len('start_'):]
        scope = args.get('scope', 'all')
        want = REQUESTS.get((kind, scope))
        if want is None:
            return f'{tool} does not take scope {scope!r}.'
        if want != st:
            return f'{tool}({scope!r}) belongs to stage {want!r}; the run is at {st!r}.'
        if pend:
            return f"waiting for your approval of {pend['gate']!r} first"
        if any(r['kind'] == kind and r['scope'] == scope for r in m['requests']):
            return 'already requested in this stage'
        if any(t['stage'] == st and t['kind'] == {'stacks': 'stack', 'maps': 'map'}.get(kind, kind)
               for t in m['tasks'].values()):
            return 'already started in this stage'
        if kind == 'maps' and scope == 'rest' and not ((m.get('wald') or {}).get('passed') or 'model' in m['approvals']):
            return 'the model gate has not passed or been approved'
        return None
    if tool == 'select_passes':
        if st != 'plan':
            return f'passes are chosen in the plan stage; the run is at {st!r}'
        if 'plan' in m['approvals']:
            return 'the plan is already approved'
        return None
    if tool == 'request_approval':
        gate = args.get('gate')
        if GATES.get(st) != gate:
            return f'stage {st!r} has no {gate!r} approval'
        if gate == 'plan' and not (m.get('selection') or {}).get('selected'):
            return 'choose the passes with select_passes first'
        if gate == 'spend' and not (m.get('spend') or {}).get('approval_required'):
            return 'the projected spend is within the job limits; no approval needed'
        if gate == 'model' and (m.get('wald') or {}).get('passed'):
            return 'the model passed the Wald test; no approval needed'
        if gate == 'qa' and (m.get('qa_verdict') or {}).get('verdict') != 'fail':
            return 'a QA approval is only needed after a fail verdict'
        if gate in m['approvals']:
            return 'already approved'
        return None
    if tool == 'record_qa_verdict':
        return None if st == 'qa_review' else f'the QA verdict is recorded in qa_review; the run is at {st!r}'
    if tool == 'record_review':
        return None if st == 'model' else f'reviews are recorded in the model stage; the run is at {st!r}'
    if tool == 'write_report':
        return None if st == 'report' else f'the report is written in the report stage; the run is at {st!r}'
    return None


def add_request(ctx: Ctx, kind: str, scope: str, by: str) -> dict:
    m = ctx.m
    dates = scope_dates(m, scope)
    if kind == 'stacks':
        done = {t['key'] for t in m['tasks'].values() if t['kind'] == 'static'}
        statics = sorted({static_key(passes(m)[d]) for d in dates} - done)
        n = len(dates) + len(statics)
        what = f'{len(dates)} pass stacks and {len(statics)} static stacks (90 m)'
    elif kind == 'qa':
        n = math.ceil(len(dates) / QA_CHUNK) + (1 if ctx.job['validation']['crosscheck_path'] else 0)
        what = f'QA tables for {len(dates)} passes'
    elif kind == 'stats':
        n, what = 2, f'anomaly offsets, then per-pixel statistics over {len(dates)} passes'
    else:
        n = len(dates)
        what = f"{n} {'Wald tests' if kind == 'wald' else '10 m detail maps'}"
    m['requests'].append({'kind': kind, 'scope': scope, 'stage': m['stage'], 'by': by, 'at': now_utc(),
                          'done': False, 'tasks': n})
    ctx.save()
    return {'queued': what, 'tasks': n, 'starts': 'right after this agent session ends'}


QA_CHUNK = 10


def execute_requests(ctx: Ctx, say=print) -> int:
    """Starts the tasks agents asked for. Runs in plain Python after the agent session."""
    m = ctx.m
    started = 0
    for req in [r for r in m['requests'] if not r['done']]:
        kind, scope, stage = req['kind'], req['scope'], req['stage']
        dates = scope_dates(m, scope)
        if kind == 'stacks':
            statics = sorted({static_key(passes(m)[d]) for d in dates})
            todo = [('static', s) for s in statics] + [('stack', d) for d in dates]
        elif kind == 'wald':
            todo = [('wald', d) for d in dates]
        elif kind == 'maps':
            todo = [('map', d) for d in dates]
        elif kind == 'qa':
            todo = [('qa', f'chunk{i // QA_CHUNK:02d}') for i in range(0, len(dates), QA_CHUNK)]
            if ctx.job['validation']['crosscheck_path']:
                todo.append(('crosscheck', 'all'))
        elif kind == 'stats':
            todo = [('offsets', 'all')]
        else:
            continue
        say(f'Starting {len(todo)} Earth Engine task(s) for {kind} ({scope}) …')
        for i, (k, key) in enumerate(todo, 1):
            if f'{k}_{key}' in m['tasks'] and m['tasks'][f'{k}_{key}']['state'] != 'FAILED':
                continue
            meta = None
            if k == 'qa':
                j = int(key[5:])
                meta = {'dates': dates[j * QA_CHUNK:(j + 1) * QA_CHUNK]}
            elif k == 'crosscheck':
                meta = {'dates': dates, 'other_path': ctx.job['validation']['crosscheck_path']}
                if not other_path_has_scenes(ctx, meta['other_path']):
                    m.setdefault('qa_notes', []).append(f"path {meta['other_path']} has no scenes over the area; "
                                                        'cross-check skipped')
                    continue
            elif k == 'offsets':
                ref = [d for d in dates if passes(m)[d]['clear'] >= ctx.job['landsat']['reference_clear']]
                meta = {'dates': dates, 'ref_dates': ref}
            try:
                launch(ctx, k, key, stage, scope, meta)
                started += 1
            except FenceError:
                raise
            except Exception as err:          # one bad pass shouldn't stop the batch
                m['tasks'][f'{k}_{key}'] = {'kind': k, 'key': key, 'stage': stage, 'scope': scope, 'id': None,
                                            'state': 'FAILED', 'attempts': MAX_ATTEMPTS, 'out': None,
                                            'meta': meta or {}, 'error': f'could not start: {clip(str(err), 300)}',
                                            'eecu_s': None, 'secs': None, 'reused': False}
                ctx.audit.log('task_start_failed', name=f'{k}_{key}', error=clip(str(err), 300))
            if i % 10 == 0:
                ctx.save()
                say(f'  {i}/{len(todo)}')
        req['done'] = True
        ctx.save()
    return started


def other_path_has_scenes(ctx: Ctx, path: int) -> bool:
    try:
        return ctx.geo.landsat(path).size().getInfo() > 0
    except Exception:
        return False


def request_approval(ctx: Ctx, gate: str, summary: str, by: str) -> dict:
    ctx.m['pending_approval'] = {'gate': gate, 'summary': summary[:4000], 'by': by, 'at': now_utc()}
    ctx.save()
    ctx.audit.log('approval_requested', gate=gate, by=by)
    return {'requested': gate, 'next': 'The person running the notebook approves or rejects it; stop here.'}


def approve(ctx: Ctx, gate: str, note: str = '') -> str:
    m = ctx.m
    want = GATES.get(m['stage'])
    if gate != want:
        raise StopRun(f"Nothing to approve for {gate!r} right now; the run is at stage {m['stage']!r}"
                      + (f" (its gate is {want!r})." if want else '.'))
    pend = m.get('pending_approval') or {}
    if pend.get('gate') != gate:
        raise StopRun(f'Nothing is waiting for your approval of {gate!r} yet. Run the advance cell; it shows the '
                      'request and its summary when there is one.')
    m['approvals'][gate] = {'at': now_utc(), 'note': note[:500], 'by': 'notebook user'}
    m['pending_approval'] = None
    ctx.save()
    ctx.audit.log('approved', gate=gate, note=note[:200])
    return f'Approved {gate!r}. Run the advance cell to continue.'


def reject(ctx: Ctx, gate: str, note: str = '') -> str:
    m = ctx.m
    m['pending_approval'] = None
    m['stopped'] = {'gate': gate, 'note': note[:500], 'at': now_utc()}
    m['stage_before_stop'] = m['stage']
    m['stage'] = 'stopped'
    ctx.save()
    ctx.audit.log('rejected', gate=gate, note=note[:200])
    return f'Rejected {gate!r}; the run is stopped. Start a new run with a changed job file.'


# ── the stage machine ─────────────────────────────────────────────────────
def next_action(ctx: Ctx) -> tuple[str, object]:
    """What has to happen next: ('agent', stage) | ('execute', None) | ('wait', [tasks]) |
    ('approval', gate) | ('advanced', stage) | ('done', None) | ('stopped', reason)."""
    m = ctx.m
    st = m['stage']
    if st == 'done':
        return 'done', None
    if st == 'stopped':
        return 'stopped', m.get('stopped')
    if any(not r['done'] for r in m['requests']):
        return 'execute', None
    tasks = stage_tasks(m, st)
    running = [t for t in tasks if t['state'] not in TERMINAL]
    if running:
        return 'wait', running
    pend = m.get('pending_approval')
    if pend:
        return 'approval', pend['gate']
    blocking = [t for t in tasks if t['state'] != 'COMPLETED' and t['kind'] in
                ('inventory', 'static', 'offsets', 'stats')]
    if blocking:
        t = blocking[0]
        raise StopRun(f"Earth Engine task {t['name']} ended {t['state']}: {t.get('error')}. "
                      'Nothing later can run without it. Check the message, then start a new run.')
    for t in tasks:                                    # a failed stack or map drops just that pass
        if t['state'] != 'COMPLETED' and t['kind'] == 'wald':
            m.setdefault('wald_failures', {})[t['key']] = clip(t.get('error') or t['state'], 160)
        if t['state'] != 'COMPLETED' and t['kind'] in ('stack', 'map'):
            if t['key'] not in m['excluded']:
                m['excluded'][t['key']] = f"{t['kind']} task {t['state'].lower()}: {clip(t.get('error') or '', 160)}"
                ctx.audit.log('pass_excluded', date=t['key'], reason=m['excluded'][t['key']])
                ctx.save()
    agent_done = lambda s: m['agent_runs'].get(s, 0) > 0

    if st == 'inventory':
        if not tasks:
            launch(ctx, 'inventory', 'all', 'inventory', 'all')
            return 'wait', stage_tasks(m, st)
        finish_inventory(ctx)
        return 'advanced', m['stage']

    if st == 'plan':
        sel = m['selection']
        if not sel.get('selected'):
            return 'agent', st
        if ctx.job['gates']['approve_plan'] and 'plan' not in m['approvals']:
            if not pend:
                auto_request(ctx, 'plan')
            return 'approval', 'plan'
        set_stage(ctx, 'pilot_stacks', f"{len(sel['selected'])} passes selected; pilot {sel['pilot']}")
        return 'advanced', m['stage']

    if st in ('pilot_stacks', 'stacks', 'wald', 'maps', 'qa', 'pilot_models'):
        kinds = {'pilot_stacks': 1, 'stacks': 1, 'wald': 1, 'maps': 1, 'qa': 1, 'pilot_models': 2}[st]
        reqs = [r for r in m['requests'] if r['stage'] == st]
        scope = 'pilot' if st.startswith('pilot') else ('all' if st == 'qa' else 'rest')
        if st in ('stacks', 'wald', 'maps') and not scope_dates(m, 'rest'):
            reqs = reqs or [None]                       # every pass was in the pilot: nothing left to do
        if len(reqs) < kinds:
            return 'agent', st
        if st == 'pilot_stacks':
            set_stage(ctx, 'pilot_models', 'pilot stacks ready')
        elif st == 'pilot_models':
            finish_pilot_models(ctx)
        elif st == 'stacks':
            set_stage(ctx, 'wald', 'stacks ready')
        elif st == 'wald':
            m['wald'] = pooled_wald(ctx, selected(m))
            set_stage(ctx, 'model', f"pooled Wald: alpha {m['wald']['alpha']}, "
                                    f"{'passed' if m['wald']['passed'] else 'failed'}")
        elif st == 'maps':
            set_stage(ctx, 'qa', 'maps ready')
        elif st == 'qa':
            finish_qa(ctx)
        return 'advanced', m['stage']

    if st == 'spend':
        if m['spend'].get('approval_required') and 'spend' not in m['approvals']:
            if not agent_done('spend'):
                return 'agent', st
            auto_request(ctx, 'spend')
            return 'approval', 'spend'
        set_stage(ctx, 'stacks', 'spend within limits' if not m['spend'].get('approval_required') else 'spend approved')
        return 'advanced', m['stage']

    if st == 'model':
        if not agent_done('model'):
            return 'agent', st
        if not m['wald']['passed'] and 'model' not in m['approvals']:
            auto_request(ctx, 'model')
            return 'approval', 'model'
        set_stage(ctx, 'maps', 'model gate passed' if m['wald']['passed'] else 'model gate approved by you')
        return 'advanced', m['stage']

    if st == 'qa_review':
        v = m.get('qa_verdict')
        if not v:
            if m['agent_runs'].get(st, 0) < 2:
                return 'agent', st
            v = code_verdict(ctx)
        if v['verdict'] == 'fail' and 'qa' not in m['approvals']:
            auto_request(ctx, 'qa')
            return 'approval', 'qa'
        set_stage(ctx, 'stats', f"QA {v['verdict']}")
        return 'advanced', m['stage']

    if st == 'stats':
        if not [r for r in m['requests'] if r['stage'] == st]:
            return 'agent', st
        off = m['tasks'].get('offsets_all')
        if off and off['state'] == 'COMPLETED' and 'offsets' not in m:
            rows = read_table(ctx, off['out'])
            m['offsets'] = {'by_date': {r['date']: _num(r['offset']) for r in rows if _num(r.get('offset')) is not None},
                            'method': rows[0].get('method') if rows else None,
                            'reference_passes': sum(int(r.get('reference') or 0) for r in rows)}
            m.save_file('offsets.json', rows)
            dropped = [d for d in selected(m) if d not in m['offsets']['by_date']]
            for d in dropped:
                m['excluded'][d] = 'no valid pixels for an anomaly offset'
            launch(ctx, 'stats', 'all', 'stats', 'all', {'dates': [d for d in selected(m)]})
            ctx.save()
            return 'wait', stage_tasks(m, st)
        if m['tasks'].get('stats_all', {}).get('state') == 'COMPLETED':
            finish_stats(ctx)
            return 'advanced', m['stage']
        return 'wait', stage_tasks(m, st)

    if st == 'report':
        if not m.get('report'):
            if m['agent_runs'].get(st, 0) < 2:
                return 'agent', st
            write_report(ctx, '_The orchestrator did not write a summary; this report is generated from the run record._',
                         by='code')
        set_stage(ctx, 'done', 'report written')
        return 'done', None

    raise StopRun(f'Unknown stage {st!r}.')


def fallback(ctx: Ctx, stage: str, say=print):
    """What the code does when a stage's agent twice failed to act, so a run never stalls on an agent."""
    m = ctx.m
    ctx.audit.log('fallback', stage=stage)
    if stage == 'plan':
        sel = m['selection']
        if not sel.get('selected'):
            sel['selected'] = list(sel['eligible'])
            sel['note'] = 'All eligible passes (chosen by code; the data scout did not choose).'
    elif stage in ('pilot_stacks', 'stacks', 'pilot_models', 'wald', 'maps', 'qa', 'stats'):
        scope = 'pilot' if stage.startswith('pilot') else ('all' if stage in ('qa', 'stats') else 'rest')
        kinds = {'pilot_stacks': ['stacks'], 'stacks': ['stacks'], 'pilot_models': ['wald', 'maps'],
                 'wald': ['wald'], 'maps': ['maps'], 'qa': ['qa'], 'stats': ['stats']}[stage]
        for k in kinds:
            if check_tool(m, f'start_{k}', {'scope': scope}) is None:
                add_request(ctx, k, scope, by='code')
    elif stage == 'spend':
        auto_request(ctx, 'spend')
    elif stage == 'model':
        m.setdefault('reviews', {})['model'] = {'by': 'code', 'note': 'No agent review was recorded.'}
    elif stage == 'qa_review':
        code_verdict(ctx)
    elif stage == 'report':
        write_report(ctx, '_The orchestrator did not write a summary; this report is generated from the run record._',
                     by='code')
    m['agent_runs'][stage] = max(m['agent_runs'].get(stage, 0), 1)
    ctx.save()
    say(f'The agents did not complete stage {stage!r} after two tries; the code did it with default rules.')


def auto_request(ctx: Ctx, gate: str):
    """If an agent didn't file the approval request, the code files a plain one so you're never stuck."""
    if ctx.m.get('pending_approval') or gate in ctx.m['approvals']:
        return
    request_approval(ctx, gate, gate_summary(ctx, gate), by='code')


def gate_summary(ctx: Ctx, gate: str) -> str:
    m = ctx.m
    if gate == 'plan':
        s = m['selection']
        return (f"{len(s['selected'])} passes selected of {len(s['eligible'])} eligible "
                f"({s['summary']['passes_found']} found). Pilot passes: {', '.join(s['pilot'])}.")
    if gate == 'spend':
        sp = m['spend']
        return (f"Projected {sp.get('projected_eecu_hours')} EECU-hours and {sp.get('projected_storage_gb')} GB "
                f"for the remaining {sp['n_rest']} passes. Why approval: {'; '.join(sp['why'])}.")
    if gate == 'model':
        w = m['wald']
        return (f"The pooled Wald test did not pass (alpha {w['alpha']}, cross-check gains "
                f"{[c['gain'] for c in w.get('cross_check', [])]}). Approving makes 10 m maps anyway.")
    if gate == 'qa':
        v = m.get('qa_verdict') or {}
        return f"QA verdict: fail. Findings: {clip(v.get('findings', ''), 1500)}"
    return gate


def code_verdict(ctx: Ctx) -> dict:
    qa = ctx.m['qa']
    bad = qa['flagged']
    verdict = {'verdict': 'pass' if not bad or len(bad) <= 0.1 * max(1, qa['passes_checked']) else 'fail',
               'findings': f'{len(bad)} of {qa["passes_checked"]} passes flagged by the objective checks.',
               'excluded': {}, 'by': 'code', 'at': now_utc()}
    for d, flags in bad.items():
        ctx.m['excluded'][d] = 'QA: ' + '; '.join(flags)
        verdict['excluded'][d] = ctx.m['excluded'][d]
    ctx.m['qa_verdict'] = verdict
    ctx.save()
    return verdict


def write_report(ctx: Ctx, markdown: str, by: str) -> str:
    """Saves the agent's narrative plus a generated appendix with the exact numbers."""
    m = ctx.m
    w, qa, stats = m.get('wald') or {}, m.get('qa') or {}, m.get('stats') or {}
    lines = [markdown.strip(), '', '---', '', '## Appendix: numbers from the run record (generated)', '',
             f"- Run: `{ctx.run_id}` · job `{ctx.job['job']}` · created {m['created_utc']}",
             f"- Passes: {len(selected(m))} used, {len(m['excluded'])} excluded, "
             f"{len((m.get('selection') or {}).get('rejected', {}))} rejected by the job rules",
             f"- Alpha {w.get('alpha')} · Wald gate {'PASS' if w.get('passed') else 'FAIL'} · "
             f"overall held-out gain {w.get('gain_overall')}",
             f"- Earth Engine compute used: {eecu_hours_used(m)} EECU-hours · Claude spend ${m['agent_cost_usd']:.2f}",
             '', '### Wald cross-check', '', '| tuned on | tested on | alpha | RMSE K | no detail K | gain |',
             '|---|---|---|---|---|---|']
    for c in w.get('cross_check', []):
        lines.append(f"| {c['tuned_on']} | {c['tested_on']} | {c['alpha']} | {c['rmse_K']} | {c['no_detail_K']} "
                     f"| {c['gain']:+.1%} |")
    lines += ['', '### QA', '', f"- Passes checked: {qa.get('passes_checked')} · flagged: {len(qa.get('flagged', {}))}"
              f" · largest energy-check difference {qa.get('energy_max_K')} K",
              f"- Cross-check: {json.dumps(qa.get('crosscheck'))}", f"- Station: {json.dumps(qa.get('station'))}"]
    if stats:
        lines += ['', '### Per-pixel statistics (5th / 50th / 95th percentile over the area)', '',
                  '| band | p05 | p50 | p95 |', '|---|---|---|---|']
        for b, v in stats['summary'].items():
            if b != 'at_station':
                lines.append(f"| {b} | {v['p05']} | {v['p50']} | {v['p95']} |")
    lines += ['', '### Outputs', ''] + [f'- {k}: `{v}`' for k, v in m['outputs'].items()]
    if m['excluded']:
        lines += ['', '### Excluded passes', ''] + [f'- {d}: {why}' for d, why in sorted(m['excluded'].items())]
    text = '\n'.join(lines) + '\n'
    key = ctx.key('report.md')
    ctx.store.write_text(key, text, 'text/markdown')
    m['report'] = {'key': key, 'by': by, 'at': now_utc()}
    m['outputs']['report'] = ctx.f.gs_uri(key)
    ctx.save()
    return key
