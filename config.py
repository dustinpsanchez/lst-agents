"""Job files: defaults, loading and validation.

A job file is YAML. Anything it leaves out takes the default below, so the
example job only spells out what you are likely to change.
"""
from __future__ import annotations

import copy
import re

import yaml

DEFAULTS = {
    'job': None,                          # required: letters, digits and hyphens
    'area': {'center_lon': -98.49, 'center_lat': 29.42, 'side_km': 56.7},
    'crs': 'EPSG:32614',
    'years': [2017, 2025],
    'months': [5, 9],
    'landsat': {
        'path': 27,
        'min_seen': 0.98,                 # a pass must image this share of the area
        'min_clear': 0.20,                # and see this share of it cloud-free
        'reference_clear': 0.90,          # passes this clear define the anomaly reference pattern
    },
    'sentinel2': {'window_days': 3, 'fallback_days': 7, 'min_cover': 0.90, 'cloud_score': 0.60},
    'model': {
        'use_embeddings': True,
        'n_trees': 100,
        'min_leaf': 5,
        'n_train': 6000,
        'seed': 42,
        'interp': 'bicubic',
        'alphas': [round(0.1 * i, 1) for i in range(11)],
        'min_lst_valid': 0.98,
        'min_x_valid': 0.999,
    },
    'grid': {'fine_m': 10, 'coarse_m': 90, 'wald_factor': 9},
    'stats': {'early_through': 2021},     # early = passes up to this year, late = after it
    'pilot_passes': 2,
    'max_passes': None,
    'exclude_dates': [],
    'gates': {
        'approve_plan': True,
        'energy_tol_k': 0.1,
        'eecu_hours_over': 50,            # projected Earth Engine compute above this waits for you
        'storage_gb_over': 10,            # projected new asset storage above this waits for you
        'ee_monthly_eecu_hours': 150,     # your Earth Engine tier's monthly allowance (Community = 150)
        'agent_budget_usd': 5.0,          # cap on Claude spend for the whole run
        'session_budget_usd': 1.0,        # cap on any single agent session
    },
    'validation': {
        'station': {'name': 'SAT', 'lon': -98.4698, 'lat': 29.5337, 'lcd_units': 'F'},
        'crosscheck_path': 28,            # a second Landsat path compared at 90 m where it overlaps (null = off)
        'plausible_c': [0, 80],           # 10 m LST outside this range (°C) is flagged
        'max_detail_rms_k': 5.0,          # model detail stronger than this is flagged
        'min_valid_share': 0.01,          # a pass needs this share of the area at 10 m to count
    },
    'models': {'orchestrator': 'sonnet', 'data-scout': 'sonnet', 'feature-builder': 'haiku',
               'model-trainer': 'sonnet', 'inference-runner': 'haiku', 'qa-validator': 'sonnet'},
}

_NAME = re.compile(r'[a-z0-9][a-z0-9-]{0,29}')
_DATE = re.compile(r'\d{4}-\d{2}-\d{2}')
_MODELS = {'sonnet', 'haiku', 'opus', 'inherit'}


class JobError(ValueError):
    """A job file that can't be run, with every problem listed."""


def _merge(base, over, path=''):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k not in base:
            raise JobError(f'Unknown setting {path}{k!r}. Check the spelling against jobs/san_antonio_p27.yaml.')
        if v is None and k == 'station':
            out[k] = None
            continue
        if isinstance(base[k], dict) and base[k] and not isinstance(v, dict):
            raise JobError(f'{path}{k} should be a section with settings under it.')
        out[k] = _merge(base[k], v, f'{path}{k}.') if isinstance(base[k], dict) and base[k] else v
    return out


def load_job(text: str) -> dict:
    """Parses YAML text, fills in defaults and validates. Raises JobError listing every problem."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise JobError(f'The job is not valid YAML: {err}') from err
    if not isinstance(raw, dict):
        raise JobError('The job must be a YAML mapping (name: value lines).')
    job = _merge(DEFAULTS, raw)
    errors = validate(job)
    if errors:
        raise JobError('The job file has problems:\n  - ' + '\n  - '.join(errors))
    job['alphas'] = sorted({0.0, *[round(float(a), 2) for a in job['model']['alphas']]})
    return job


def _num(errors, label, v, lo=None, hi=None, integer=False):
    ok = isinstance(v, (int, float)) and not isinstance(v, bool) and (not integer or float(v).is_integer())
    if not ok or (lo is not None and v < lo) or (hi is not None and v > hi):
        rng = f' between {lo} and {hi}' if lo is not None and hi is not None else ''
        errors.append(f'{label} must be {"a whole number" if integer else "a number"}{rng} (got {v!r}).')


def validate(job: dict) -> list[str]:
    e: list[str] = []
    if not isinstance(job.get('job'), str) or not _NAME.fullmatch(job['job']):
        e.append("job: give the job a short name of lowercase letters, digits and hyphens, e.g. 'sa-p27'.")
    a = job['area']
    _num(e, 'area.center_lon', a['center_lon'], -180, 180)
    _num(e, 'area.center_lat', a['center_lat'], -80, 84)
    _num(e, 'area.side_km', a['side_km'], 1, 120)
    if not (isinstance(job['crs'], str) and re.fullmatch(r'EPSG:326\d\d|EPSG:327\d\d', job['crs'])):
        e.append('crs must be a UTM zone code such as EPSG:32614.')
    y, mo = job['years'], job['months']
    if not (isinstance(y, list) and len(y) == 2 and all(isinstance(v, int) for v in y) and 2013 <= y[0] <= y[1] <= 2100):
        e.append('years must be [first, last], e.g. [2017, 2025]; Landsat 8 starts in 2013.')
    if not (isinstance(mo, list) and len(mo) == 2 and all(isinstance(v, int) for v in mo) and 1 <= mo[0] <= mo[1] <= 12):
        e.append('months must be [first, last], e.g. [5, 9].')
    ls = job['landsat']
    _num(e, 'landsat.path', ls['path'], 1, 233, integer=True)
    for k in ('min_seen', 'min_clear', 'reference_clear'):
        _num(e, f'landsat.{k}', ls[k], 0, 1)
    s2 = job['sentinel2']
    _num(e, 'sentinel2.window_days', s2['window_days'], 0, 15, integer=True)
    _num(e, 'sentinel2.fallback_days', s2['fallback_days'], 0, 30, integer=True)
    if isinstance(s2['window_days'], int) and isinstance(s2['fallback_days'], int) and s2['fallback_days'] < s2['window_days']:
        e.append('sentinel2.fallback_days must be at least window_days.')
    _num(e, 'sentinel2.min_cover', s2['min_cover'], 0, 1)
    _num(e, 'sentinel2.cloud_score', s2['cloud_score'], 0, 1)
    m = job['model']
    _num(e, 'model.n_trees', m['n_trees'], 10, 500, integer=True)
    _num(e, 'model.min_leaf', m['min_leaf'], 1, 100, integer=True)
    _num(e, 'model.n_train', m['n_train'], 500, 50000, integer=True)
    _num(e, 'model.seed', m['seed'], 0, 2**31 - 1, integer=True)
    if m['interp'] not in ('bicubic', 'bilinear'):
        e.append("model.interp must be 'bicubic' or 'bilinear'.")
    if not (isinstance(m['alphas'], list) and m['alphas'] and
            all(isinstance(v, (int, float)) and 0 <= v <= 1.5 for v in m['alphas'])):
        e.append('model.alphas must be a list of numbers from 0 to 1.5.')
    _num(e, 'model.min_lst_valid', m['min_lst_valid'], 0.5, 1)
    _num(e, 'model.min_x_valid', m['min_x_valid'], 0.5, 1)
    g = job['grid']
    _num(e, 'grid.fine_m', g['fine_m'], 10, 30, integer=True)
    _num(e, 'grid.coarse_m', g['coarse_m'], 60, 120, integer=True)
    _num(e, 'grid.wald_factor', g['wald_factor'], 2, 12, integer=True)
    if not e and g['coarse_m'] % g['fine_m']:
        e.append('grid.coarse_m must be a whole multiple of grid.fine_m.')
    if not e:
        blocks = a['side_km'] * 1000 / (g['coarse_m'] * g['wald_factor'])
        if blocks < 20:
            e.append(f'The area holds only {blocks:.0f} Wald blocks across; make it larger or lower grid.wald_factor.')
    _num(e, 'stats.early_through', job['stats']['early_through'], 2013, 2100, integer=True)
    _num(e, 'pilot_passes', job['pilot_passes'], 1, 5, integer=True)
    if job['max_passes'] is not None:
        _num(e, 'max_passes', job['max_passes'], 1, 1000, integer=True)
    if not (isinstance(job['exclude_dates'], list) and all(isinstance(d, str) and _DATE.fullmatch(d)
                                                             for d in job['exclude_dates'])):
        e.append("exclude_dates must be a list of dates written 'YYYY-MM-DD'.")
    gt = job['gates']
    if not isinstance(gt['approve_plan'], bool):
        e.append('gates.approve_plan must be true or false.')
    _num(e, 'gates.energy_tol_k', gt['energy_tol_k'], 0.001, 1)
    _num(e, 'gates.eecu_hours_over', gt['eecu_hours_over'], 0, 100000)
    _num(e, 'gates.storage_gb_over', gt['storage_gb_over'], 0, 10000)
    _num(e, 'gates.ee_monthly_eecu_hours', gt['ee_monthly_eecu_hours'], 1, 1e6)
    _num(e, 'gates.agent_budget_usd', gt['agent_budget_usd'], 0.1, 50)
    _num(e, 'gates.session_budget_usd', gt['session_budget_usd'], 0.05, 10)
    v = job['validation']
    st = v['station']
    if st is not None:
        if not isinstance(st, dict) or not {'name', 'lon', 'lat'} <= set(st):
            e.append('validation.station needs name, lon and lat (or set it to null).')
        elif not re.fullmatch(r'[A-Za-z0-9_-]{1,20}', str(st['name'])):
            e.append('validation.station.name must be letters, digits, - or _.')
    if v['crosscheck_path'] is not None:
        _num(e, 'validation.crosscheck_path', v['crosscheck_path'], 1, 233, integer=True)
        if v['crosscheck_path'] == ls['path']:
            e.append('validation.crosscheck_path must differ from landsat.path.')
    pc = v['plausible_c']
    if not (isinstance(pc, list) and len(pc) == 2 and pc[0] < pc[1]):
        e.append('validation.plausible_c must be [low, high] in °C.')
    for k, model in job['models'].items():
        if model not in _MODELS:
            e.append(f"models.{k} must be one of {sorted(_MODELS)}.")
    return e
