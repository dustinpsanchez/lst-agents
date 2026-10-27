"""Pure calculations: pass rules, Wald pooling and alpha choice, spend projection, QA rules.

No Earth Engine or network calls here, so all of it is unit-tested.
"""
from __future__ import annotations

import math


def cand_name(alpha: float) -> str:
    """Column name for one alpha in the Wald tables: 0.4 → a040."""
    return f'a{round(alpha * 100):03d}'


# ── Which passes are usable ──────────────────────────────────────────────
def eligibility(rows: list[dict], job: dict, emb_years: list[int]) -> tuple[list[dict], dict]:
    """Applies the job's rules to the inventory. Returns (eligible passes, {date: reason} for the rest)."""
    ls, s2 = job['landsat'], job['sentinel2']
    excluded = set(job['exclude_dates'])
    ok, rejected = [], {}
    for r in sorted(rows, key=lambda r: r['date']):
        d = r['date']
        seen, clear = r.get('seen') or 0.0, r.get('clear') or 0.0
        cov_w, cov_fb = r.get('s2_cover_w') or 0.0, r.get('s2_cover_fb') or 0.0
        if d in excluded:
            rejected[d] = 'excluded in the job file'
            continue
        if seen < ls['min_seen']:
            rejected[d] = f'imaged only {seen:.0%} of the area'
            continue
        if clear < ls['min_clear']:
            rejected[d] = f'only {clear:.0%} of the area clear'
            continue
        flags = []
        if cov_w >= s2['min_cover']:
            window, cover = s2['window_days'], cov_w
        elif cov_fb >= s2['min_cover']:
            window, cover = s2['fallback_days'], cov_fb
            flags.append(f"Sentinel-2 needed ±{s2['fallback_days']} days")
        else:
            rejected[d] = (f"Sentinel-2 covers only {max(cov_w, cov_fb):.0%} of the area "
                           f"within ±{s2['fallback_days']} days")
            continue
        year = int(d[:4])
        emb_year = None
        if job['model']['use_embeddings']:
            if not emb_years:
                rejected[d] = 'no Satellite Embedding year available'
                continue
            emb_year = max([y for y in emb_years if y <= year] or [min(emb_years)])
            if emb_year != year:
                flags.append(f'embedding year {emb_year}')
        ok.append({'date': d, 'year': year, 'spacecraft': r.get('spacecraft'), 'time_utc': r.get('time_utc'),
                   'clear': round(clear, 3), 's2_window': window, 's2_cover': round(cover, 3),
                   'emb_year': emb_year, 'flags': flags})
    if job['max_passes'] and len(ok) > job['max_passes']:
        keep = {p['date'] for p in sorted(ok, key=lambda p: -p['clear'])[: job['max_passes']]}
        for p in ok:
            if p['date'] not in keep:
                rejected[p['date']] = f"beyond max_passes ({job['max_passes']}); less clear than the rest"
        ok = [p for p in ok if p['date'] in keep]
    return ok, rejected


def pick_pilot(passes: list[dict], n: int, primary_window: int) -> list[str]:
    """The clearest passes with a same-week Sentinel-2 match, preferring different satellites and years."""
    order = sorted(passes, key=lambda p: (p['s2_window'] != primary_window, -p['clear'], p['date']))
    picked: list[dict] = []
    for p in order:
        if len(picked) == n:
            break
        if picked and any(q['spacecraft'] == p['spacecraft'] or q['year'] == p['year'] for q in picked):
            continue
        picked.append(p)
    for p in order:                                  # top up if variety ran out
        if len(picked) == n:
            break
        if p not in picked:
            picked.append(p)
    return sorted(p['date'] for p in picked)


def summarize_inventory(rows: list[dict], eligible: list[dict], rejected: dict) -> dict:
    by_year: dict[int, dict] = {}
    for r in rows:
        y = int(r['date'][:4])
        b = by_year.setdefault(y, {'passes': 0, 'eligible': 0})
        b['passes'] += 1
    for p in eligible:
        by_year[p['year']]['eligible'] += 1
    reasons: dict[str, int] = {}
    for why in rejected.values():
        key = why.split(' (')[0]
        key = 'too cloudy' if 'clear' in key else ('Sentinel-2 gap' if 'Sentinel-2' in key else key)
        reasons[key] = reasons.get(key, 0) + 1
    clear = sorted(p['clear'] for p in eligible)
    return {
        'passes_found': len(rows), 'eligible': len(eligible), 'rejected': len(rejected),
        'rejected_by_reason': reasons,
        'by_year': {str(k): v for k, v in sorted(by_year.items())},
        'clear_share_of_eligible': ({'min': clear[0], 'median': clear[len(clear) // 2], 'max': clear[-1]}
                                    if clear else None),
        'wide_s2_window': sum(1 for p in eligible if any('Sentinel-2' in f for f in p['flags'])),
        'other_embedding_year': sum(1 for p in eligible if any('embedding' in f for f in p['flags'])),
        'by_spacecraft': {s: sum(1 for p in eligible if p['spacecraft'] == s)
                          for s in sorted({p['spacecraft'] for p in eligible if p['spacecraft']})},
    }


# ── Wald test: pool every pass, choose one alpha, cross-check both halves ─
def pool_wald(rows: list[dict], alphas: list[float]) -> dict:
    """rows: one per pass and half with n (pixels scored) and mse_<candidate>. Returns the pooled verdict."""
    cands = ['nofix'] + [cand_name(a) for a in alphas]
    sse = {h: {c: 0.0 for c in cands} for h in ('west', 'east')}
    n = {'west': 0, 'east': 0}
    per_pass: dict[str, dict] = {}
    for r in rows:
        h, k = r['half'], int(r.get('n') or 0)
        if h not in n or k == 0 or any(r.get(f'mse_{c}') is None for c in cands):
            continue
        n[h] += k
        pp = per_pass.setdefault(r['date'], {'n': 0, **{c: 0.0 for c in cands}})
        pp['n'] += k
        for c in cands:
            v = float(r[f'mse_{c}']) * k
            sse[h][c] += v
            pp[c] += v
    if not n['west'] or not n['east']:
        return {'passed': False, 'alpha': 0.0, 'reason': 'no scored pixels in one half of the area',
                'n_eval': n, 'n_passes': len(per_pass)}
    rm = lambda s, k: math.sqrt(s / k)
    table = {c: {'west_K': rm(sse['west'][c], n['west']), 'east_K': rm(sse['east'][c], n['east']),
                 'both_K': rm(sse['west'][c] + sse['east'][c], n['west'] + n['east'])} for c in cands}
    a_names = [cand_name(a) for a in alphas]
    zero = cand_name(0.0)
    cross = []
    for tune, test in (('west', 'east'), ('east', 'west')):
        best = min(a_names, key=lambda c: table[c][f'{tune}_K'])
        cross.append({'tuned_on': tune, 'tested_on': test, 'alpha': alphas[a_names.index(best)],
                      'rmse_K': table[best][f'{test}_K'], 'no_detail_K': table[zero][f'{test}_K'],
                      'gain': 1 - table[best][f'{test}_K'] / table[zero][f'{test}_K']})
    best_both = min(a_names, key=lambda c: table[c]['both_K'])
    alpha = alphas[a_names.index(best_both)]
    passes = {}
    for d, pp in sorted(per_pass.items()):
        r0, ra = rm(pp[zero], pp['n']), rm(pp[best_both], pp['n'])
        passes[d] = {'n': pp['n'], 'rmse_alpha_K': round(ra, 4), 'rmse_no_detail_K': round(r0, 4),
                     'gain': round(1 - ra / r0, 4) if r0 > 0 else 0.0}
    hurt = sorted((d for d, v in passes.items() if v['gain'] < -0.05), key=lambda d: passes[d]['gain'])
    return {
        'alpha': alpha, 'passed': alpha > 0 and all(c['gain'] > 0 for c in cross),
        'table': {c: {k: round(v, 4) for k, v in t.items()} for c, t in table.items()},
        'cross_check': [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in cross],
        'n_eval': n, 'n_passes': len(per_pass), 'per_pass': passes,
        'passes_where_detail_hurts': hurt,
        'gain_overall': round(1 - table[best_both]['both_K'] / table[zero]['both_K'], 4),
    }


# ── Spend: project the full run from the pilot ───────────────────────────
def spend_projection(pilot: dict, n_rest: int, static_years_rest: int, n_all: int, grid: dict, job: dict) -> dict:
    """pilot: {'eecu_s': {kind: [seconds per task]}, 'bytes': {kind: [bytes per asset]},
    'secs': {kind: [run seconds per task]}}. Kinds: static, stack, wald, map."""
    gates = job['gates']

    def mean(kind, field):
        vals = [v for v in pilot.get(field, {}).get(kind, []) if v is not None]
        return sum(vals) / len(vals) if vals else None

    per = {k: mean(k, 'eecu_s') for k in ('static', 'stack', 'wald', 'map')}
    known = all(per[k] is not None for k in ('stack', 'wald', 'map')) and (per['static'] is not None or not static_years_rest)
    out = {'n_rest': n_rest, 'static_years_rest': static_years_rest, 'n_all': n_all,
           'eecu_hours_per_pass': None, 'projected_eecu_hours': None,
           'notes': ['QA and statistics are estimated at half a map export per pass; actual use is tracked as it runs.']}
    if known:
        per_pass = per['stack'] + per['wald'] + per['map']
        allowance = 0.5 * per['map']
        total_s = n_rest * per_pass + static_years_rest * (per['static'] or 0) + n_all * allowance
        out['eecu_hours_per_pass'] = round(per_pass / 3600, 3)
        out['projected_eecu_hours'] = round(total_s / 3600, 1)
        out['months_at_tier'] = round(total_s / 3600 / gates['ee_monthly_eecu_hours'], 2)
    else:
        out['notes'].append('Earth Engine did not report compute use for every pilot task, so the total is unknown.')
    b = {k: mean(k, 'bytes') for k in ('static', 'stack', 'map')}
    stats_bytes = grid['n_fine'] ** 2 * 8 * 2                 # 8 int16 bands, before compression
    if b['stack'] is not None and b['map'] is not None:
        gb = (n_rest * (b['stack'] + b['map']) + static_years_rest * (b['static'] or 0) + stats_bytes) / 1e9
        out['projected_storage_gb'] = round(gb, 2)
    else:
        out['projected_storage_gb'] = None
        out['notes'].append('Asset sizes were not available, so storage is unknown.')
    t = {k: mean(k, 'secs') for k in ('stack', 'wald', 'map')}
    if all(v is not None for v in t.values()):
        out['hours_if_2_tasks_run_at_once'] = round(n_rest * sum(t.values()) / 2 / 3600, 1)
    reasons = []
    if out['projected_eecu_hours'] is None:
        reasons.append('compute use unknown')
    elif out['projected_eecu_hours'] > gates['eecu_hours_over']:
        reasons.append(f"projected {out['projected_eecu_hours']} EECU-hours is over the job's "
                       f"{gates['eecu_hours_over']}")
    if out['projected_storage_gb'] is None:
        reasons.append('storage unknown')
    elif out['projected_storage_gb'] > gates['storage_gb_over']:
        reasons.append(f"projected {out['projected_storage_gb']} GB is over the job's {gates['storage_gb_over']} GB")
    out['approval_required'] = bool(reasons)
    out['why'] = reasons
    return out


# ── QA rules: objective checks the code applies to every pass ────────────
def qa_judgement(rows: list[dict], job: dict) -> dict:
    """rows: one per pass from the QA tables. Returns {date: {'ok': bool, 'flags': [...], **numbers}}."""
    v, tol = job['validation'], job['gates']['energy_tol_k']
    lo, hi = v['plausible_c']
    out = {}
    for r in rows:
        flags = []
        e = r.get('energy_max_K')
        if e is None:
            flags.append('no valid blocks to check')
        elif e > tol:
            flags.append(f'energy check {e:.3f} K > {tol} K')
        p_lo, p_hi = r.get('lst_p001_C'), r.get('lst_p999_C')
        if p_lo is not None and p_lo < lo:
            flags.append(f'0.1th percentile {p_lo:.1f} °C below {lo} °C')
        if p_hi is not None and p_hi > hi:
            flags.append(f'99.9th percentile {p_hi:.1f} °C above {hi} °C')
        rms = r.get('detail_rms_K')
        if rms is not None and rms > v['max_detail_rms_k']:
            flags.append(f'model detail RMS {rms:.2f} K above {v["max_detail_rms_k"]} K')
        share = r.get('valid_share')
        if share is not None and share < v['min_valid_share']:
            flags.append(f'only {share:.1%} of the area has 10 m output')
        out[r['date']] = {'ok': not flags, 'flags': flags,
                          **{k: r.get(k) for k in ('energy_max_K', 'lst_p001_C', 'lst_p50_C', 'lst_p999_C',
                                                   'detail_rms_K', 'valid_share', 'station_lst_C')}}
    return out


# ── Station: nearest hourly air temperature to each overpass ─────────────
def lcd_air_temp(df, when_local, units: str = 'F', max_minutes: int = 60):
    """Nearest hourly dry-bulb temperature (NOAA LCD, local standard time). Returns (°C, time) or None."""
    import pandas as pd
    date_col = next((c for c in df.columns if c.upper() == 'DATE'), None)
    temp_col = next((c for c in df.columns if 'hourlydrybulbtemperature' in c.lower().replace('_', '')), None)
    if date_col is None or temp_col is None:
        raise ValueError('Could not find the DATE and HourlyDryBulbTemperature columns in the LCD file.')
    t = pd.to_datetime(df[date_col], errors='coerce')
    vals = pd.to_numeric(df[temp_col].astype(str).str.extract(r'(-?\d+(?:\.\d+)?)')[0], errors='coerce')
    ok = t.notna() & vals.notna()
    t, vals = t[ok], vals[ok]
    if t.empty:
        return None
    gap = (t - pd.Timestamp(when_local)).abs()
    i = gap.idxmin()
    if gap[i] > pd.Timedelta(minutes=max_minutes):
        return None
    val = float(vals[i])
    return ((val - 32) * 5 / 9 if units.upper() == 'F' else val), t[i]


def station_compare(pairs: list[tuple[float, float]]) -> dict | None:
    """pairs of (LST °C, air °C). Correlation and mean difference; not a gate."""
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if len(pairs) < 3:
        return {'n': len(pairs)} if pairs else None
    xs, ys = zip(*pairs)
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in pairs)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return {'n': len(pairs), 'r': round(sxy / (sx * sy), 3) if sx and sy else None,
            'mean_lst_minus_air_K': round(mx - my, 2)}


def frac_year(date: str) -> float:
    import datetime as dt
    d = dt.date.fromisoformat(date)
    start = dt.date(d.year, 1, 1)
    return d.year + (d - start).days / ((dt.date(d.year + 1, 1, 1) - start).days)
