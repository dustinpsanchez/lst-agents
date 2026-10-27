import math

from lst_agents import analysis
from lst_agents.analysis import cand_name
from lst_agents.grid import make_grid


def inv_row(d, clear=0.9, seen=1.0, cw=0.95, cf=0.97, sc='LANDSAT_8'):
    return {'date': d, 'spacecraft': sc, 'time_utc': '17:03:11.1', 'seen': seen, 'clear': clear,
            's2_cover_w': cw, 's2_cover_fb': cf}


def test_eligibility_rules(job):
    rows = [inv_row('2018-06-01'), inv_row('2018-06-17', clear=0.1), inv_row('2018-07-03', seen=0.5),
            inv_row('2018-07-19', cw=0.5, cf=0.95), inv_row('2018-08-04', cw=0.2, cf=0.3),
            inv_row('2025-08-10', sc='LANDSAT_9')]
    job['exclude_dates'] = ['2018-06-01']
    ok, rej = analysis.eligibility(rows, job, [2017, 2018, 2024])
    assert [p['date'] for p in ok] == ['2018-07-19', '2025-08-10']
    assert set(rej) == {'2018-06-01', '2018-06-17', '2018-07-03', '2018-08-04'}
    assert ok[0]['s2_window'] == 7 and 'Sentinel-2 needed ±7 days' in ok[0]['flags']
    assert ok[1]['emb_year'] == 2024 and ok[1]['flags'] == ['embedding year 2024']


def test_pilot_prefers_variety(job):
    rows = [inv_row('2018-06-01', 0.99), inv_row('2018-06-09', 0.98), inv_row('2023-07-01', 0.95, sc='LANDSAT_9')]
    ok, _ = analysis.eligibility(rows, job, [2018, 2023])
    assert analysis.pick_pilot(ok, 2, 3) == ['2018-06-01', '2023-07-01']


def wald_rows(date, n, mse_by_alpha, nofix):
    out = []
    for half in ('west', 'east'):
        r = {'date': date, 'half': half, 'n': n, 'mse_nofix': nofix}
        for a, v in mse_by_alpha[half].items():
            r[f'mse_{cand_name(a)}'] = v
        out.append(r)
    return out


def test_pool_wald_picks_alpha_and_cross_checks():
    alphas = [0.0, 0.2, 0.4, 0.6]
    curve = {'west': {0.0: 1.0, 0.2: 0.8, 0.4: 0.7, 0.6: 0.75}, 'east': {0.0: 1.2, 0.2: 1.0, 0.4: 0.9, 0.6: 0.95}}
    rows = wald_rows('2023-07-01', 1000, curve, 1.3) + wald_rows('2023-07-17', 1000, curve, 1.3)
    res = analysis.pool_wald(rows, alphas)
    assert res['alpha'] == 0.4 and res['passed']
    assert all(c['gain'] > 0 for c in res['cross_check'])
    assert math.isclose(res['table'][cand_name(0.4)]['west_K'], math.sqrt(0.7), rel_tol=1e-3)


def test_pool_wald_fails_when_detail_hurts():
    alphas = [0.0, 0.5, 1.0]
    curve = {'west': {0.0: 1.0, 0.5: 1.1, 1.0: 1.4}, 'east': {0.0: 1.0, 0.5: 1.2, 1.0: 1.5}}
    res = analysis.pool_wald(wald_rows('2023-07-01', 500, curve, 1.1), alphas)
    assert res['alpha'] == 0.0 and not res['passed']


def test_pool_wald_empty():
    res = analysis.pool_wald([], [0.0, 0.5])
    assert not res['passed'] and 'no scored pixels' in res['reason']


def test_spend_projection(job):
    grid = make_grid(job)
    pilot = {'eecu_s': {'stack': [1800, 1800], 'wald': [360, 360], 'map': [3600, 3600], 'static': [900]},
             'bytes': {'stack': [20e6, 20e6], 'map': [60e6, 60e6], 'static': [110e6]},
             'secs': {'stack': [600, 600], 'wald': [300, 300], 'map': [1200, 1200]}}
    sp = analysis.spend_projection(pilot, n_rest=100, static_years_rest=7, n_all=102, grid=grid, job=job)
    # per pass 1.6 h; 100 × 1.6 + 7 × 0.25 + 102 × 0.5 = 212.75 h
    assert sp['eecu_hours_per_pass'] == 1.6
    assert sp['projected_eecu_hours'] == 212.8
    assert sp['approval_required'] and 'over the job' in sp['why'][0]
    small = analysis.spend_projection({'eecu_s': {'stack': [10], 'wald': [10], 'map': [10]},
                                       'bytes': {'stack': [1e6], 'map': [1e6]}}, 3, 0, 5, grid, job)
    assert not small['approval_required']
    unknown = analysis.spend_projection({'eecu_s': {'stack': [None]}, 'bytes': {}}, 3, 0, 5, grid, job)
    assert unknown['approval_required'] and 'unknown' in ' '.join(unknown['why'])


def test_qa_judgement(job):
    rows = [{'date': 'a', 'energy_max_K': 0.01, 'lst_p001_C': 20, 'lst_p999_C': 60, 'detail_rms_K': 1.0,
             'valid_share': 0.5},
            {'date': 'b', 'energy_max_K': 0.3, 'lst_p001_C': -5, 'lst_p999_C': 90, 'detail_rms_K': 7.0,
             'valid_share': 0.001},
            {'date': 'c', 'energy_max_K': None}]
    out = analysis.qa_judgement(rows, job)
    assert out['a']['ok'] and not out['b']['ok'] and len(out['b']['flags']) == 5
    assert out['c']['flags'] == ['no valid blocks to check']


def test_station_compare_and_time():
    assert analysis.station_compare([(40, 30), (45, 33), (50, 36)])['r'] == 1.0
    assert abs(analysis.frac_year('2021-07-02') - 2021.5) < 0.01
