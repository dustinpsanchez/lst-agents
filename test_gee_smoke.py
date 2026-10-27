"""Builds every Earth Engine graph against a stand-in, to catch Python mistakes before they cost compute."""
from lst_agents import gee
from lst_agents.grid import make_grid
from conftest import FakeEE


def test_every_builder_runs(job, monkeypatch):
    fake = FakeEE()
    monkeypatch.setattr(gee, 'ee', fake)
    geo = gee.Geo(job, make_grid(job))
    p = {'date': '2023-07-14', 's2_window': 3, 'emb_year': 2023, 'spacecraft': 'LANDSAT_9', 'clear': 0.9}
    S, T, D = FakeEE('stack'), FakeEE('static'), FakeEE('detail')
    geo.inventory_fc()
    geo.static_stack(2023)
    geo.pass_stack(p)
    geo.wald_fc(p, S, T, job['alphas'])
    geo.detail_image(p, S, T)
    geo.qa_fc([(p, S, T, D), (dict(p, date='2023-07-30'), S, T, D)], 0.4, (549000.0, 3260000.0))
    geo.offsets_fc([(p, S, T), (dict(p, date='2023-07-30'), S, T), (dict(p, date='2023-08-15'), S, T)],
                   ['2023-07-14', '2023-07-30', '2023-08-15'])
    geo.offsets_fc([(p, S, T)], [])
    geo.stats_image([(p, S, T, D), (dict(p, date='2019-07-30'), S, T, D)], {'2023-07-14': 310.0,
                                                                          '2019-07-30': 309.0},
                    0.4, {'2023-07-14': 2023.53, '2019-07-30': 2019.58}, 2021)
    geo.crosscheck_fc([(p, S)], 28)
    assert len(fake._log) > 200
    assert len(gee.features(job)) == 83


def test_no_embeddings(job, monkeypatch):
    monkeypatch.setattr(gee, 'ee', FakeEE())
    job['model']['use_embeddings'] = False
    geo = gee.Geo(job, make_grid(job))
    geo.static_stack(None)
    assert len(gee.features(job)) == 19
