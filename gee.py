"""Earth Engine side: datasets, predictors, and the batch tasks each stage starts.

The method is the one that passed Phase 1:
    10 m LST = conserve(smooth(T90) + alpha × model detail)
Because conserve() only adds one constant per 90 m block, this splits into
    10 m LST = Sc + alpha × Dc
where Sc is energy-conserving bicubic of the observed 90 m LST and Dc is the
model's detail with each block's mean removed. Each pass stores only Dc (int16,
hundredths of a kelvin); Sc is rebuilt from the 90 m stack whenever it's needed.
So alpha can be chosen once, from every pass's Wald test, after the maps exist.
"""
from __future__ import annotations

import ee

from .fences import pass_name

LANDSAT_IDS = ('LANDSAT/LC08/C02/T1_L2', 'LANDSAT/LC09/C02/T1_L2')
S2_ID = 'COPERNICUS/S2_SR_HARMONIZED'
CS_ID = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'
EMB_ID = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
DEM_ID = 'COPERNICUS/DEM/GLO30_2024_1'
WC_ID = 'ESA/WorldCover/v200'

S2_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
INDICES = ['ndvi', 'ndbi', 'mndwi', 'albedo']
PASS_FEATURES = S2_BANDS + INDICES
EMB_BANDS = [f'A{i:02d}' for i in range(64)]
TERRAIN = ['elev', 'slope']
LC_CLASSES = {'lc_tree': 10, 'lc_shrub': 20, 'lc_grass': 30, 'lc_crop': 40,
              'lc_built': 50, 'lc_bare': 60, 'lc_water': 80}
QA_MASK = 0b11111          # fill, dilated cloud, cirrus, cloud, cloud shadow
BIG = int(1e10)


def static_features(job) -> list[str]:
    return (EMB_BANDS if job['model']['use_embeddings'] else []) + TERRAIN + list(LC_CLASSES)


def features(job) -> list[str]:
    return PASS_FEATURES + static_features(job)


# ── small helpers (same as Phase 1) ───────────────────────────────────────
def all_valid(img):
    """1 where every band has data, 0 elsewhere; never masked, so it averages into a coverage share."""
    return img.mask().reduce(ee.Reducer.min()).gte(1).unmask(0).rename('valid')


def to_coarse(img, coarse_p):
    """Area-weighted block mean of an image on a finer grid."""
    return img.reduceResolution(ee.Reducer.mean(), maxPixels=1024).reproject(coarse_p)


def fill_coarse(img, coarse_p):
    """Fills masked coarse pixels from neighbours so interpolation doesn't eat into valid blocks."""
    return img.unmask(img.focalMean(radius=2, kernelType='square', units='pixels'), False).reproject(coarse_p)


def prep_landsat(img):
    """Surface temperature in kelvin; fill, cloud, cirrus and shadow masked."""
    st = img.select('ST_B10')
    clear = img.select('QA_PIXEL').bitwiseAnd(QA_MASK).eq(0)
    return st.multiply(0.00341802).add(149.0).rename('LST').updateMask(clear.And(st.gt(0)))


class Geo:
    """Every Earth Engine object for one run: grids, region, halves and datasets."""

    def __init__(self, job: dict, grid: dict):
        self.job, self.g = job, grid
        self.crs = grid['crs']
        x0, y0, x1, y1, xm = (grid[k] for k in ('x0', 'y0', 'x1', 'y1', 'xm'))
        self.region = ee.Geometry.Rectangle([x0, y0, x1, y1], self.crs, False)
        self.halves = {'west': ee.Geometry.Rectangle([x0, y0, xm, y1], self.crs, False),
                       'east': ee.Geometry.Rectangle([xm, y0, x1, y1], self.crs, False)}
        self.fine_p = ee.Projection(self.crs, grid['fine_t'])
        self.coarse_p = ee.Projection(self.crs, grid['coarse_t'])
        self.wald_p = ee.Projection(self.crs, grid['wald_t'])
        m = job['model']
        self.mlv, self.mxv = m['min_lst_valid'], m['min_x_valid']
        self.features = features(job)
        self.static = static_features(job)

    def kw(self, transform_key='coarse_t', tile=4):
        return dict(crs=self.crs, crsTransform=self.g[transform_key], maxPixels=BIG, tileScale=tile)

    # ── datasets ──────────────────────────────────────────────────────────
    def landsat(self, path):
        y0, y1 = self.job['years']
        m0, m1 = self.job['months']
        return (ee.ImageCollection(LANDSAT_IDS[0]).merge(ee.ImageCollection(LANDSAT_IDS[1]))
                .filterBounds(self.region)
                .filterDate(f'{y0}-01-01', f'{y1 + 1}-01-01')
                .filter(ee.Filter.calendarRange(m0, m1, 'month'))
                .filter(ee.Filter.eq('PROCESSING_LEVEL', 'L2SP'))
                .filter(ee.Filter.eq('WRS_PATH', int(path))))

    def landsat_day(self, path, date):
        """One pass (all rows on that date) as LST in kelvin on its native 30 m grid."""
        day = self.landsat(path).filter(ee.Filter.eq('DATE_ACQUIRED', date))
        proj = ee.Image(day.first()).select('ST_B10').projection()
        return day.map(prep_landsat).mosaic().setDefaultProjection(proj), proj, day

    def s2(self, date, window):
        d = ee.Date(date)
        thr = self.job['sentinel2']['cloud_score']
        return (ee.ImageCollection(S2_ID).filterBounds(self.region)
                .filterDate(d.advance(-window, 'day'), d.advance(window + 1, 'day'))
                .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80))
                .linkCollection(ee.ImageCollection(CS_ID), ['cs_cdf'])
                .map(lambda im: im.updateMask(im.select('cs_cdf').gte(thr)).select(S2_BANDS).divide(10000)))

    def s2_cover(self, date, window):
        """(number of Sentinel-2 images, share of the area with at least one clear look)."""
        col = self.s2(date, window).select('B4')
        dummy = ee.ImageCollection([ee.Image.constant(0).toFloat().rename('B4').updateMask(ee.Image.constant(0))])
        seen = col.merge(dummy).count().select([0]).gt(0).unmask(0).rename('cov')
        cov = seen.reduceRegion(ee.Reducer.mean(), self.region, **self.kw('check_t')).get('cov')
        return col.size(), cov

    def embedding_years(self):
        ts = ee.ImageCollection(EMB_ID).filterBounds(self.region).aggregate_array('system:time_start')
        return ts.map(lambda t: ee.Date(t).get('year')).distinct().sort()

    # ── predictors ────────────────────────────────────────────────────────
    def s2_features(self, date, window):
        """The 10 per-pass predictors on the 10 m grid: six bands and four indices."""
        r = self.s2(date, window).median()
        idx = ee.Image.cat([
            r.normalizedDifference(['B8', 'B4']).rename('ndvi'),
            r.normalizedDifference(['B11', 'B8']).rename('ndbi'),
            r.normalizedDifference(['B3', 'B11']).rename('mndwi'),
            r.expression('0.356*B2 + 0.130*B4 + 0.373*B8 + 0.085*B11 + 0.072*B12 - 0.0018',
                         {b: r.select(b) for b in ['B2', 'B4', 'B8', 'B11', 'B12']}).rename('albedo')])
        return ee.Image.cat([r, idx]).reproject(self.fine_p).select(PASS_FEATURES).toFloat()

    def static_x(self, emb_year):
        """Predictors that don't change between passes, on the 10 m grid."""
        dem_col = ee.ImageCollection(DEM_ID).filterBounds(self.region).select('DEM')
        dem = dem_col.mosaic().setDefaultProjection(ee.Image(dem_col.first()).projection()).rename('elev')
        terrain = ee.Image.cat([dem, ee.Terrain.slope(dem).rename('slope')]).resample('bilinear')
        wc = ee.Image(ee.ImageCollection(WC_ID).first()).select('Map')
        lc = ee.Image.cat([wc.eq(v).rename(k) for k, v in LC_CLASSES.items()])
        parts = [terrain.reproject(self.fine_p), lc.reproject(self.fine_p)]
        if self.job['model']['use_embeddings']:
            emb = (ee.ImageCollection(EMB_ID).filterBounds(self.region)
                   .filterDate(f'{emb_year}-01-01', f'{emb_year + 1}-01-01').mosaic())
            parts.insert(0, emb.reproject(self.fine_p))
        return ee.Image.cat(parts).select(self.static).toFloat()

    # ── batch products ────────────────────────────────────────────────────
    def inventory_fc(self):
        """One row per pass on the job's path: how much is imaged, how much is clear, Sentinel-2 match."""
        col = self.landsat(self.job['landsat']['path'])
        s2 = self.job['sentinel2']
        region, kw = self.region, self.kw('check_t')

        def per_date(ds):
            ds = ee.String(ds)
            day = col.filter(ee.Filter.eq('DATE_ACQUIRED', ds))
            first = ee.Image(day.first())
            seen = day.select('ST_B10').mosaic().gt(0).unmask(0)
            clear = seen.And(day.select('QA_PIXEL').mosaic().bitwiseAnd(QA_MASK).eq(0)).unmask(0)
            share = (ee.Image.cat([seen.rename('seen'), clear.rename('clear')])
                     .reduceRegion(ee.Reducer.mean(), region, **kw))
            nw, cw = self.s2_cover(ds, s2['window_days'])
            nf, cf = self.s2_cover(ds, s2['fallback_days'])
            return ee.Feature(None, {
                'date': ds, 'spacecraft': first.get('SPACECRAFT_ID'), 'time_utc': first.get('SCENE_CENTER_TIME'),
                'path': first.get('WRS_PATH'), 'rows': day.aggregate_array('WRS_ROW'),
                'scene_cloud': day.aggregate_mean('CLOUD_COVER'),
                'seen': share.get('seen'), 'clear': share.get('clear'),
                's2_n_w': nw, 's2_cover_w': cw, 's2_n_fb': nf, 's2_cover_fb': cf})

        return ee.FeatureCollection(col.aggregate_array('DATE_ACQUIRED').distinct().map(per_date))

    def static_stack(self, emb_year):
        """Static predictors as 90 m block means, plus the share of each block where all of them exist."""
        X = self.static_x(emb_year)
        return ee.Image.cat([to_coarse(X, self.coarse_p),
                             to_coarse(all_valid(X), self.coarse_p).rename('static_frac')]).toFloat()

    def pass_stack(self, p):
        """Per-pass 90 m stack: Sentinel-2 predictors, observed LST (K) and coverage shares."""
        path = self.job['landsat']['path']
        lst30, proj, _ = self.landsat_day(path, p['date'])
        X = self.s2_features(p['date'], p['s2_window'])
        cp = self.coarse_p
        return ee.Image.cat([
            to_coarse(X, cp),
            to_coarse(lst30.reproject(proj), cp).rename('T'),
            to_coarse(all_valid(lst30).reproject(proj), cp).rename('lst_frac'),
            to_coarse(all_valid(X), cp).rename('s2_frac'),
        ]).toFloat().set({'date': p['date'], 's2_window': p['s2_window'], 'spacecraft': p.get('spacecraft') or ''})

    def x90(self, stack, static):
        """All predictors at 90 m and the observed LST, masked to blocks that pass the coverage rules."""
        X = ee.Image.cat([stack.select(PASS_FEATURES), static.select(self.static)])
        xv = stack.select('s2_frac').gte(self.mxv).And(static.select('static_frac').gte(self.mxv))
        good = stack.select('lst_frac').gte(self.mlv).And(xv)
        return X.updateMask(xv), stack.select('T').updateMask(good)

    def train_rf(self, X, T, proj):
        m = self.job['model']
        samples = (X.addBands(T.rename('T'))
                   .sample(region=self.region, projection=proj, numPixels=m['n_train'], seed=m['seed'],
                           dropNulls=True, tileScale=4))
        rf = (ee.Classifier.smileRandomForest(numberOfTrees=m['n_trees'], minLeafPopulation=m['min_leaf'],
                                              seed=m['seed'])
              .setOutputMode('REGRESSION').train(samples, 'T', self.features))
        return rf, samples

    def smooth_up(self, img_coarse, fine_p, coarse_p):
        return fill_coarse(img_coarse, coarse_p).resample(self.job['model']['interp']).reproject(fine_p)

    def detail_parts(self, rf, X_fine, fine_p, coarse_p):
        """The model on the finer grid and its detail inside each coarse block (block means removed)."""
        model = X_fine.classify(rf, 'model').reproject(fine_p)
        D = model.subtract(self.smooth_up(to_coarse(model, coarse_p), fine_p, coarse_p))
        Dc = D.subtract(to_coarse(D, coarse_p).reproject(fine_p))
        return model, Dc

    def conserved_smooth(self, T_coarse, fine_p, coarse_p):
        """Energy-conserving bicubic: smooth surface plus one constant per block (alpha 0)."""
        S = self.smooth_up(T_coarse, fine_p, coarse_p)
        return S.add(T_coarse.subtract(to_coarse(S, coarse_p)).reproject(fine_p))

    def wald_fc(self, p, stack, static, alphas):
        """Wald test for one pass: degrade 90 m → 810 m, downscale back, score every alpha on each half."""
        from .analysis import cand_name
        X90, T90 = self.x90(stack, static)
        cp, wp = self.coarse_p, self.wald_p
        good_w = (to_coarse(all_valid(T90).reproject(cp), wp).gte(self.mlv)
                  .And(to_coarse(all_valid(X90).reproject(cp), wp).gte(self.mxv)))
        T_w = to_coarse(T90, wp).updateMask(good_w)
        X_w = to_coarse(X90, wp).updateMask(good_w)
        rf_w, samples = self.train_rf(X_w, T_w, wp)
        _, Dc = self.detail_parts(rf_w, X90, cp, wp)
        S = self.smooth_up(T_w, cp, wp)
        Sc = self.conserved_smooth(T_w, cp, wp)
        cands = {'nofix': S}
        for a in alphas:
            cands[cand_name(a)] = Sc.add(Dc.multiply(a))
        names = list(cands)
        err = ee.Image.cat([img.subtract(T90).rename(c) for c, img in cands.items()])
        sq = err.updateMask(all_valid(err)).pow(2)
        red = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
        feats = []
        for h, geom in self.halves.items():
            out = ee.Dictionary(sq.reduceRegion(red, geom, **self.kw()))
            props = {'date': p['date'], 'half': h, 'n': out.get(f'{names[0]}_count'), 'n_train': samples.size()}
            for c in names:
                props[f'mse_{c}'] = out.get(f'{c}_mean')
            feats.append(ee.Feature(None, props))
        return ee.FeatureCollection(feats)

    def detail_image(self, p, stack, static):
        """Model detail at 10 m for one pass, int16 in hundredths of a kelvin, block means removed."""
        X90, T90 = self.x90(stack, static)
        rf, _ = self.train_rf(X90, T90, self.coarse_p)
        X10 = ee.Image.cat([self.s2_features(p['date'], p['s2_window']), self.static_x(p.get('emb_year'))])
        _, Dc = self.detail_parts(rf, X10, self.fine_p, self.coarse_p)
        Dc = Dc.updateMask(T90.mask().reproject(self.fine_p))
        return Dc.multiply(100).round().toInt16().rename('detail_cK').set({
            'date': p['date'], 'units': 'hundredths of a kelvin',
            'note': '10 m LST = energy-conserving bicubic of T + alpha x detail_cK / 100'})

    def lst10(self, stack, static, detail, alpha):
        """10 m LST in kelvin for one pass at a given alpha."""
        _, T90 = self.x90(stack, static)
        Sc = self.conserved_smooth(T90, self.fine_p, self.coarse_p)
        return Sc.add(detail.select('detail_cK').toFloat().multiply(alpha / 100.0)).rename('LST')

    def qa_fc(self, items, alpha, station_xy):
        """QA numbers for a batch of passes: energy check, value range, detail strength, station value."""
        feats = []
        n_fine = self.g['n_fine'] ** 2
        pt = ee.Geometry.Point(list(station_xy), self.crs) if station_xy else None
        pct = ee.Reducer.percentile([0.1, 50, 99.9], ['p001', 'p50', 'p999']).combine(
            ee.Reducer.count(), sharedInputs=True)
        for p, stack, static, detail in items:
            _, T90 = self.x90(stack, static)
            lst = self.lst10(stack, static, detail, alpha)
            energy = to_coarse(lst, self.coarse_p).subtract(T90).abs().rename('energy')
            d = {
                'energy': energy.reduceRegion(ee.Reducer.max(), self.region, **self.kw()),
                'lst': lst.subtract(273.15).rename('lst').reduceRegion(pct, self.region, **self.kw('fine_t', 8)),
                'det': detail.select('detail_cK').toFloat().divide(100).pow(2).rename('det2')
                .reduceRegion(ee.Reducer.mean(), self.region, **self.kw('fine_t', 8)),
                'T': T90.reduceRegion(ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
                                      self.region, **self.kw()),
            }
            if pt is not None:
                d['station'] = lst.subtract(273.15).rename('station').reduceRegion(
                    ee.Reducer.first(), pt, crs=self.crs, crsTransform=self.g['fine_t'])
            props = {'date': p['date'], 'n_fine_total': n_fine}
            flat = {'energy': ('energy', 'energy'), 'lst_p001': ('lst', 'lst_p001'), 'lst_p50': ('lst', 'lst_p50'),
                    'lst_p999': ('lst', 'lst_p999'), 'lst_count': ('lst', 'lst_count'), 'det2': ('det', 'det2'),
                    'T_mean': ('T', 'T_mean'), 'T_count': ('T', 'T_count'), 'station': ('station', 'station')}
            for name, (group, key) in flat.items():
                if group in d:
                    props[name] = ee.Dictionary(d[group]).get(key, -9999)
            for k, v in d.items():          # the raw dictionaries too (as JSON text), in case a key is named differently
                props[f'raw_{k}'] = ee.String.encodeJSON(v)
            feats.append(ee.Feature(None, props))
        return ee.FeatureCollection(feats)

    def offsets_fc(self, items, ref_dates):
        """Per-pass offset for anomalies. With enough clear reference passes, the offset is the pass's
        mean difference from the reference pattern, so a half-cloudy pass isn't biased by which half is clear."""
        T = {p['date']: self.x90(stack, static)[1] for p, stack, static in items}
        stat = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)
        amean = {d: ee.Dictionary(t.reduceRegion(stat, self.region, **self.kw())) for d, t in T.items()}
        refs = [d for d in ref_dates if d in T]
        feats = []
        if len(refs) >= 3:
            C = ee.ImageCollection([T[d].subtract(ee.Number(amean[d].get('T_mean'))) for d in refs]).mean()
        else:
            C = None
        for d, t in T.items():
            props = {'date': d, 'area_mean': amean[d].get('T_mean'), 'n_valid': amean[d].get('T_count'),
                     'reference': int(d in refs), 'method': 'reference pattern' if C is not None else 'area mean'}
            if C is not None:
                o = ee.Dictionary(t.subtract(C).reduceRegion(stat, self.region, **self.kw()))
                props.update({'offset': o.get('T_mean'), 'n_offset': o.get('T_count')})
            else:
                props.update({'offset': amean[d].get('T_mean'), 'n_offset': amean[d].get('T_count')})
            feats.append(ee.Feature(None, props))
        return ee.FeatureCollection(feats)

    def stats_image(self, items, offsets, alpha, frac_years, early_through):
        """Per-pixel statistics of the per-pass anomalies at 10 m, int16 in hundredths of a kelvin."""
        imgs = []
        for p, stack, static, detail in items:
            a = self.lst10(stack, static, detail, alpha).subtract(offsets[p['date']]).rename('a')
            t = ee.Image.constant(frac_years[p['date']] - 2020).toFloat().rename('t').updateMask(a.mask())
            imgs.append(ee.Image.cat([t, a]).set('year', int(p['date'][:4])))
        col = ee.ImageCollection(imgs)
        A = col.select('a')
        parts = [A.count().rename('n_looks'),
                 A.mean().multiply(100).rename('mean_cK'),
                 A.reduce(ee.Reducer.stdDev()).multiply(100).rename('sd_cK'),
                 col.select(['t', 'a']).reduce(ee.Reducer.linearFit()).select('scale')
                 .multiply(1000).rename('trend_cK_per_decade')]
        years = {int(p['date'][:4]) for p, *_ in items}
        if any(y <= early_through for y in years) and any(y > early_through for y in years):
            early = col.filter(ee.Filter.lte('year', early_through)).select('a')
            late = col.filter(ee.Filter.gt('year', early_through)).select('a')
            parts += [early.count().rename('n_early'), early.mean().multiply(100).rename('early_cK'),
                      late.count().rename('n_late'), late.mean().multiply(100).rename('late_cK')]
        return ee.Image.cat(parts).round().toInt16().set({
            'units': 'hundredths of a kelvin (n_* are counts)', 'alpha': alpha,
            'anomaly': 'pixel minus the pass offset (reference-pattern method)',
            'early_through': early_through})

    def crosscheck_fc(self, p27_items, other_path):
        """Compares the 90 m mean-anomaly pattern of the job's path with another path where they overlap.
        Independent dates, same time of day; checks that the anomaly pattern is reproducible."""
        cp, region, kw = self.coarse_p, self.region, self.kw()
        col = self.landsat(other_path)
        mlv = self.mlv

        def one(ds):
            ds = ee.String(ds)
            day = col.filter(ee.Filter.eq('DATE_ACQUIRED', ds))
            proj = ee.Image(day.first()).select('ST_B10').projection()
            lst30 = day.map(prep_landsat).mosaic().setDefaultProjection(proj)
            seen30 = day.select('ST_B10').mosaic().gt(0).unmask(0).setDefaultProjection(proj)
            T = to_coarse(lst30.reproject(proj), cp)
            frac = to_coarse(all_valid(lst30).reproject(proj), cp)
            seen = to_coarse(seen30.reproject(proj), cp)
            return T.updateMask(frac.gte(mlv)).rename('T').addBands(seen.gte(0.98).rename('seen'))

        other = ee.ImageCollection(col.aggregate_array('DATE_ACQUIRED').distinct().map(one))
        strip = other.select('seen').mean().gte(0.5).selfMask().rename('strip')
        n_strip = ee.Number(strip.reduceRegion(ee.Reducer.count(), region, **kw).get('strip'))
        stat = ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True)

        def anomaly(T):
            T = ee.Image(T).select('T').updateMask(strip)
            s = ee.Dictionary(T.reduceRegion(stat, region, **kw))
            cnt = ee.Number(ee.Algorithms.If(s.get('T_count'), s.get('T_count'), 0))
            mean = ee.Number(ee.Algorithms.If(s.get('T_mean'), s.get('T_mean'), 0))
            return T.subtract(mean).rename('a').set('share', cnt.divide(n_strip.max(1)))

        a_other = other.map(anomaly).filter(ee.Filter.gte('share', 0.6)).select('a')
        stacks = [stack.select('T').updateMask(stack.select('lst_frac').gte(mlv)) for _, stack in p27_items]
        a_main = ee.ImageCollection([anomaly(t) for t in stacks]).filter(ee.Filter.gte('share', 0.6)).select('a')
        both = (a_main.mean().rename('m_main').addBands(a_other.mean().rename('m_other'))
                .updateMask(a_main.count().gte(5).And(a_other.count().gte(5)).And(strip)))
        r = both.reduceRegion(ee.Reducer.pearsonsCorrelation(), region, **kw)
        diff = both.select('m_main').subtract(both.select('m_other'))
        dstat = ee.Image.cat([diff.pow(2).rename('d2'), diff.abs().rename('ad')]).reduceRegion(
            ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True), region, **kw)
        r, dstat = ee.Dictionary(r), ee.Dictionary(dstat)
        return ee.FeatureCollection([ee.Feature(None, {
            'other_path': other_path, 'strip_pixels': n_strip,
            'passes_main': a_main.size(), 'passes_other': a_other.size(),
            'r': r.get('correlation', -9999), 'msd': dstat.get('d2_mean', -9999),
            'mad': dstat.get('ad_mean', -9999), 'n_pixels': dstat.get('d2_count', -9999),
            'raw_r': ee.String.encodeJSON(r), 'raw_diff': ee.String.encodeJSON(dstat)})])


# ── exports and task bookkeeping ──────────────────────────────────────────
def export_image(img, description, asset_id, region, crs, transform, pyramiding='mean'):
    task = ee.batch.Export.image.toAsset(
        image=img, description=description[:100], assetId=asset_id, region=region,
        crs=crs, crsTransform=transform, maxPixels=BIG, pyramidingPolicy={'.default': pyramiding})
    task.start()
    return task.id


def export_table(fc, description, bucket, key_prefix):
    task = ee.batch.Export.table.toCloudStorage(
        collection=fc, description=description[:100], bucket=bucket,
        fileNamePrefix=key_prefix, fileFormat='GeoJSON')
    task.start()
    return task.id


def task_states(ids: list[str]) -> dict[str, dict]:
    """Status of many tasks with one listing call; falls back to single lookups for any not listed."""
    from ee import _cloud_api_utils as cau
    want = set(ids)
    out = {}
    if not want:
        return out
    try:
        for op in ee.data.listOperations():
            t = cau.convert_operation_to_task(op)
            if t.get('id') in want:
                out[t['id']] = t
    except Exception:          # listing can fail on very large histories; single lookups still work
        pass
    missing = [i for i in want if i not in out]
    for i in range(0, len(missing), 50):
        for t in ee.data.getTaskStatus(missing[i:i + 50]):
            out[t['id']] = t
    return out


def asset_exists(aid: str) -> bool:
    try:
        ee.data.getAsset(aid)
        return True
    except ee.EEException:
        return False


def asset_size(aid: str) -> int | None:
    try:
        a = ee.data.getAsset(aid)
    except ee.EEException:
        return None
    v = a.get('sizeBytes')
    return int(v) if v is not None else None


def ensure_container(aid: str, kind: str = 'FOLDER'):
    if not asset_exists(aid):
        ee.data.createAsset({'type': kind}, aid)


def pass_asset(fences, run_id: str, folder: str, date: str) -> str:
    prefix = {'stack90': 's', 'detail10': 'd'}[folder]
    return fences.asset_id(run_id, folder, f'{prefix}_{pass_name(date)}')
