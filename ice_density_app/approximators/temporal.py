import pandas as pd
from .base import ApproximatorBase

class TemporalInterpolator(ApproximatorBase):
    """
    Временной интерполятор, аппроксимирующий плотность по ближайшим во времени измерениям.
    Качество аппроксимации выражается в днях до ближайшего измерения,
    которое линейно преобразуется в разброс через коэффициент alpha.
    """

    kind = 'temporal'

    def __init__(self, df, unique_coords, alpha=0.2):
        super().__init__()
        self.df = df
        self.alpha = alpha
        self.unique_coords = unique_coords
        self.grouped = df.sort_values('date_only').groupby(['longitude', 'latitude'])

    def interpolate_point(self, lon, lat, target_date):
        cached = self.get_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached

        try:
            group = self.grouped.get_group((lon, lat))
        except KeyError:
            return None

        before = group[group['date_only'] < target_date]
        after = group[group['date_only'] > target_date]
        prev_row = before.iloc[-1] if not before.empty else None
        next_row = after.iloc[0] if not after.empty else None

        if prev_row is not None and next_row is not None:
            t0 = prev_row['date_only']
            t1 = next_row['date_only']
            dens0 = prev_row['density']
            dens1 = next_row['density']
            total_days = (t1 - t0).days
            frac = (target_date - t0).days / total_days if total_days > 0 else 0
            density = dens0 + (dens1 - dens0) * frac
        elif prev_row is not None:
            density = prev_row['density']
        elif next_row is not None:
            density = next_row['density']
        else:
            return None

        self.set_to_cache(lon, lat, target_date, float(density))
        return float(density)

    def approximate(self, target_date, coords_to_interpolate, known_points):
        results = {}
        known = {(p['longitude'], p['latitude']) for p in known_points}
        for coord in coords_to_interpolate:
            if coord in known:
                continue
            val = self.interpolate_point(coord[0], coord[1], target_date)
            if val is not None:
                results[coord] = val
        return results

    def quality_metric(self, target_date, coord, known_points=None):
        lon, lat = coord
        cached = self.get_quality_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached

        try:
            group = self.grouped.get_group((lon, lat))
        except KeyError:
            return None

        if group.empty:
            return None

        group_dates = pd.to_datetime(group['date_only'])
        min_days = abs((group_dates - pd.to_datetime(target_date)).dt.days).min()
        stddev = float(self.alpha * min_days)
        self.set_quality_to_cache(lon, lat, target_date, stddev)
        return stddev
