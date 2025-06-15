import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from .base import ApproximatorBase

class SmoothingSplineApproximator(ApproximatorBase):
    """
    Временной аппроксиматор на основе сглаживающего сплайна.
    Использует UnivariateSpline из scipy для аппроксимации плотности во времени.
    Метрика качества — RMSE между предсказанием и обучающими точками (трактуется как разброс).
    """

    kind = 'temporal'

    def __init__(self, df, unique_coords, smoothing_factor=None):
        super().__init__()
        self.df = df
        self.unique_coords = unique_coords
        self.smoothing_factor = smoothing_factor
        self.grouped = df.copy()
        self.grouped['ordinal_day'] = self.grouped['date_only'].apply(lambda d: d.toordinal())
        self.coord_groups = self.grouped.groupby(['longitude', 'latitude'])

    def fit_spline(self, x, y):
        if len(x) < 4:
            return lambda t: y[0] if len(y) else None, None
        spline = UnivariateSpline(x, y, s=self.smoothing_factor)
        predictions = spline(x)
        rmse = float(np.sqrt(np.mean((y - predictions) ** 2)))
        return spline, rmse

    def approximate(self, target_date, coords_to_interpolate, known_points):
        results = {}
        day = pd.to_datetime(target_date).toordinal()
        known_set = {(p['longitude'], p['latitude']) for p in known_points}

        for coord in coords_to_interpolate:
            lon, lat = coord
            if coord in known_set:
                continue

            cached = self.get_from_cache(lon, lat, target_date)
            if cached is not None:
                results[coord] = cached
                continue

            try:
                group = self.coord_groups.get_group((lon, lat))
            except KeyError:
                continue

            if len(group) < 4:
                continue

            x = group['ordinal_day'].values
            y = group['density'].values
            spline, rmse = self.fit_spline(x, y)

            if spline is not None:
                prediction = float(spline(day))
                results[coord] = prediction
                self.set_to_cache(lon, lat, target_date, prediction)
                if rmse is not None:
                    self.set_quality_to_cache(lon, lat, target_date, rmse)

        return results

    def quality_metric(self, target_date, coord, known_points=None):
        lon, lat = coord
        cached = self.get_quality_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached

        if known_points is not None:
            _ = self.approximate(target_date, [coord], known_points)
            return self.get_quality_from_cache(lon, lat, target_date)

        return None
