# core/approximators/spline.py

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from .base import ApproximatorBase

class SplineTemporalApproximator(ApproximatorBase):
    """
    Временной аппроксиматор с использованием сглаживающего кубического сплайна
    через SciPy's UnivariateSpline для данных в каждой точке.
    """
    kind = 'temporal'

    def __init__(self, df: pd.DataFrame, unique_coords: set, smoothing_factor: float = None, k: int = 3):
        """
        :param df: DataFrame с полями ['date_only', 'longitude', 'latitude', 'density']
        :param unique_coords: множестов координат (lon, lat)
        :param smoothing_factor: параметр s (smoothing_factor в UnivariateSpline)
                                  если None — автоподбор (GCV-подход)
        :param k: степень сплайна (1 ≤ k ≤ 5), обычно 3 (кубический)
        """
        super().__init__()
        self.df = df
        self.unique_coords = unique_coords
        self.s = smoothing_factor
        self.k = k
        self.splines: dict[tuple[float, float], UnivariateSpline] = {}
        self._prepare_splines()

    def _prepare_splines(self):
        """
        Построение одного UnivariateSpline для каждой точки.
        Работает по всем координатам, для которых есть ≥ k+1 точек.
        """
        grouped = self.df.groupby(['longitude', 'latitude'])
        for coord, grp in grouped:
            # для корректной работы splines нужно не менее k+1 наблюдений
            if len(grp) < self.k + 1:
                continue

            dates = pd.to_datetime(grp['date_only'])
            x = dates.map(lambda d: d.timetuple().tm_yday).values.astype(float)
            y = grp['density'].values.astype(float)

            # создаём сплайн
            try:
                spline = UnivariateSpline(x, y, k=self.k, s=self.s)
            except Exception:
                continue

            self.splines[coord] = spline

    def interpolate_point(self, lon, lat, target_date) -> float | None:
        """
        Получение значения из сплайна, либо из кеша.
        """
        cached = self.get_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached

        spline = self.splines.get((lon, lat))
        if not spline:
            return None

        t = pd.to_datetime(target_date).timetuple().tm_yday
        try:
            v = float(spline(t))
        except ValueError:
            return None

        self.set_to_cache(lon, lat, target_date, v)
        return v

    def approximate(self, target_date, coords_to_interpolate, known_points):
        """
        Интерполяция по набору координат, исключая известные real_points.
        """
        results = {}
        known = { (p['longitude'], p['latitude']) for p in known_points }
        for coord in coords_to_interpolate:
            if coord in known:
                continue
            val = self.interpolate_point(coord[0], coord[1], target_date)
            if val is not None:
                results[coord] = val
        return results

    def quality_metric(self, target_date, coord, known_points=None) -> float | None:
        """
        Использует среднеквадратичную ошибку (RMSE) на обучающих данных точки.
        """
        lon, lat = coord
        cached_q = self.get_quality_from_cache(lon, lat, target_date)
        if cached_q is not None:
            return cached_q

        spline = self.splines.get((lon, lat))
        if not spline:
            return None

        grp = self.df[(self.df.longitude == lon) & (self.df.latitude == lat)]
        dates = pd.to_datetime(grp['date_only'])
        x = dates.map(lambda d: d.timetuple().tm_yday).values.astype(float)
        y = grp['density'].values.astype(float)

        preds = spline(x)
        rmse = float(np.sqrt(np.mean((preds - y)**2)))
        self.set_quality_to_cache(lon, lat, target_date, rmse)
        return rmse
