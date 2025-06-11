import numpy as np
import pandas as pd
from collections import defaultdict
from .base import ApproximatorBase

class FourierTemporalApproximator(ApproximatorBase):
    """
    Временной аппроксиматор, основанный на аппроксимации функции плотности
    льда в каждой точке с помощью конечного ряда Фурье.
    """
    kind = 'temporal'

    def __init__(self, df: pd.DataFrame, unique_coords: set, harmonics: int = 3):
        """
        :param df: DataFrame с колонками ['date_only', 'longitude', 'latitude', 'density']
        :param unique_coords: множество всех уникальных координат (lon, lat)
        :param harmonics: количество гармоник в ряде Фурье (обычно 2–5)
        """
        super().__init__()
        self.df = df
        self.unique_coords = unique_coords
        self.harmonics = harmonics
        self.coefficients_cache = {}  # (lon, lat) -> (a0, [a_k], [b_k])
        self._prepare()

    def _day_of_year(self, date: pd.Timestamp) -> float:
        """
        Преобразует дату в день года (1–365).
        """
        return date.timetuple().tm_yday

    def _prepare(self):
        """
        Строит кэш коэффициентов Фурье для всех точек.
        """
        grouped = self.df.groupby(['longitude', 'latitude'])
        for (lon, lat), group in grouped:
            if len(group) < 2 * self.harmonics + 1:
                # Недостаточно данных
                continue

            # Получаем значения времени и плотности
            t = group['date_only'].apply(self._day_of_year).values.astype(float)
            y = group['density'].values.astype(float)
            T = 365.0

            # Формируем матрицу признаков X:
            X = np.ones((len(t), 1 + 2 * self.harmonics))
            for k in range(1, self.harmonics + 1):
                X[:, 2 * k - 1] = np.cos(2 * np.pi * k * t / T)
                X[:, 2 * k] = np.sin(2 * np.pi * k * t / T)

            # Решаем задачу МНК: X @ beta ≈ y
            try:
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                a0 = beta[0]
                a_coeffs = beta[1::2]
                b_coeffs = beta[2::2]
                self.coefficients_cache[(lon, lat)] = (a0, a_coeffs, b_coeffs)
            except np.linalg.LinAlgError:
                continue  # пропускаем если не решается

    def interpolate_point(self, lon: float, lat: float, target_date) -> float | None:
        """
        Вычисляет аппроксимированное значение по ряду Фурье.
        """
        cached = self.get_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached

        coeffs = self.coefficients_cache.get((lon, lat))
        if coeffs is None:
            return None  # Нет данных для точки

        a0, a, b = coeffs
        t = self._day_of_year(target_date)
        T = 365.0
        result = a0
        for k in range(len(a)):
            result += a[k] * np.cos(2 * np.pi * (k + 1) * t / T)
            result += b[k] * np.sin(2 * np.pi * (k + 1) * t / T)

        self.set_to_cache(lon, lat, target_date, float(result))
        return float(result)

    def approximate(self, target_date, coords_to_interpolate, known_points):
        """
        Вычисляет аппроксимированные значения для всех координат.
        """
        interp_results = {}
        known_coords = {(p['longitude'], p['latitude']) for p in known_points}
        for coord in coords_to_interpolate:
            if coord in known_coords:
                continue
            value = self.interpolate_point(coord[0], coord[1], target_date)
            if value is not None:
                interp_results[coord] = value
        return interp_results

    def quality_metric(self, target_date, coord, known_points=None) -> float | None:
        """
        Качество аппроксимации — среднеквадратичное отклонение от ряда Фурье
        на обучающих точках (если есть).
        """
        lon, lat = coord
        cached = self.get_quality_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached

        group = self.df[(self.df['longitude'] == lon) & (self.df['latitude'] == lat)]
        if group.empty:
            return None
        actual = group['density'].values
        predicted = np.array([
            self.interpolate_point(lon, lat, d)
            for d in group['date_only']
        ])
        if None in predicted:
            return None
        error = float(np.sqrt(np.mean((predicted - actual) ** 2)))
        self.set_quality_to_cache(lon, lat, target_date, error)
        return error
