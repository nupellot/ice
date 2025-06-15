import numpy as np
from scipy.fft import fft, ifft
import pandas as pd
from .base import ApproximatorBase

class FourierTemporalApproximator(ApproximatorBase):
    """
    Временной аппроксиматор на основе разложения по ряду Фурье.
    Метрика качества: RMSE между моделью и обучающими точками.
    Возвращает разброс в виде стандартного отклонения.
    """

    kind = 'temporal'

    def __init__(self, df, unique_coords, n_harmonics=10):
        super().__init__()
        self.df = df
        self.unique_coords = unique_coords
        self.n_harmonics = n_harmonics
        self.grouped = df.copy()
        self.grouped['ordinal_day'] = self.df['date_only'].apply(lambda d: d.toordinal())
        self.coord_groups = self.grouped.groupby(['longitude', 'latitude'])

    def fit_fourier_series(self, x, y):
        N = len(x)
        if N < 2:
            return lambda t: y[0] if len(y) else None, 0.0

        x_norm = np.linspace(0, 2 * np.pi, N)
        y_fft = fft(y)

        # Обрезаем до n_harmonics
        fft_filtered = np.zeros_like(y_fft)
        fft_filtered[:self.n_harmonics+1] = y_fft[:self.n_harmonics+1]
        ifft_func = ifft(fft_filtered).real

        def series_func(t_day):
            t_scaled = (t_day - x.min()) / (x.max() - x.min()) * 2 * np.pi
            t_idx = int((t_scaled / (2 * np.pi)) * N) % N
            return ifft_func[t_idx]

        rmse = float(np.sqrt(np.mean((ifft_func - y) ** 2)))
        return series_func, rmse

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

            if len(group) < 3:
                continue

            x = group['ordinal_day'].values
            y = group['density'].values
            model, rmse = self.fit_fourier_series(x, y)
            pred = model(day)
            if pred is not None:
                results[coord] = float(pred)
                self.set_to_cache(lon, lat, target_date, float(pred))
                self.set_quality_to_cache(lon, lat, target_date, float(rmse))

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
