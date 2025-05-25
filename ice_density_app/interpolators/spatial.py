import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from numpy.linalg import solve
from .base import InterpolatorBase

class KrigingInterpolator(InterpolatorBase):
    """
    Пространственный кригинг с ограничением числа соседей и радиуса.
    Метрика качества: кригинговская дисперсия (варианс).
    """
    kind = 'spatial'

    def __init__(self, sill=34.37, range_param=59.15, max_neighbors=10, max_radius_km=30):
        super().__init__()
        self.sill = sill
        self.range = range_param
        self.max_neighbors = max_neighbors
        self.max_radius_km = max_radius_km

    def variogram(self, h):
        return self.sill * (1 - np.exp(-h / self.range))

    @staticmethod
    def haversine_np(lon1, lat1, lon2, lat2):
        """
        Векторизированный расчёт расстояния в километрах между (lon1, lat1) и массивами (lon2, lat2)
        """
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c

    def interpolate(self, target_date, coords_to_interpolate, known_points):
        """
        Возвращает словарь {(lon, lat): density, ...} для всех координат.
        """
        if not known_points:
            return {}
        known_coords = np.array([[p['longitude'], p['latitude']] for p in known_points])
        known_values = np.array([p['density'] for p in known_points])
        known_lons = np.radians(known_coords[:, 0])
        known_lats = np.radians(known_coords[:, 1])
        interp_results = {}
        known_coords_set = {(p['longitude'], p['latitude']) for p in known_points}
        coords_to_interpolate = list(coords_to_interpolate)
        for coord in coords_to_interpolate:
            lon, lat = coord
            cached = self.get_from_cache(lon, lat, target_date)
            if cached is not None:
                interp_results[coord] = cached
                continue
            if coord in known_coords_set:
                continue
            clon, clat = np.radians(coord)
            dists = self.haversine_np(clon, clat, known_lons, known_lats)
            sorted_idx = np.argsort(dists)
            valid = sorted_idx[dists[sorted_idx] <= self.max_radius_km][:self.max_neighbors]
            if len(valid) < 3:
                continue
            coords_used = known_coords[valid]
            values_used = known_values[valid]
            n = len(coords_used)
            gamma = self.variogram(cdist(coords_used, coords_used))
            mat = np.zeros((n + 1, n + 1))
            mat[:n, :n] = gamma
            mat[n, :n] = 1
            mat[:n, n] = 1
            mat[n, n] = 0
            gamma_vec = self.variogram(cdist([coord], coords_used).flatten())
            rhs = np.append(gamma_vec, 1)
            try:
                weights = solve(mat, rhs)
                lambdas = weights[:-1]
                interp_value = float(np.dot(lambdas, values_used))
                self.set_to_cache(lon, lat, target_date, interp_value)
                interp_results[coord] = interp_value
                # Кэшируем качество интерполяции (варианс кригинга)
                kriging_variance = (
                    self.variogram(0)  # обычно = 0
                    - np.dot(weights, np.append(gamma_vec, 1))
                )
                self.set_quality_to_cache(lon, lat, target_date, float(kriging_variance))
            except np.linalg.LinAlgError:
                continue
        return interp_results

    def quality_metric(self, target_date, coord, known_points=None):
        """
        Возвращает кригинговскую дисперсию (variance) для точки.
        Если нет данных — возвращает None.
        """
        lon, lat = coord
        cached = self.get_quality_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached
        # Для эффективности — просто вызвать interpolate для одной точки (если надо)
        if known_points is not None:
            self.interpolate(target_date, [coord], known_points)
            return self.get_quality_from_cache(lon, lat, target_date)
        return None

class DelaunayInterpolator(InterpolatorBase):
    """
    Интерполяция через триангуляцию Делоне (линейно внутри треугольников).
    Метрика качества: расстояние до ближайшей вершины треугольника (или None, если вне).
    """
    kind = 'spatial'

    def __init__(self):
        super().__init__()
        self.triangulation_cache = {}

    def build_triangulation(self, known_points, target_date):
        if target_date in self.triangulation_cache:
            return self.triangulation_cache[target_date]
        points = np.array([[p['longitude'], p['latitude']] for p in known_points])
        if len(points) < 3:
            return None
        tri = Delaunay(points)
        self.triangulation_cache[target_date] = tri
        return tri

    def interpolate(self, target_date, coords_to_interpolate, known_points):
        if not known_points:
            return {}
        tri = self.build_triangulation(known_points, target_date)
        if tri is None:
            return {}
        points = np.array([[p['longitude'], p['latitude']] for p in known_points])
        values = np.array([p['density'] for p in known_points])
        interp_results = {}
        known_coords_set = {(p['longitude'], p['latitude']) for p in known_points}

        for coord in coords_to_interpolate:
            lon, lat = coord
            cached = self.get_from_cache(lon, lat, target_date)
            if cached is not None:
                interp_results[coord] = cached
                continue
            if coord in known_coords_set:
                continue
            simplex = tri.find_simplex(coord)
            if simplex == -1:
                # Вне выпуклой оболочки — метрика качества = расстояние до ближайшей известной точки
                min_dist = np.min(np.linalg.norm(points - coord, axis=1))
                self.set_quality_to_cache(lon, lat, target_date, float(min_dist))
                continue
            vertices = tri.simplices[simplex]
            transform = tri.transform[simplex]
            delta = coord - transform[2]
            bary = np.dot(transform[:2, :], delta)
            bary_coords = np.append(bary, 1 - bary.sum())
            if np.any(bary_coords < -1e-12):
                continue
            value = np.dot(bary_coords, values[vertices])
            self.set_to_cache(lon, lat, target_date, float(value))
            # Качество — максимальное расстояние до вершин треугольника
            max_dist = np.max(np.linalg.norm(points[vertices] - coord, axis=1))
            self.set_quality_to_cache(lon, lat, target_date, float(max_dist))
            interp_results[coord] = float(value)
        return interp_results

    def quality_metric(self, target_date, coord, known_points=None):
        """
        Возвращает максимальное расстояние до вершин треугольника (или до ближайшей точки вне оболочки).
        """
        lon, lat = coord
        cached = self.get_quality_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached
        if known_points is not None:
            self.interpolate(target_date, [coord], known_points)
            return self.get_quality_from_cache(lon, lat, target_date)
        return None
