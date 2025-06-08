import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from numpy.linalg import solve
from .base import ApproximatorBase

class KrigingInterpolator(ApproximatorBase):
    """
    Пространственный кригинг с экспоненциальной вариограммой, ограничением на радиус и число соседей.
    """
    kind = 'spatial'

    def __init__(self, sill=34.37, range_param=59.15, max_neighbors=10, max_radius_km=30):
        """
        :param sill: параметр плато (sill) вариограммы
        :param range_param: радиус автокорреляции
        :param max_neighbors: максимальное число соседей
        :param max_radius_km: радиус поиска соседей в километрах
        """
        super().__init__()
        self.sill = sill
        self.range = range_param
        self.max_neighbors = max_neighbors
        self.max_radius_km = max_radius_km

    # @property
    # def kind(self) -> str:
    #     return 'spatial'

    def variogram(self, h: np.ndarray) -> np.ndarray:
        """
        Экспоненциальная модель вариограммы.
        """
        return self.sill * (1 - np.exp(-h / self.range))

    @staticmethod
    def haversine_np(lon1, lat1, lon2, lat2):
        """
        Векторизированное расстояние (Haversine) между одной точкой и множеством.
        """
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c  # Радиус Земли в км

    def approximate(self, target_date, coords_to_approximate, known_points):
        """
        Интерполяция методом кригинга.
        """
        if not known_points:
            return {}

        known_coords = np.array([[p['longitude'], p['latitude']] for p in known_points])
        known_values = np.array([p['density'] for p in known_points])
        known_lons = np.radians(known_coords[:, 0])
        known_lats = np.radians(known_coords[:, 1])

        interp_results = {}
        known_coords_set = {(p['longitude'], p['latitude']) for p in known_points}

        for lon, lat in coords_to_approximate:
            if (lon, lat) in known_coords_set:
                continue

            cached = self.get_from_cache(lon, lat, target_date)
            if cached is not None:
                interp_results[(lon, lat)] = cached
                continue

            clon, clat = np.radians([lon, lat])
            dists = self.haversine_np(clon, clat, known_lons, known_lats)
            sorted_idx = np.argsort(dists)
            valid_idx = sorted_idx[dists[sorted_idx] <= self.max_radius_km][:self.max_neighbors]
            if len(valid_idx) < 3:
                continue

            coords_used = known_coords[valid_idx]
            values_used = known_values[valid_idx]

            gamma_matrix = self.variogram(cdist(coords_used, coords_used))
            n = len(coords_used)
            mat = np.zeros((n + 1, n + 1))
            mat[:n, :n] = gamma_matrix
            mat[n, :n] = mat[:n, n] = 1
            mat[n, n] = 0

            gamma_vec = self.variogram(cdist([[lon, lat]], coords_used).flatten())
            rhs = np.append(gamma_vec, 1)

            try:
                weights = solve(mat, rhs)
                lambdas = weights[:-1]
                interpolated = float(np.dot(lambdas, values_used))
                self.set_to_cache(lon, lat, target_date, interpolated)

                # Качество = кригинговская дисперсия
                kriging_variance = self.variogram(0) - np.dot(weights, np.append(gamma_vec, 1))
                self.set_quality_to_cache(lon, lat, target_date, float(kriging_variance))

                interp_results[(lon, lat)] = interpolated
            except np.linalg.LinAlgError:
                continue

        return interp_results

    def quality_metric(self, target_date, coord, known_points=None):
        """
        Возвращает дисперсию кригинга, если она уже вычислена.
        """
        lon, lat = coord
        cached = self.get_quality_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached

        if known_points is not None:
            self.approximate(target_date, [coord], known_points)
            return self.get_quality_from_cache(lon, lat, target_date)

        return None


class DelaunayInterpolator(ApproximatorBase):
    """
    Пространственная интерполяция по триангуляции Делоне.
    """
    kind = 'spatial'

    def __init__(self):
        super().__init__()
        self.triangulation_cache = {}

    # @property
    # def kind(self) -> str:
    #     return 'spatial'

    def build_triangulation(self, known_points, target_date):
        """
        Строит или достаёт из кэша триангуляцию для известного множества точек.
        """
        if target_date in self.triangulation_cache:
            return self.triangulation_cache[target_date]

        coords = np.array([[p['longitude'], p['latitude']] for p in known_points])
        if len(coords) < 3:
            return None

        tri = Delaunay(coords)
        self.triangulation_cache[target_date] = tri
        return tri

    def approximate(self, target_date, coords_to_approximate, known_points):
        """
        Возвращает значения для координат внутри треугольников Делоне.
        """
        if not known_points:
            return {}

        interp_results = {}
        coords_known = np.array([[p['longitude'], p['latitude']] for p in known_points])
        values_known = np.array([p['density'] for p in known_points])
        known_coords_set = {(p['longitude'], p['latitude']) for p in known_points}

        tri = self.build_triangulation(known_points, target_date)
        if tri is None:
            return {}

        for lon, lat in coords_to_approximate:
            if (lon, lat) in known_coords_set:
                continue

            cached = self.get_from_cache(lon, lat, target_date)
            if cached is not None:
                interp_results[(lon, lat)] = cached
                continue

            simplex = tri.find_simplex([lon, lat])
            if simplex == -1:
                # Вне треугольников — метрика качества = расстояние до ближайшей точки
                min_dist = np.min(np.linalg.norm(coords_known - [lon, lat], axis=1))
                self.set_quality_to_cache(lon, lat, target_date, float(min_dist))
                continue

            vertices = tri.simplices[simplex]
            transform = tri.transform[simplex]
            delta = np.array([lon, lat]) - transform[2]
            bary = np.dot(transform[:2, :], delta)
            bary_coords = np.append(bary, 1 - bary.sum())

            if np.any(bary_coords < -1e-12):
                continue

            value = float(np.dot(bary_coords, values_known[vertices]))
            self.set_to_cache(lon, lat, target_date, value)

            max_dist = float(np.max(np.linalg.norm(coords_known[vertices] - [lon, lat], axis=1)))
            self.set_quality_to_cache(lon, lat, target_date, max_dist)

            interp_results[(lon, lat)] = value

        return interp_results

    def quality_metric(self, target_date, coord, known_points=None):
        """
        Метрика качества — расстояние до ближайшей вершины (внутри) или ближайшей точки (вне).
        """
        lon, lat = coord
        cached = self.get_quality_from_cache(lon, lat, target_date)
        if cached is not None:
            return cached

        if known_points is not None:
            self.approximate(target_date, [coord], known_points)
            return self.get_quality_from_cache(lon, lat, target_date)

        return None
