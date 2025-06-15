import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from numpy.linalg import solve
from .base import ApproximatorBase

import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import solve
from .base import ApproximatorBase

class KrigingInterpolator(ApproximatorBase):
    """
    Пространственный интерполятор на основе Кригинга.
    Возвращает оценку значения и дисперсию интерполяции, которая трактуется как разброс (sqrt(var)).
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
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c

    def approximate(self, target_date, coords_to_interpolate, known_points):
        if not known_points:
            return {}
        known_coords = np.array([[p['longitude'], p['latitude']] for p in known_points])
        known_values = np.array([p['density'] for p in known_points])
        known_lons = np.radians(known_coords[:, 0])
        known_lats = np.radians(known_coords[:, 1])
        known_set = {(p['longitude'], p['latitude']) for p in known_points}

        results = {}
        for coord in coords_to_interpolate:
            lon, lat = coord
            if coord in known_set:
                continue
            cached = self.get_from_cache(lon, lat, target_date)
            if cached is not None:
                results[coord] = cached
                continue

            clon, clat = np.radians(coord)
            dists = self.haversine_np(clon, clat, known_lons, known_lats)
            sorted_idx = np.argsort(dists)
            valid_idx = sorted_idx[dists[sorted_idx] <= self.max_radius_km][:self.max_neighbors]
            if len(valid_idx) < 3:
                continue

            coords_used = known_coords[valid_idx]
            values_used = known_values[valid_idx]
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
                prediction = float(np.dot(lambdas, values_used))
                self.set_to_cache(lon, lat, target_date, prediction)
                results[coord] = prediction

                kriging_variance = self.variogram(0) - np.dot(weights, rhs)
                stddev = float(np.sqrt(max(kriging_variance, 0)))
                self.set_quality_to_cache(lon, lat, target_date, stddev)
            except np.linalg.LinAlgError:
                continue
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


class DelaunayInterpolator(ApproximatorBase):
    """
    Пространственный аппроксиматор на основе триангуляции Делоне.
    Метрика качества — максимальное расстояние до вершины треугольника, интерпретируемое как разброс.
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

    def approximate(self, target_date, coords_to_interpolate, known_points):
        if not known_points:
            return {}

        tri = self.build_triangulation(known_points, target_date)
        if tri is None:
            return {}

        points = np.array([[p['longitude'], p['latitude']] for p in known_points])
        values = np.array([p['density'] for p in known_points])
        known_set = {(p['longitude'], p['latitude']) for p in known_points}

        results = {}
        for coord in coords_to_interpolate:
            lon, lat = coord
            if coord in known_set:
                continue

            cached = self.get_from_cache(lon, lat, target_date)
            if cached is not None:
                results[coord] = cached
                continue

            simplex = tri.find_simplex(coord)
            if simplex == -1:
                min_dist = float(np.min(np.linalg.norm(points - coord, axis=1)))
                self.set_quality_to_cache(lon, lat, target_date, min_dist)
                continue

            vertices = tri.simplices[simplex]
            transform = tri.transform[simplex]
            delta = coord - transform[2]
            bary = np.dot(transform[:2, :], delta)
            bary_coords = np.append(bary, 1 - bary.sum())
            if np.any(bary_coords < -1e-8):
                continue

            value = np.dot(bary_coords, values[vertices])
            results[coord] = float(value)
            self.set_to_cache(lon, lat, target_date, float(value))

            max_dist = float(np.max(np.linalg.norm(points[vertices] - coord, axis=1)))
            self.set_quality_to_cache(lon, lat, target_date, max_dist)

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
