import os
from flask import Flask, jsonify, render_template, request
import pandas as pd
from collections import defaultdict
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from numpy.linalg import solve
import time

app = Flask(__name__)

from dotenv import load_dotenv
load_dotenv()

MAPBOX_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN', '')

class InterpolatorBase:
    def __init__(self):
        self.interp_cache = defaultdict(dict)

    def get_from_cache(self, lon, lat, target_date):
        return self.interp_cache.get((lon, lat), {}).get(target_date)

    def set_to_cache(self, lon, lat, target_date, value):
        self.interp_cache[(lon, lat)][target_date] = value

    def interpolate(self, target_date, coords_to_interpolate, known_points):
        raise NotImplementedError

class TemporalInterpolator(InterpolatorBase):
    def __init__(self, df, unique_coords):
        super().__init__()
        self.df = df
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

    def interpolate(self, target_date, coords_to_interpolate, known_points):
        interp_results = {}
        known_coords = {(p['longitude'], p['latitude']) for p in known_points}
        for coord in coords_to_interpolate:
            if coord in known_coords:
                continue
            dens = self.interpolate_point(coord[0], coord[1], target_date)
            if dens is not None:
                interp_results[coord] = dens
        return interp_results

class SpatialInterpolator(InterpolatorBase):
    def __init__(self, sill=34.37, range_param=59.15):
        super().__init__()
        self.sill = sill
        self.range = range_param

    def variogram(self, h):
        return self.sill * (1 - np.exp(-h / self.range))

    @staticmethod
    def haversine_np(lon1, lat1, lon2, lat2):
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return 6371 * c

    def interpolate(self, target_date, coords_to_interpolate, known_points):
        if not known_points:
            return {}
        known_coords = np.array([[p['longitude'], p['latitude']] for p in known_points])
        known_values = np.array([p['density'] for p in known_points])
        known_coords_set = {(p['longitude'], p['latitude']) for p in known_points}
        known_lons = np.radians(known_coords[:, 0])
        known_lats = np.radians(known_coords[:, 1])
        interp_results = {}
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
            weights = 1 / (self.variogram(dists) + 1e-6)
            weights /= weights.sum()
            density_interp = float(np.dot(weights, known_values))
            self.set_to_cache(lon, lat, target_date, density_interp)
            interp_results[coord] = density_interp
        return interp_results

class KrigingInterpolator(InterpolatorBase):
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

    def interpolate(self, target_date, coords_to_interpolate, known_points):
        if not known_points:
            return {}
        known_coords = np.array([[p['longitude'], p['latitude']] for p in known_points])
        known_values = np.array([p['density'] for p in known_points])
        known_lons = np.radians(known_coords[:, 0])
        known_lats = np.radians(known_coords[:, 1])
        interp_results = {}
        known_coords_set = {(p['longitude'], p['latitude']) for p in known_points}
        coords_to_interpolate = list(coords_to_interpolate)
        t0 = time.time()
        for idx, coord in enumerate(coords_to_interpolate):
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
            except np.linalg.LinAlgError:
                continue
            lambdas = weights[:-1]
            interp_value = float(np.dot(lambdas, values_used))
            self.set_to_cache(lon, lat, target_date, interp_value)
            interp_results[coord] = interp_value
            if idx % 100 == 0 and idx > 0:
                print(f'{idx} / {len(coords_to_interpolate)} points (elapsed: {time.time()-t0:.2f} sec)')
        print(f"Kriging finished {len(coords_to_interpolate)} points in {time.time()-t0:.2f} sec")
        return interp_results

class DelaunayInterpolator(InterpolatorBase):
    """
    Интерполяция по триангуляции Делоне с линейной интерполяцией в треугольниках.
    """
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
            interp_results[coord] = float(value)

        return interp_results

# --- Загрузка данных ---
DF = pd.read_csv('density_below_85_latitude.csv', parse_dates=['date'])
DF['date_only'] = DF['date'].dt.date
UNIQUE_COORDS_DF = pd.read_csv('unique_cords_below_85_latitude.csv')
UNIQUE_COORDS = set(zip(UNIQUE_COORDS_DF.longitude, UNIQUE_COORDS_DF.latitude))

# --- Инициализация интерполяторов ---
temporal_interpolator = TemporalInterpolator(DF, UNIQUE_COORDS)
spatial_interpolator = SpatialInterpolator()
kriging_interpolator = KrigingInterpolator(max_neighbors=10, max_radius_km=30)
delaunay_interpolator = DelaunayInterpolator()

@app.route('/')
def index():
    return render_template('index.html', mapbox_token=MAPBOX_TOKEN)

@app.route('/data')
def data():
    date_str = request.args.get('date', '')
    method = request.args.get('method', 'temporal').lower()
    try:
        target_date = pd.to_datetime(date_str).date()
    except Exception:
        return jsonify({"real_points": [], "interp_points": []})

    real_points_df = DF[DF['date_only'] == target_date]
    real_points_list = real_points_df[['longitude', 'latitude', 'density']].to_dict(orient='records')
    real_coords = {(p['longitude'], p['latitude']) for p in real_points_list}
    coords_to_interp = UNIQUE_COORDS - real_coords

    if method == 'kriging':
        interp_dict = kriging_interpolator.interpolate(target_date, coords_to_interp, real_points_list)
    elif method == 'spatial':
        interp_dict = spatial_interpolator.interpolate(target_date, coords_to_interp, real_points_list)
    elif method == 'delaunay':
        interp_dict = delaunay_interpolator.interpolate(target_date, coords_to_interp, real_points_list)
    else:
        interp_dict = temporal_interpolator.interpolate(target_date, coords_to_interp, real_points_list)

    interp_points_list = [{'longitude': k[0], 'latitude': k[1], 'density': v} for k, v in interp_dict.items()]

    return jsonify({
        'real_points': real_points_list,
        'interp_points': interp_points_list
    })

if __name__ == '__main__':
    app.run(debug=True)
