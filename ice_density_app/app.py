import os
import json
import pandas as pd
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from approximators import (
    create_approximator,
    list_methods,
    COMBINED_NAME,
)

# --- Инициализация Flask ---
app = Flask(__name__)
load_dotenv()
MAPBOX_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN', '')

# --- Загрузка данных ---
DF = pd.read_csv('density_below_85_latitude.csv', parse_dates=['date'])
DF['date_only'] = DF['date'].dt.date

UNIQUE_COORDS_DF = pd.read_csv('unique_cords_below_85_latitude.csv')
UNIQUE_COORDS = set(zip(UNIQUE_COORDS_DF.longitude, UNIQUE_COORDS_DF.latitude))

# --- Базовые аппроксиматоры (одиночные и для комбинирования) ---
SHARED_APPROXIMATORS = {
    'temporal': create_approximator('temporal', df=DF, unique_coords=UNIQUE_COORDS),
    'kriging': create_approximator('kriging', max_neighbors=15, max_radius_km=30),
    'delaunay': create_approximator('delaunay'),
    # Исправлено название параметра harmonics
    'fourier': create_approximator('fourier', df=DF, unique_coords=UNIQUE_COORDS, harmonics=10)
}

# --- Главная страница ---
@app.route('/')
def index():
    return render_template('index.html', mapbox_token=MAPBOX_TOKEN)

# --- API получения данных для карты ---
@app.route('/data')
def data():
    date_str = request.args.get('date', '')
    method_name = request.args.get('method', 'temporal').lower()
    weights_json = request.args.get('weights')
    auto_weights = request.args.get('auto_weights', 'false').lower() == 'true'

    # Парсинг даты
    try:
        target_date = pd.to_datetime(date_str).date()
    except Exception:
        return jsonify({"real_points": [], "interp_points": []})

    # Реальные точки на эту дату
    real_points_df = DF[DF['date_only'] == target_date]
    real_points_list = real_points_df[['longitude', 'latitude', 'density']].to_dict(orient='records')
    real_coords = {(p['longitude'], p['latitude']) for p in real_points_list}
    coords_to_interp = UNIQUE_COORDS - real_coords

    # Парсинг весов
    weights = None
    if weights_json:
        try:
            weights = json.loads(weights_json)
        except json.JSONDecodeError:
            weights = None

    # --- Выбор метода аппроксимации ---
    if method_name == COMBINED_NAME:
        used_methods = [
            name for name in SHARED_APPROXIMATORS
            if (weights is None or weights.get(name, 0) > 0)
        ]
        if not used_methods:
            used_methods = ['kriging', 'temporal']
            weights = None

        comb_dict = {name: SHARED_APPROXIMATORS[name] for name in used_methods}
        approximator = create_approximator(
            COMBINED_NAME,
            interpolators_dict=comb_dict,
            weights=weights,
            auto_weights=auto_weights
        )
    elif method_name in SHARED_APPROXIMATORS:
        approximator = SHARED_APPROXIMATORS[method_name]
    else:
        approximator = None

    # --- Интерполяция ---
    if approximator is not None:
        interp_dict = approximator.approximate(target_date, coords_to_interp, real_points_list)
    else:
        interp_dict = {}

    interp_points_list = [
        {'longitude': lon, 'latitude': lat, 'density': dens}
        for (lon, lat), dens in interp_dict.items()
    ]

    return jsonify({
        'real_points': real_points_list,
        'interp_points': interp_points_list
    })

# --- Эндпоинт: список доступных методов ---
@app.route('/methods')
def list_available_methods():
    kinds = ['temporal', 'spatial']
    return jsonify({
        kind: list_methods(kind)
        for kind in kinds
    })

if __name__ == '__main__':
    app.run(debug=True)
