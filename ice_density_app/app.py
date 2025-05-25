import os
from flask import Flask, jsonify, render_template, request
import pandas as pd
from dotenv import load_dotenv

# Импортируем интерполяторы из пакета
from interpolators import (
    TemporalInterpolator,
    KrigingInterpolator,
    DelaunayInterpolator,
    WeightedCombinedInterpolator,
    INTERPOLATOR_CLASSES
)

# --- Flask-приложение ---
app = Flask(__name__)
load_dotenv()
MAPBOX_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN', '')

# --- Загрузка и подготовка данных ---
DF = pd.read_csv('density_below_85_latitude.csv', parse_dates=['date'])
DF['date_only'] = DF['date'].dt.date
UNIQUE_COORDS_DF = pd.read_csv('unique_cords_below_85_latitude.csv')
UNIQUE_COORDS = set(zip(UNIQUE_COORDS_DF.longitude, UNIQUE_COORDS_DF.latitude))

# --- Инициализация интерполяторов (можно подбирать параметры) ---
temporal = TemporalInterpolator(DF, UNIQUE_COORDS)
kriging = KrigingInterpolator(max_neighbors=10, max_radius_km=30)
delaunay = DelaunayInterpolator()

# --- Главная страница ---
@app.route('/')
def index():
    return render_template('index.html', mapbox_token=MAPBOX_TOKEN)

# --- API получения данных с поддержкой комбинированных методов ---
@app.route('/data')
def data():
    date_str = request.args.get('date', '')
    method = request.args.get('method', 'temporal').lower()
    # Пример для комбинированных: ?method=combined&weights={"kriging":0.7,"temporal":0.3,"delaunay":0.0}&auto_weights=false

    # Обработка пользовательских весов и авто-взвешивания
    weights = request.args.get('weights')
    if weights:
        import json
        try:
            weights = json.loads(weights)
        except Exception:
            weights = None
    auto_weights = request.args.get('auto_weights', 'false').lower() == 'true'

    try:
        target_date = pd.to_datetime(date_str).date()
    except Exception:
        return jsonify({"real_points": [], "interp_points": []})

    real_points_df = DF[DF['date_only'] == target_date]
    real_points_list = real_points_df[['longitude', 'latitude', 'density']].to_dict(orient='records')
    real_coords = {(p['longitude'], p['latitude']) for p in real_points_list}
    coords_to_interp = UNIQUE_COORDS - real_coords

    # --- Выбор нужного интерполянта ---
    if method == 'kriging':
        interp = kriging
    elif method == 'delaunay':
        interp = delaunay
    elif method == 'temporal':
        interp = temporal
    elif method == 'combined':
        # Формируем dict нужных методов (из request.args или по умолчанию)
        # Можно дать возможность клиенту выбирать, кого комбинировать
        comb_dict = {}
        for name in INTERPOLATOR_CLASSES:
            if name in ['combined']: continue
            if weights is not None and weights.get(name, 0) > 0:
                # Только если вес не нулевой
                if name == 'kriging':
                    comb_dict[name] = kriging
                elif name == 'delaunay':
                    comb_dict[name] = delaunay
                elif name == 'temporal':
                    comb_dict[name] = temporal
        if not comb_dict:
            comb_dict = {'kriging': kriging, 'temporal': temporal}  # дефолт
            weights = None  # дефолт: равные веса

        interp = WeightedCombinedInterpolator(comb_dict, weights=weights, auto_weights=auto_weights)
    else:
        # 'none' или неизвестный метод — только реальные точки, без интерполяции
        interp = None

    if interp is not None:
        interp_dict = interp.interpolate(target_date, coords_to_interp, real_points_list)
    else:
        interp_dict = {}

    interp_points_list = [
        {'longitude': k[0], 'latitude': k[1], 'density': v}
        for k, v in interp_dict.items()
    ]

    return jsonify({
        'real_points': real_points_list,
        'interp_points': interp_points_list
    })

if __name__ == '__main__':
    app.run(debug=True)
