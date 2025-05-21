import os
from flask import Flask, jsonify, render_template, request
import pandas as pd

app = Flask(__name__)

# Загрузка .env (если используется python-dotenv)
from dotenv import load_dotenv
load_dotenv()

# Читаем Mapbox токен из переменных окружения
MAPBOX_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN', '')

# Загружаем данные о плотности льда
# Поле date приведено к datetime, для фильтрации по дате создаем дату без времени
DF = pd.read_csv('density_below_85_latitude.csv', parse_dates=['date'])
DF['date_only'] = DF['date'].dt.date

# Загружаем список всех уникальных координат
UNIQUE_COORDS = pd.read_csv('unique_cords_below_85_latitude.csv')
# Приводим к списку кортежей для быстрого поиска
ALL_COORDS = set(zip(UNIQUE_COORDS.longitude, UNIQUE_COORDS.latitude))

# Для ускорения поиска ближайших срезов:
# Группируем по координатам и сортируем по date_only
GROUPED = DF.sort_values('date_only').groupby(['longitude', 'latitude'])

@app.route('/')
def index():
    # Рендерим шаблон и передаем токен Mapbox
    return render_template('index.html', mapbox_token=MAPBOX_TOKEN)

@app.route('/data')
def get_data():
    """
    Возвращает два списка точек:
    - real_points: данные за выбранную дату
    - interp_points: точки, для которых выполняется временная интерполяция
    Интерполяция: среднее между ближайшими срезами до и после запрошенной даты.
    """
    date_str = request.args.get('date', '')
    try:
        target_date = pd.to_datetime(date_str).date()
    except Exception:
        return jsonify({"real_points": [], "interp_points": []})

    # Фильтруем реальные данные за целевую дату
    df_real = DF[DF['date_only'] == target_date]
    real_coords = set(zip(df_real.longitude, df_real.latitude))

    # Для оставшихся координат выполняем интерполяцию
    interp_points = []
    for lon, lat in ALL_COORDS:
        if (lon, lat) in real_coords:
            continue
        key = (lon, lat)
        group = GROUPED.get_group(key)
        # Даты до и после
        before = group[group['date_only'] < target_date]
        after  = group[group['date_only'] > target_date]

        # Берем ближайший по датам
        prev_row = before.iloc[-1] if not before.empty else None
        next_row = after.iloc[0]  if not after.empty  else None

        if prev_row is not None and next_row is not None:
            # Линейная интерполяция по времени
            t0 = prev_row['date_only']
            t1 = next_row['date_only']
            dens0 = prev_row['density']
            dens1 = next_row['density']
            # Доля прохождения между t0 и t1
            total_days = (t1 - t0).days
            frac = (target_date - t0).days / total_days if total_days > 0 else 0
            density_interp = dens0 + (dens1 - dens0) * frac
        elif prev_row is not None:
            density_interp = prev_row['density']
        elif next_row is not None:
            density_interp = next_row['density']
        else:
            # Нет данных ни до, ни после — пропускаем точку
            continue

        interp_points.append({
            'longitude': lon,
            'latitude': lat,
            'density': float(density_interp)
        })

    # Формируем ответ
    real_points = df_real[['longitude', 'latitude', 'density']].to_dict(orient='records')
    return jsonify({
        'real_points': real_points,
        'interp_points': interp_points
    })

if __name__ == '__main__':
    app.run(debug=True)
