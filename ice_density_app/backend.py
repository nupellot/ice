from flask import Flask, jsonify, render_template, request
import os
import pandas as pd

app = Flask(__name__)

# Загружаем данные
df = pd.read_csv('density_below_85_latitude.csv', parse_dates=['date'])
df['date_only'] = df['date'].dt.date

# Загружаем уникальные координаты
unique_coords = pd.read_csv('unique_cords_below_85_latitude.csv')
unique_coords_set = set(zip(unique_coords['longitude'], unique_coords['latitude']))

# Предрасчёт среднего значения плотности по каждой точке
mean_density_by_point = df.groupby(['longitude', 'latitude'])['density'].mean().reset_index()

@app.route('/')
def index():
    token = os.getenv('MAPBOX_ACCESS_TOKEN', '')
    return render_template('index.html', mapbox_token=token)

@app.route('/data')
def get_data():
    date_str = request.args.get('date', '2019-01-08')
    try:
        date_filter = pd.to_datetime(date_str).date()
    except Exception:
        return jsonify({"real_points": [], "interp_points": []})

    # Данные за выбранную дату
    df_real = df[df['date_only'] == date_filter]
    real_points_set = set(zip(df_real['longitude'], df_real['latitude']))

    # Координаты, которых нет в данных за выбранную дату
    missing_coords = unique_coords_set - real_points_set

    # Интерполированные точки — среднее по этим координатам
    df_interp = mean_density_by_point[
        mean_density_by_point.apply(lambda row: (row['longitude'], row['latitude']) in missing_coords, axis=1)
    ]

    real_points = df_real[['longitude', 'latitude', 'density']].to_dict(orient='records')
    interp_points = df_interp[['longitude', 'latitude', 'density']].to_dict(orient='records')

    return jsonify({"real_points": real_points, "interp_points": interp_points})

if __name__ == '__main__':
    app.run(debug=True)
