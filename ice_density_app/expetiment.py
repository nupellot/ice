import numpy as np
import pandas as pd
from approximators import create_approximator

# --- Загрузка данных ---
df = pd.read_csv('density_below_85_latitude.csv', parse_dates=['date'])
df['date_only'] = df['date'].dt.date

# --- Определяем квадрат пропуска ---
lon_min, lon_max = 125, 130
lat_min, lat_max = 79, 81

# --- Разделение данных на обучающую и тестовую выборки ---
# Обучающая выборка - все точки вне квадрата
train_df = df[~((df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) &
                (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max))]

# Тестовая выборка - точки внутри квадрата
test_df = df[(df['longitude'] >= lon_min) & (df['longitude'] <= lon_max) &
             (df['latitude'] >= lat_min) & (df['latitude'] <= lat_max)]

# --- Уникальные координаты из обучающей выборки ---
unique_coords_train = set(zip(train_df['longitude'], train_df['latitude']))

# --- Дата для эксперимента ---
target_date = pd.to_datetime('2019-06-11').date()

# --- Тестовые точки на дату ---
test_points = test_df[test_df['date_only'] == target_date][['longitude', 'latitude', 'density']]
if test_points.empty:
    print("В тестовой области нет данных на выбранную дату!")
    exit(1)

# --- Известные точки обучающей выборки на дату ---
train_points_df = train_df[train_df['date_only'] == target_date]
train_points = train_points_df[['longitude', 'latitude', 'density']].to_dict(orient='records')

# --- Координаты для интерполяции — берём тестовые координаты ---
coords_to_interp = set(zip(test_points['longitude'], test_points['latitude']))

# --- Инициализация аппроксиматоров на обучающих данных ---
approximators = {
    'temporal': create_approximator('temporal', df=train_df, unique_coords=unique_coords_train),
    'kriging': create_approximator('kriging', max_neighbors=15, max_radius_km=30),
    'delaunay': create_approximator('delaunay'),
    # Для fourier и spline важно использовать правильный параметр harmonics (или smoothing_factor)
    'fourier': create_approximator('fourier', df=train_df, unique_coords=unique_coords_train, n_harmonics=10),
    'spline': create_approximator('spline', df=train_df, unique_coords=unique_coords_train, smoothing_factor=1.0)
}

def evaluate_predictions(true_df, pred_dict):
    true_vals = []
    pred_vals = []
    missing_coords = []
    for idx, row in true_df.iterrows():
        coord = (row['longitude'], row['latitude'])
        true_val = row['density']
        pred_val = pred_dict.get(coord)
        if pred_val is not None:
            true_vals.append(true_val)
            pred_vals.append(pred_val)
        else:
            missing_coords.append(coord)
    if len(true_vals) == 0:
        print('Внимание: нет совпадающих предсказаний для оценки!')
        return {'RMSE': float('nan'), 'MAE': float('nan'), 'missing_count': len(missing_coords)}
    true_vals = np.array(true_vals)
    pred_vals = np.array(pred_vals)
    rmse = np.sqrt(np.mean((true_vals - pred_vals)**2))
    mae = np.mean(np.abs(true_vals - pred_vals))
    print(f'Оценено точек: {len(true_vals)} из {len(true_df)}, пропущено: {len(missing_coords)}')
    if len(missing_coords) > 0:
        print(f'Пропущенные координаты (пример): {missing_coords[:5]}')
    return {'RMSE': rmse, 'MAE': mae, 'missing_count': len(missing_coords)}

# --- Запуск эксперимента ---
for name, interp in approximators.items():
    print(f'\nЗапуск аппроксимации методом: {name}')
    pred = interp.approximate(target_date, coords_to_interp, train_points)
    metrics = evaluate_predictions(test_points, pred)
    print(f'Метод: {name}, RMSE: {metrics["RMSE"]:.3f}, MAE: {metrics["MAE"]:.3f}')
