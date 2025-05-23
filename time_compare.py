import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

df = pd.read_csv('density.csv', parse_dates=['date'])
unique_coords = df[['longitude', 'latitude']].drop_duplicates().reset_index(drop=True)

# Для хранения ошибок по моделям для каждой точки
model_names = ['linear', 'exp', 'sin', 'spline']
errors_all = {name: [] for name in model_names}
best_model_counts = {name: 0 for name in model_names}

def linear_func(t, a, b):
    return a * t + b

def exp_func(t, a, b):
    return a * np.exp(b * t)

def sin_func(t, a, b, c, d):
    return a * np.sin(b * t + c) + d

for idx, row in unique_coords.iterrows():
    lon, lat = row['longitude'], row['latitude']
    data_point = df[(df['longitude'] == lon) & (df['latitude'] == lat)].sort_values('date')

    if len(data_point) < 5:
        # Пропускаем точки с недостаточным количеством данных
        continue

    t = (data_point['date'] - data_point['date'].iloc[0]).dt.total_seconds() / (3600*24)
    y = data_point['density'].values

    errors = {}

    # Линейная
    try:
        popt_lin, _ = curve_fit(linear_func, t, y)
        y_pred_lin = linear_func(t, *popt_lin)
        errors['linear'] = np.mean((y - y_pred_lin) ** 2)
    except:
        errors['linear'] = np.inf

    # Экспоненциальная
    try:
        y_shift = 0
        if np.any(y <= 0):
            y_shift = abs(np.min(y)) + 1
        y_exp = y + y_shift
        popt_exp, _ = curve_fit(exp_func, t, y_exp, maxfev=10000)
        y_pred_exp = exp_func(t, *popt_exp) - y_shift
        errors['exp'] = np.mean((y - y_pred_exp) ** 2)
    except:
        errors['exp'] = np.inf

    # Синусоидальная
    try:
        guess_amplitude = (np.max(y) - np.min(y)) / 2
        guess_offset = np.mean(y)
        guess_freq = 2 * np.pi / (t.iloc[-1] - t.iloc[0]) if (t.iloc[-1] - t.iloc[0]) != 0 else 1
        guess_phase = 0
        p0 = [guess_amplitude, guess_freq, guess_phase, guess_offset]
        popt_sin, _ = curve_fit(sin_func, t, y, p0=p0, maxfev=10000)
        y_pred_sin = sin_func(t, *popt_sin)
        errors['sin'] = np.mean((y - y_pred_sin) ** 2)
    except:
        errors['sin'] = np.inf

    # Сглаживающий сплайн
    try:
        s = len(t) * np.var(y) * 0.2
        us = UnivariateSpline(t, y, s=s)
        y_pred_spline = us(t)
        errors['spline'] = np.mean((y - y_pred_spline) ** 2)
    except:
        errors['spline'] = np.inf

    # Сохраняем ошибки
    for model in model_names:
        errors_all[model].append(errors.get(model, np.inf))

    # Считаем лучший вариант
    best_model = min(errors, key=errors.get)
    best_model_counts[best_model] += 1

# Создаём DataFrame со статистикой
stats = {
    'Модель': [],
    'Количество лучших выборов': [],
    'Доля лучших выборов, %': [],
    'Среднее MSE': [],
    'Стандартное отклонение MSE': []
}

total_points = sum(best_model_counts.values())

for model in model_names:
    model_errors = np.array(errors_all[model])
    model_errors = model_errors[np.isfinite(model_errors)]  # Отбрасываем inf
    count_best = best_model_counts[model]
    stats['Модель'].append(model)
    stats['Количество лучших выборов'].append(count_best)
    stats['Доля лучших выборов, %'].append(round(100 * count_best / total_points, 2) if total_points > 0 else 0)
    stats['Среднее MSE'].append(np.mean(model_errors) if len(model_errors) > 0 else np.nan)
    stats['Стандартное отклонение MSE'].append(np.std(model_errors) if len(model_errors) > 0 else np.nan)

import pandas as pd
df_stats = pd.DataFrame(stats)

import ace_tools as tools; tools.display_dataframe_to_user(name="Сравнительная статистика моделей интерполяции", dataframe=df_stats)

print(df_stats)
