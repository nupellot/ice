import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Параметры
CSV_PATH = 'density.csv'  # путь к файлу данных
OUT_CSV = 'interpolation_model_stats.csv'  # куда сохранить результат
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # число потоков (ядер)

MIN_OBS = 4  # минимальное число наблюдений для анализа

def linear_func(t, a, b):
    return a * t + b

def exp_func(t, a, b):
    return a * np.exp(b * t)

def sin_func(t, a, b, c, d):
    return a * np.sin(b * t + c) + d

def process_point(args):
    lon, lat, df = args
    data_point = df[(df['longitude'] == lon) & (df['latitude'] == lat)].sort_values('date')
    n_obs = len(data_point)
    if n_obs < MIN_OBS:
        return None

    t = (data_point['date'] - data_point['date'].iloc[0]).dt.total_seconds() / (3600*24)
    y = data_point['density'].values

    errors = {}
    model_names = ['linear', 'exp', 'sin', 'spline']

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

    best_model = min(errors, key=errors.get)

    result = {
        'longitude': lon,
        'latitude': lat,
        'n_observations': n_obs,
        'best_model': best_model,
    }
    for m in model_names:
        result[f'{m}_mse'] = errors.get(m, np.inf)
    return result

def main():
    print(f"Загрузка данных из {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH, parse_dates=['date'])
    unique_coords = df[['longitude', 'latitude']].drop_duplicates().values.tolist()
    print(f"Обнаружено {len(unique_coords)} уникальных точек.")

    # Подготовка аргументов для параллельного анализа
    args_iter = ((lon, lat, df) for lon, lat in unique_coords)

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(process_point, args) for args in args_iter]
        for i, future in enumerate(as_completed(futures), 1):
            res = future.result()
            if res is not None:
                results.append(res)
            if i % 1000 == 0:
                print(f"Обработано {i} точек...")

    result_df = pd.DataFrame(results)
    print(f"Сохраняю результаты в {OUT_CSV} ...")
    result_df.to_csv(OUT_CSV, index=False)
    print("Готово!")
    print(result_df.head())

    # Статистика по моделям
    stats = result_df['best_model'].value_counts(normalize=False).to_frame('count')
    stats['percent'] = 100 * stats['count'] / stats['count'].sum()
    print("\nРаспределение лучших моделей:")
    print(stats)

    print("\nСредний MSE по моделям:")
    for m in ['linear', 'exp', 'sin', 'spline']:
        mse = result_df[f'{m}_mse']
        print(f"{m:10}: mean={mse[mse < np.inf].mean():.5f}, std={mse[mse < np.inf].std():.5f}")

if __name__ == "__main__":
    main()
