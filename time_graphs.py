import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline

df = pd.read_csv('density.csv', parse_dates=['date'])
unique_coords = df[['longitude', 'latitude']].drop_duplicates().reset_index(drop=True)

batch_size = 6
n_points = len(unique_coords)

# Цвета и стили для моделей
model_styles = {
    'linear': {'color': 'blue', 'linestyle': '--', 'label': 'Линейная'},
    'exp': {'color': 'green', 'linestyle': '-.', 'label': 'Экспоненциальная'},
    'sin': {'color': 'orange', 'linestyle': ':', 'label': 'Синусоидальная'},
    'spline': {'color': 'red', 'linestyle': '-', 'label': 'Сглаживающий сплайн'},
}

def linear_func(t, a, b):
    return a * t + b

def exp_func(t, a, b):
    return a * np.exp(b * t)

def sin_func(t, a, b, c, d):
    return a * np.sin(b * t + c) + d

def format_formula(model, params):
    if model == 'linear':
        a, b = params
        return f"$\\rho = {a:.3f} t + {b:.3f}$"
    elif model == 'exp':
        a, b = params
        return f"$\\rho = {a:.3f} e^{{{b:.3f} t}}$"
    elif model == 'sin':
        a, b, c, d = params
        return f"$\\rho = {a:.3f} \\sin({b:.3f} t + {c:.3f}) + {d:.3f}$"
    elif model == 'spline':
        return "Сглаживающий сплайн"
    else:
        return ""

for start in range(0, n_points, batch_size):
    end = min(start + batch_size, n_points)
    batch_coords = unique_coords.iloc[start:end]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (index, row) in enumerate(batch_coords.iterrows()):
        lon, lat = row['longitude'], row['latitude']
        ax = axes[i]

        data_point = df[(df['longitude'] == lon) & (df['latitude'] == lat)].sort_values('date')
        if len(data_point) < 5:
            ax.plot(data_point['date'], data_point['density'], '-o', color='gray', label='Данные')
            ax.set_title(f'Lon: {lon}, Lat: {lat}\n(Мало данных для регрессии)')
            ax.set_xlabel('Дата')
            ax.set_ylabel('Плотность льда (ρ)')
            ax.grid(True, alpha=0.4)
            ax.legend()
            continue

        t = (data_point['date'] - data_point['date'].iloc[0]).dt.total_seconds() / (3600*24)
        y = data_point['density'].values

        errors = {}
        params = {}

        # Линейная
        try:
            popt_lin, _ = curve_fit(linear_func, t, y)
            y_pred_lin = linear_func(t, *popt_lin)
            errors['linear'] = np.mean((y - y_pred_lin)**2)
            params['linear'] = popt_lin
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
            errors['exp'] = np.mean((y - y_pred_exp)**2)
            params['exp'] = popt_exp
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
            errors['sin'] = np.mean((y - y_pred_sin)**2)
            params['sin'] = popt_sin
        except:
            errors['sin'] = np.inf

        # Сглаживающий сплайн
        try:
            s = len(t) * np.var(y) * 0.2  # 20% от дисперсии — это эмпирический коэффициент
            us = UnivariateSpline(t, y, s=s)
            y_pred_spline = us(t)
            errors['spline'] = np.mean((y - y_pred_spline)**2)
            params['spline'] = us
        except:
            errors['spline'] = np.inf

        best_model = min(errors, key=errors.get)

        # Сначала строим все модели для сравнения (блекло)
        t_dense = np.linspace(t.min(), t.max(), 200)
        dates_dense = pd.to_datetime(data_point['date'].iloc[0]) + pd.to_timedelta(t_dense, unit="D")

        # Точки данных крупные и полупрозрачные
        ax.plot(data_point['date'], y, '-o', color='blue', alpha=1, markersize=5, label='Исходные Данные')

        for model in ['linear', 'exp', 'sin', 'spline']:
            if errors[model] == np.inf:
                continue
            if model == 'linear':
                y_fit = linear_func(t_dense, *params[model])
            elif model == 'exp':
                y_fit = exp_func(t_dense, *params[model])
                if np.any(y <= 0):
                    y_fit -= abs(np.min(y)) + 1
            elif model == 'sin':
                y_fit = sin_func(t_dense, *params[model])
            elif model == 'spline':
                y_fit = params[model](t_dense)
            ax.plot(dates_dense, y_fit, color=model_styles[model]['color'],
                    linestyle=model_styles[model]['linestyle'],
                    linewidth=1.5 if model != best_model else 2,
                    alpha=1 if model != best_model else 1,
                    label=f"{model_styles[model]['label']}{' (выбрана)' if model == best_model else ''}")

        formula_text = format_formula(best_model, params[best_model] if best_model != 'spline' else None)
        ax.set_title(
            f"Lon: {lon}, Lat: {lat}\n"
            f"{model_styles[best_model]['label']} интерполяция\n"
            f"{formula_text}\n"
            f"$\\bf{{MSE: {errors[best_model]:.3f}}}$",
            fontsize=13
        )
        ax.set_xlabel('Дата', fontsize=11)
        ax.set_ylabel('Плотность льда $\\rho$', fontsize=11)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=10, loc='best')

    for j in range(i + 1, batch_size):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
