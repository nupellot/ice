import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline, splrep

# Загрузка данных
df = pd.read_csv('density.csv', parse_dates=['date'])
unique_coords = df[['longitude', 'latitude']].drop_duplicates().reset_index(drop=True)

batch_size = 6
n_points = len(unique_coords)

# Цвета и стили моделей
model_styles = {
    'linear': {'color': 'blue',   'linestyle': '--', 'label': 'Линейная'},
    'exp':    {'color': 'green',  'linestyle': '-.', 'label': 'Экспоненциальная'},
    'sin':    {'color': 'orange', 'linestyle': ':',  'label': 'Синусоидальная'},
    'spline': {'color': 'red',    'linestyle': '-',  'label': 'Сглаживающий сплайн'},
}

def linear_func(t, a, b):
    return a * t + b

def exp_func(t, a, b):
    return a * np.exp(b * t)

def sin_func(t, a, b, c, d):
    return a * np.sin(b * t + c) + d

def format_formula(model, params, t_data=None):
    if model == 'linear':
        a, b = params
        return r"$\rho(t) = %.3f\, t + %.3f$" % (a, b)
    elif model == 'exp':
        a, b = params
        return r"$\rho(t) = %.3f\, e^{%.3f t}$" % (a, b)
    elif model == 'sin':
        a, b, c, d = params
        return r"$\rho(t) = %.3f\, \sin(%.3f t + %.3f) + %.3f$" % (a, b, c, d)
    elif model == 'spline' and params is not None and t_data is not None:
        knots, coeffs, degree = splrep(t_data, params(t_data), s=0)
        return r"Сглаж. сплайн, порядок %d, %d узлов" % (degree, len(knots))
    else:
        return r"Сглаживающий сплайн"

for start in range(0, n_points, batch_size):
    batch_coords = unique_coords.iloc[start:start+batch_size]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (_, row) in enumerate(batch_coords.iterrows()):
        ax = axes[i]
        lon, lat = row['longitude'], row['latitude']
        sub = df[(df['longitude']==lon)&(df['latitude']==lat)].sort_values('date')

        # Если мало точек
        if len(sub) < 5:
            ax.plot(sub['date'], sub['density'], '-o', color='gray', label=r'$\rho(t)$')
            ax.set_title(f"Lon: {lon}, Lat: {lat}\n(Мало данных)")
            ax.set_xlabel('Дата $t$')
            ax.set_ylabel(r'$\rho$')
            ax.grid(alpha=0.4)
            ax.legend()
        else:
            t = (sub['date'] - sub['date'].iloc[0]).dt.total_seconds()/(3600*24)
            y = sub['density'].values

            errors, params = {}, {}

            # Линейная
            try:
                p_lin, _ = curve_fit(linear_func, t, y)
                errors['linear'] = np.mean((y - linear_func(t,*p_lin))**2)
                params['linear'] = p_lin
            except: errors['linear'] = np.inf

            # Экспоненциальная
            try:
                shift = max(0, -np.min(y)) + 1 if np.any(y<=0) else 0
                p_exp, _ = curve_fit(exp_func, t, y+shift, maxfev=10000)
                fit = exp_func(t,*p_exp) - shift
                errors['exp'] = np.mean((y - fit)**2)
                params['exp'] = p_exp
            except: errors['exp'] = np.inf

            # Синусоидальная
            try:
                A = (y.max()-y.min())/2
                D = y.mean()
                F = 2*np.pi/(t.iloc[-1]-t.iloc[0]) if t.iloc[-1]!=t.iloc[0] else 1
                p0 = [A, F, 0, D]
                p_sin, _ = curve_fit(sin_func, t, y, p0=p0, maxfev=10000)
                errors['sin'] = np.mean((y - sin_func(t,*p_sin))**2)
                params['sin'] = p_sin
            except: errors['sin'] = np.inf

            # Сглаживающий сплайн
            try:
                s = len(t)*np.var(y)*0.2
                spline = UnivariateSpline(t, y, s=s)
                errors['spline'] = np.mean((y - spline(t))**2)
                params['spline'] = spline
            except: errors['spline'] = np.inf

            best = min(errors, key=errors.get)

            # Исходные данные
            ax.plot(sub['date'], y, '-o', color='blue', alpha=0.8, markersize=5, label=r'$\rho(t)$')

            t_dense = np.linspace(t.min(), t.max(), 200)
            dates_dense = sub['date'].iloc[0] + pd.to_timedelta(t_dense, unit='D')

            for model in ['linear','exp','sin','spline']:
                if errors[model]==np.inf: continue
                if model=='linear': fit = linear_func(t_dense,*params[model])
                elif model=='exp':
                    fit = exp_func(t_dense,*params[model])
                    if np.any(y<=0): fit -= shift
                elif model=='sin': fit = sin_func(t_dense,*params[model])
                else: fit = params['spline'](t_dense)
                ax.plot(
                    dates_dense, fit,
                    color=model_styles[model]['color'],
                    linestyle=model_styles[model]['linestyle'],
                    linewidth=2 if model==best else 1,
                    alpha=1 if model==best else 0.5,
                    label=f"{model_styles[model]['label']}{' (выбрана)' if model==best else ''}"
                )

            # Добавляем вертикальные линии на границах лет
            start_year = sub['date'].dt.year.min()
            end_year   = sub['date'].dt.year.max()
            for yr in range(start_year+1, end_year+1):
                x_line = pd.Timestamp(year=yr, month=1, day=1)
                ax.axvline(x=x_line, color='gray', linestyle='--', linewidth=1, alpha=0.3)

            # Формула и оформление
            formula = format_formula(best, params.get(best), t_data=t)
            ax.set_title(
                f"Lon: {lon}, Lat: {lat}\n"
                f"{model_styles[best]['label']} интерполяция\n"
                f"{formula}\n"
                f"$\\bf{{MSE: {errors[best]:.3f}}}$",
                fontsize=12
            )
            ax.set_xlabel('Дата $t$')
            ax.set_ylabel(r'$\rho$')
            ax.grid(alpha=0.4)
            ax.legend(fontsize=8, loc='best')

        # Форматирование дат для каждой оси
        locator   = mdates.AutoDateLocator(minticks=3, maxticks=5)
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', which='major', labelrotation=30, labelbottom=True)

    # Убираем лишние оси
    for j in range(len(batch_coords), batch_size):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()
