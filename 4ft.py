import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.linalg import lstsq
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

MAX_POLY_DEGREE = 3
MAX_HARMONICS = 15
EXTRAP_DAYS = 365
BATCH_SIZE = 6
CV_SPLITS = 5

df = pd.read_csv('density.csv', parse_dates=['date'])
unique_coords = df[['longitude', 'latitude']].drop_duplicates().reset_index(drop=True)

def build_design_matrix(t, tau, d, N):
    poly_cols = [np.ones_like(t)] + [t**p for p in range(1, d+1)]
    fourier_cols = [func(2 * np.pi * k * tau)
                    for k in range(1, N+1) for func in (np.cos, np.sin)]
    return np.vstack(poly_cols + fourier_cols).T

def fit_trend_plus_fourier(t, y, degree, n_harmonics):
    t0, T = t.min(), t.max() - t.min()
    tau = (t - t0) / T
    X = build_design_matrix(t, tau, degree, n_harmonics)
    coeffs, *_ = lstsq(X, y, rcond=None)

    def predict_fn(t_query):
        tau_q = (t_query - t0) / T
        Xq = build_design_matrix(t_query, tau_q, degree, n_harmonics)
        return Xq @ coeffs

    return coeffs, t0, T, predict_fn


from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import numpy as np

def cross_val_score_trend_fourier(t, y, d, N, n_splits=3, min_val_size=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    errors = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(t), 1):
        print(f"Fold {fold_idx}: train size = {len(train_idx)}, val size = {len(val_idx)}")
        if len(val_idx) < min_val_size:
            print(f"  Skipping fold {fold_idx} due to small val size")
            continue
        t_train, y_train = t[train_idx], y[train_idx]
        t_val, y_val = t[val_idx], y[val_idx]
        try:
            _, _, _, predict_fn = fit_trend_plus_fourier(t_train, y_train, d, N)
            y_pred = predict_fn(t_val)
            fold_mse = mean_squared_error(y_val, y_pred)
            print(f"  Fold {fold_idx} MSE = {fold_mse:.6f}")
            errors.append(fold_mse)
        except Exception as e:
            print(f"  Exception in fold {fold_idx}: {e}")
            return np.inf
    if not errors:
        print("No valid folds for CV")
        return np.inf
    mean_error = np.mean(errors)
    print(f"Mean CV MSE for d={d}, N={N}: {mean_error:.6f}")
    return mean_error

# Пример вызова внутри основного цикла:
# для параметров d и N:
# score = cross_val_score_trend_fourier(t_days.values, y_vals, d, N)




for start in range(0, len(unique_coords), BATCH_SIZE):
    batch = unique_coords.iloc[start:start+BATCH_SIZE]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, batch.iterrows()):
        lon, lat = row['longitude'], row['latitude']
        sub = df[(df.longitude == lon) & (df.latitude == lat)].sort_values('date')

        if len(sub) < 5:
            ax.text(0.5, 0.5, 'Мало данных', ha='center', va='center')
            ax.set_title(f"Lon: {lon}, Lat: {lat}")
            ax.axis('off')
            continue

        dates = sub['date']
        t_days = (dates - dates.iloc[0]).dt.total_seconds() / 86400.0
        y_vals = sub['density'].values

        print(f"\nОбработка точки Lon={lon}, Lat={lat}, число измерений: {len(t_days)}")
        best_score = np.inf
        best_params = (0, 1)

        for d in range(MAX_POLY_DEGREE + 1):
            for N in range(1, MAX_HARMONICS + 1):
                score = cross_val_score_trend_fourier(t_days.values, y_vals, d, N)
                print(f"  deg={d}, N={N}, CV MSE={score:.5f}")
                if score < best_score:
                    best_score = score
                    best_params = (d, N)

        d_opt, N_opt = best_params
        print(f"Выбраны параметры: deg={d_opt}, N={N_opt} с CV MSE={best_score:.5f}")

        coeffs, t0, T, predict_fn = fit_trend_plus_fourier(t_days, y_vals, d_opt, N_opt)
        t_dense = np.linspace(t_days.min(), t_days.max() + EXTRAP_DAYS, 400)
        dates_dense = dates.iloc[0] + pd.to_timedelta(t_dense, unit='D')
        y_pred_dense = predict_fn(t_dense)
        mse_train = np.mean((y_vals - predict_fn(t_days))**2)

        ax.plot(dates, y_vals, '-o', color='blue', markersize=4, label=r'Исходные $\rho(t)$')
        ax.plot(dates_dense, y_pred_dense, color='purple',
                label=f'Модель (deg={d_opt}, N={N_opt})')
        ax.set_title(f"Lon: {lon}, Lat: {lat} | deg={d_opt}, N={N_opt}\nMSE = {mse_train:.4f}",
                     fontsize=11)
        ax.set_xlabel('Дата $t$')
        ax.set_ylabel(r'$\rho$')
        ax.grid(alpha=0.3)

        locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', labelrotation=30)
        ax.legend(fontsize=8)

    for j in range(len(batch), BATCH_SIZE):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()
