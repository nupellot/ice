import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.linalg import lstsq

# --- Параметры аппроксимации ---
N_HARMONICS = 20        # число гармоник Фурье
BATCH_SIZE  = 6        # сколько точек рисовать за раз

# --- Загрузка данных ---
# Ожидается файл 'density.csv' с колонками: date, longitude, latitude, density
df = pd.read_csv('density.csv', parse_dates=['date'])

# Получаем список уникальных координат (lon, lat)
unique_coords = (
    df[['longitude', 'latitude']]
    .drop_duplicates()
    .reset_index(drop=True)
)

def fit_fourier(t, y, N=N_HARMONICS):
    """
    Аппроксимирует y(t) рядом Фурье с N гармониками методом МНК.

    Параметры:
      t: одномерный массив времен (в днях от начальной даты)
      y: одномерный массив значений (плотностей)
      N: число гармоник

    Возвращает:
      fourier_func(t_new) -> y_approx
      coeffs           -> вектор коэффициентов [a0, a1, b1, a2, b2, ...]
    """
    # 1. Определим период T = max(t)-min(t) и начальную точку t0
    t0 = t.min()
    T  = t.max() - t0
    # Переведём все точки в нормированное время τ ∈ [0,1]
    tau = (t - t0) / T

    # 2. Построим матрицу признаков X: столбцы [1, cos(2πk\tau), sin(2πk\tau)]
    #    Всего 2*N + 1 столбец: первая константная колонка, затем по две на кажду k
    cols = [np.ones_like(tau)]
    for k in range(1, N+1):
        cols.append(np.cos(2*np.pi*k*tau))
        cols.append(np.sin(2*np.pi*k*tau))
    X = np.vstack(cols).T      # размер M×(2N+1)

    # 3. Решаем X c = y по МНК (псевдообращение)
    coeffs, *_ = lstsq(X, y, rcond=None)

    # 4. Собираем функцию, которая для новых t_query выдаёт аппроксимацию
    def fourier_func(t_query):
        # нормируем t_query так же, по тем же t0 и T
        tau_q = (t_query - t0) / T
        cols_q = [np.ones_like(tau_q)]
        for k in range(1, N+1):
            cols_q.append(np.cos(2*np.pi*k*tau_q))
            cols_q.append(np.sin(2*np.pi*k*tau_q))
        Xq = np.vstack(cols_q).T
        return Xq.dot(coeffs)

    return fourier_func, coeffs

# --- Основной цикл: перебор координат батчами по BATCH_SIZE ---
for start in range(0, len(unique_coords), BATCH_SIZE):
    batch = unique_coords.iloc[start : start + BATCH_SIZE]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for ax, (_, row) in zip(axes, batch.iterrows()):
        lon, lat = row['longitude'], row['latitude']
        # Выделяем данные по координате и сортируем по дате
        sub = df[
            (df.longitude == lon) & (df.latitude == lat)
        ].sort_values('date')

        # Если слишком мало точек — рисуем сообщение
        if len(sub) < 5:
            ax.text(0.5, 0.5, 'Мало данных', ha='center', va='center')
            ax.set_title(f"Lon: {lon}, Lat: {lat}")
            ax.axis('off')
            continue

        # Переводим даты в число дней от первой (t0)
        dates = sub['date']
        t0    = dates.iloc[0]
        t     = (dates - t0).dt.total_seconds() / (3600*24)
        y     = sub['density'].values

        # Строим аппроксимацию рядом Фурье
        fourier_fn, coeffs = fit_fourier(t, y, N=N_HARMONICS)

        # Готовим плотную сетку для отрисовки кривых
        t_dense     = np.linspace(t.min(), t.max(), 300)
        dates_dense = t0 + pd.to_timedelta(t_dense, unit='D')

        # Считаем значения аппроксимаций
        y_fourier = fourier_fn(t_dense)

        # MSE на обучающих точках
        y_fit_train = fourier_fn(t)
        mse_fourier = np.mean((y - y_fit_train)**2)

        # Рисуем исходные точки синим и соединяем линией
        ax.plot(dates, y, '-o', color='blue', markersize=4, label=r'Исходные $\rho(t)$')

        # Рисуем ряд Фурье
        ax.plot(dates_dense, y_fourier,
                color='purple', linestyle='-',
                label=f'Ряд Фурье (N={N_HARMONICS})')

        # Подписи и форматирование
        ax.set_title(
            f"Lon: {lon}, Lat: {lat}\nMSE Фурье: {mse_fourier:.4f}",
            fontsize=12
        )
        ax.set_xlabel('Дата $t$', fontsize=11)
        ax.set_ylabel(r'$\rho$', fontsize=11)
        ax.grid(alpha=0.3)

        # Ось X: читаемые метки дат
        locator   = mdates.AutoDateLocator(minticks=3, maxticks=5)
        formatter = mdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(30)
            lbl.set_horizontalalignment('right')

        ax.legend(fontsize=9)

    # Убираем пустые подплоты
    for j in range(len(batch), BATCH_SIZE):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()
