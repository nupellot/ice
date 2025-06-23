import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numpy.linalg import lstsq

# === ПАРАМЕТРЫ МОДЕЛИ ===
DEGREE      = 1    # степень полинома тренда (1 = линейный тренд)
N_HARMONICS = 3    # число гармоник в Fourier-ряде
PERIOD_DAYS = 365  # период сезонности (дней) — годовые циклы

# === ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ===
df = pd.read_csv('density.csv', parse_dates=['date'])
df = df.sort_values('date').reset_index(drop=True)

# переводим даты в "дни от начала"
t0 = df['date'].iloc[0]
t  = (df['date'] - t0).dt.total_seconds() / (3600*24)  # в днях
y  = df['density'].values

# === СОЗДАНИЕ ДИЗАЙН- МАТРИЦЫ X ===

# 1) Полиномиальные признаки: [1, t, t^2, ..., t^DEGREE]
poly_cols = [t**i for i in range(DEGREE+1)]

# 2) Fourier-признаки:
#    сначала константа a0 (для ряда Фурье),
#    потом по две колонки на каждую гармонику k: cos(2πk t/P), sin(2πk t/P)
tau = t / PERIOD_DAYS
season_cols = [np.ones_like(tau)]
for k in range(1, N_HARMONICS+1):
    season_cols.append(np.cos(2 * np.pi * k * tau))
    season_cols.append(np.sin(2 * np.pi * k * tau))

# объединяем все признаки
X = np.vstack(poly_cols + season_cols).T  # размер: (M, DEGREE+1 + 2*N_HARMONICS)

# === ОЦЕНКА ПАРАМЕТРОВ ЧЕРЕЗ МНК ===
# решаем min ||X θ − y||^2
theta, *_ = lstsq(X, y, rcond=None)

# === ФУНКЦИЯ МОДЕЛИ ДЛЯ НОВЫХ ТОЧЕК ===
def model(t_query):
    # полиномиальные части
    poly_q = [t_query**i for i in range(DEGREE+1)]
    # Fourier-части
    tau_q = t_query / PERIOD_DAYS
    season_q = [np.ones_like(tau_q)]
    for k in range(1, N_HARMONICS+1):
        season_q.append(np.cos(2 * np.pi * k * tau_q))
        season_q.append(np.sin(2 * np.pi * k * tau_q))
    Xq = np.vstack(poly_q + season_q).T
    return Xq.dot(theta)

# === ЭКСТРАПОЛЯЦИЯ И ВИЗУАЛИЗАЦИЯ ===
# границы: от начала до конца + 90 дней вперёд
t_dense     = np.linspace(t.min(), t.max() + 90, 500)
dates_dense = t0 + pd.to_timedelta(t_dense, unit='D')
y_fit       = model(t_dense)

# считаем MSE на обучающих данных
mse = np.mean((y - model(t))**2)

plt.figure(figsize=(12, 6))
# 1) Исходные данные: синие точки, соединённые линией
plt.plot(df['date'], y, '-o', color='blue', label=r'Исходные $\rho(t)$')

# 2) Модель (тренд + Fourier)
plt.plot(dates_dense, y_fit, '--', color='red',
         label=f'Тренд + Fourier (N={N_HARMONICS})')

# 3) граница имеющихся данных
plt.axvline(df['date'].iloc[-1], color='gray', linestyle=':', label='Конец данных')

# оформление
plt.title(f'Полином степени {DEGREE} + Fourier (N={N_HARMONICS}) — MSE = {mse:.4f}', fontsize=14)
plt.xlabel('Дата', fontsize=12)
plt.ylabel(r'Плотность $\rho$', fontsize=12)
plt.grid(alpha=0.3)
plt.legend()

# читаемые метки дат
locator   = mdates.AutoDateLocator(minticks=4, maxticks=6)
formatter = mdates.DateFormatter('%Y-%m-%d')
ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
plt.xticks(rotation=30, ha='right')

plt.tight_layout()
plt.show()
