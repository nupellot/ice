import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Загружаем данные
df = pd.read_csv("ice_density_data.csv")

# Очищаем от NaN
df = df.dropna(subset=["lon", "lat", "density"])

# Исключаем явно ошибочные данные (например, отрицательные плотности)
df = df[df["density"] >= 0]

# Проверяем, не пустой ли DataFrame после очистки
if df.empty:
    raise ValueError("После фильтрации не осталось данных! Проверь исходный CSV-файл.")

# ОГРАНИЧИВАЕМ количество точек (если их >10,000)
if len(df) > 5000:
    df = df.sample(5000, random_state=42)  # Берем случайные 5000 точек для интерполяции

# Определяем границы карты
lon_min, lon_max = df["lon"].min(), df["lon"].max()
lat_min, lat_max = df["lat"].min(), df["lat"].max()

# Уменьшаем размер сетки
grid_size = 30  # Было 50 → уменьшаем для экономии памяти
grid_lon = np.linspace(lon_min, lon_max, grid_size)
grid_lat = np.linspace(lat_min, lat_max, grid_size)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# Используем линейную интерполяцию вместо кригинга
grid_density = griddata(
    (df["lon"], df["lat"]), df["density"], (grid_lon, grid_lat), method="linear"
)

# Визуализация
plt.figure(figsize=(10, 6))
contour = plt.contourf(grid_lon, grid_lat, grid_density, cmap="coolwarm", levels=20)
plt.colorbar(contour, label="Плотность льда")
plt.scatter(df["lon"], df["lat"], c="black", s=10, label="Исходные точки")
plt.legend()
plt.title("Карта распределения плотности льда")
plt.xlabel("Долгота")
plt.ylabel("Широта")
plt.show()
