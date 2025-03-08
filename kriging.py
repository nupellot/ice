import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import griddata

# Загружаем данные
df = pd.read_csv("ice_density_data.csv")

# Очищаем от NaN
df = df.dropna(subset=["lon", "lat", "density"])
df = df[df["density"] >= 0]  # Фильтруем ошибки

# Ограничиваем число точек (если данных > 5000, берем случайные 5000)
if len(df) > 5000:
    df = df.sample(5000, random_state=42)

# Определяем границы для карты
lon_min, lon_max = df["lon"].min(), df["lon"].max()
lat_min, lat_max = df["lat"].min(), df["lat"].max()

# Создаем сетку широта-долгота
grid_size = 30  # Можно уменьшить до 20
grid_lon = np.linspace(lon_min, lon_max, grid_size)
grid_lat = np.linspace(lat_min, lat_max, grid_size)
grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)

# Интерполяция данных (линейная)
grid_density = griddata(
    (df["lon"], df["lat"]), df["density"], (grid_lon, grid_lat), method="linear"
)

# Настройка карты
plt.figure(figsize=(12, 8))
m = Basemap(projection="ortho", lon_0=(lon_min + lon_max) / 2, lat_0=(lat_min + lat_max) / 2)

# Рисуем карту
m.drawcoastlines()
m.drawparallels(np.arange(-90., 91., 10.), labels=[1, 0, 0, 0])
m.drawmeridians(np.arange(-180., 181., 20.), labels=[0, 0, 0, 1])

# Преобразуем координаты в сферические
x, y = m(grid_lon, grid_lat)
x_data, y_data = m(df["lon"].values, df["lat"].values)

# Отображаем плотность льда
contour = plt.contourf(x, y, grid_density, cmap="coolwarm", levels=20)
plt.colorbar(contour, label="Плотность льда")

# Отображаем исходные точки
plt.scatter(x_data, y_data, c="black", s=10, label="Исходные точки")

plt.legend()
plt.title("Карта распределения плотности льда (сферическая)")
plt.show()
