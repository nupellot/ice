import pandas as pd
from flask import Flask, render_template
import plotly.express as px
import plotly.io as pio
import json

# Настройки для Flask
app = Flask(__name__)

# Загрузка данных из CSV
def load_data(file_path):
    # Замените путь на ваш CSV файл
    data = pd.read_csv(file_path)
    return data

# Генерация графика
def create_map(data, date):
    # Фильтрация данных по выбранной дате
    date_data = data[data['time'] == date]
    
    # Создание карты с помощью Plotly и Mapbox
    fig = px.scatter_mapbox(date_data,
                            lat='latitude',
                            lon='longitude',
                            size='density',  # Плотность отображается размером точек
                            color='density',  # Плотность отображается цветом
                            hover_name='density',
                            hover_data=['latitude', 'longitude'],
                            color_continuous_scale='Viridis',
                            title=f'Ice Density on {date}')

    # Настройка карты
    fig.update_layout(mapbox_style="open-street-map",  # Используем бесплатный стиль карты
                      mapbox_zoom=5,
                      mapbox_center={"lat": 90, "lon": 0},  # Центрируем карту на северном полюсе
                      margin={"r":0,"t":0,"l":0,"b":0},
                      mapbox=dict(
                          accesstoken="your_mapbox_access_token",  # Если у вас есть токен, замените его
                          style="open-street-map",
                          zoom=3,
                          center={"lat": 90, "lon": 0},
                          bearing=0,
                          pitch=0
                      ))

    # Возвращаем график в формате HTML
    return pio.to_html(fig, full_html=False)

@app.route('/')
def index():
    # Путь к вашему файлу
    file_path = '../density.csv'
    
    # Загрузка данных
    data = load_data(file_path)

    # Выбор дня для визуализации
    date = '2019-01-08'  # Замените на нужную вам дату или параметр

    # Создание карты для выбранной даты
    map_html = create_map(data, date)
    
    # Рендеринг страницы с картой
    return render_template('index.html', map_html=map_html)

if __name__ == '__main__':
    app.run(debug=True)
