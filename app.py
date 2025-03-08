from flask import Flask, jsonify, render_template
import sqlite3
import numpy as np
import json

app = Flask(__name__)

# Функция для получения данных из БД
def get_ice_data():
    conn = sqlite3.connect("ice_v2019_2022.sqlite")  # Укажи путь к своей БД
    cursor = conn.cursor()
    
    query = """
    SELECT round(lon), round(lat), v_ref
    FROM vRefData
    JOIN iceGrid ON iceGrid.id = vRefData.cell_id 
    JOIN timeGrid ON timeGrid.id = vRefData.time_id 
    WHERE correctTime = '2019-01-08'
    limit 5000;
    """
    
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()
    
    return data

# Функция для генерации полигонов
def generate_polygons():
    ice_data = get_ice_data()
    
    grid_size = 0.25  # Примерный размер ячейки (25x25 км в градусах)
    
    features = []
    for lon, lat, v_ref in ice_data:
        # Создаём квадрат (полигон)
        polygon = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [lon, lat],
                    [lon + grid_size, lat],
                    [lon + grid_size, lat + grid_size],
                    [lon, lat + grid_size],
                    [lon, lat]
                ]]
            },
            "properties": {
                "v_ref": v_ref
            }
        }
        features.append(polygon)

    return {
        "type": "FeatureCollection",
        "features": features
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data")
def data():
    return jsonify(generate_polygons())

if __name__ == "__main__":
    app.run(debug=True)
