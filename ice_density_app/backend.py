from flask import Flask, jsonify, render_template, request
import pandas as pd

app = Flask(__name__)

df = pd.read_csv('density.csv', parse_dates=['time'])
df['date'] = df['time'].dt.date

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    date_str = request.args.get('date')
    if not date_str:
        date_str = '2019-01-08'  # дата по умолчанию

    try:
        date_filter = pd.to_datetime(date_str).date()
    except Exception:
        return jsonify([])  # вернуть пустой список, если дата некорректна

    filtered = df[df['date'] == date_filter]
    points = filtered[['longitude', 'latitude', 'density']].to_dict(orient='records')
    return jsonify(points)

if __name__ == '__main__':
    app.run(debug=True)
