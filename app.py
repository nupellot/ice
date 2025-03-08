from flask import Flask, jsonify, render_template
import pandas as pd

app = Flask(__name__)

df = pd.read_csv("ice_density_data.csv")
df = df.dropna(subset=["lon", "lat", "density"])
df = df[df["density"] >= 0]

# Ограничиваем объём данных, чтобы не перегружать фронтенд
if len(df) > 10000:
    df = df.sample(10000, random_state=42)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/heatmap")
def heatmap():
    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True)
