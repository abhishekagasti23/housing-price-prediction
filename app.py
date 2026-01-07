from flask import Flask, request, render_template_string, jsonify
import numpy as np
import pickle

app = Flask(__name__)


# Load trained artifacts

with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# HTML Template

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Prediction</title>
    <style>
        body {
            font-family: Arial;
            background-color: #f4f6f8;
            padding: 40px;
        }
        .container {
            max-width: 650px;
            background: white;
            padding: 30px;
            margin: auto;
            border-radius: 8px;
            box-shadow: 0 0 12px rgba(0,0,0,0.1);
        }
        input, select, button {
            width: 100%;
            padding: 10px;
            margin-top: 12px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            border: none;
            font-size: 16px;
            cursor: pointer;
        }
        h2 {
            text-align: center;
        }
        .result {
            margin-top: 20px;
            font-size: 20px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>House Price Prediction</h2>

        <form method="post">
            <input name="features" placeholder="longitude, latitude, age, rooms, bedrooms, population, households, income" required>

            <select name="ocean_proximity">
                <option value="INLAND">INLAND</option>
                <option value="NEAR BAY">NEAR BAY</option>
                <option value="NEAR OCEAN">NEAR OCEAN</option>
                <option value="<1H OCEAN">&lt;1H OCEAN</option>
            </select>

            <button type="submit">Predict</button>
        </form>

        {% if prediction %}
        <div class="result">
            <strong>Predicted Price:</strong> {{ prediction }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# Web UI Route

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            raw_input = request.form["features"]
            numeric_features = [float(x) for x in raw_input.split(",")]

            if len(numeric_features) != 8:
                raise ValueError("Please enter exactly 8 numeric values.")

            ocean_proximity = request.form["ocean_proximity"]
            categories = ["INLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN"]

            encoded_ocean = [1 if ocean_proximity == c else 0 for c in categories]

            final_features = np.concatenate(
                [numeric_features, encoded_ocean]
            ).reshape(1, -1)

            scaled = scaler.transform(final_features)
            pred = model.predict(scaled)[0]

            prediction = f"{pred:,.2f}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

# Optional API (No Postman needed)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()

    numeric_features = np.array(data["numeric_features"])
    ocean_proximity = data["ocean_proximity"]

    categories = ["INLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN"]
    encoded_ocean = [1 if ocean_proximity == c else 0 for c in categories]

    final_features = np.concatenate(
        [numeric_features, encoded_ocean]
    ).reshape(1, -1)

    scaled = scaler.transform(final_features)
    prediction = model.predict(scaled)[0]

    return jsonify({"prediction": float(prediction)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
