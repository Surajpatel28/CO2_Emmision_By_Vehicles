from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from flask_cors import CORS

 

app=Flask(__name__)
CORS(app) 

model= joblib.load("co2_emission_model.pkl")

# Load mean and std
scaler_values = np.load("scaler_values.npy", allow_pickle=True).item()
mean = scaler_values["mean"]
std = scaler_values["std"]
selected_features = scaler_values["features"]

@app.route("/")
def home():
    return "CO2 Emission Prediction API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        user_input = np.array([[
            float(data["engine_size"]),
            float(data["fuel_consumption"]),
            int(data["fuel_type"] == "Fuel Type_N"),
            int(data["fuel_type"] == "Fuel Type_X"),
            int(data["fuel_type"] == "Fuel Type_Z"),
            int(data["gears"])
        ]])
    except KeyError as e:
        return jsonify({"error": f"Missing feature: {str(e)}"}), 400
    except ValueError:
        return jsonify({"error": "Invalid input type. Please enter correct values."}), 400

    # Normalize manually using saved mean & std
    features_normalized = (user_input - mean) / std

    prediction = model.predict(features_normalized)[0]
    co2_emission = round(float(prediction), 2)  # Convert to float and round to 2 decimal places

    # CO2 Emission Assessment
    if co2_emission < 100:
        assessment = "Excellent. The vehicle is very eco-friendly with low CO₂ emissions."
    elif co2_emission < 150:
        assessment = "Good. The car has reasonable CO₂ emissions, making it an efficient choice."
    elif co2_emission < 200:
        assessment = "Moderate. The car emits a fair amount of CO₂. Consider a more fuel-efficient option."
    else:
        assessment = "High. The vehicle has high CO₂ emissions, which may contribute significantly to pollution."

    return jsonify({
        "co2_emission": f"{co2_emission} g/km",
        "assessment": assessment,
        "note": "CO₂ emissions represent tailpipe emissions (in g/km) for combined city and highway driving."
    })


if __name__ == "__main__":
    app.run(debug=True)