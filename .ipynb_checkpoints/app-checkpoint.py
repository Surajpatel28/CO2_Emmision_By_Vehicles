import streamlit as st
import pandas as pd
import joblib

model = joblib.load("linear_model.pkl")
scaler = joblib.load("scaler.pkl")
power_transformer = joblib.load("power_transformer.pkl")

# List of columns used during model training
model_cols = joblib.load("model_columns.pkl")  # This is a list of column names in the correct order

st.title("🚗 CO₂ Emissions Predictor")

st.write("Predict the CO₂ emissions of a car based on its specifications.")

# User inputs
cylinders = st.number_input("Number of Cylinders", min_value=2, max_value=16, value=4)
fuel_consumption = st.number_input("Fuel Consumption (L/100km)", min_value=1.0, max_value=30.0, value=7.5)
engine_size = st.number_input("Engine Size (Liters)", min_value=0.5, max_value=10.0, value=2.0)

# Transmission options
transmission_options = {
    "A4 (Automatic 4-speed)": "Transmission_A4",
    "A5 (Automatic 5-speed)": "Transmission_A5",
    "A6 (Automatic 6-speed)": "Transmission_A6",
    "A7 (Automatic 7-speed)": "Transmission_A7",
    "A8 (Automatic 8-speed)": "Transmission_A8",
    "A9 (Automatic 9-speed)": "Transmission_A9",
    "AM6 (Automated Manual 6-speed)": "Transmission_AM6",
    "AM7 (Automated Manual 7-speed)": "Transmission_AM7",
    "AM8 (Automated Manual 8-speed)": "Transmission_AM8",
    "AS5 (Auto Select 5-speed)": "Transmission_AS5",
    "AS6 (Auto Select 6-speed)": "Transmission_AS6",
    "AS7 (Auto Select 7-speed)": "Transmission_AS7",
    "AS8 (Auto Select 8-speed)": "Transmission_AS8",
    "AS9 (Auto Select 9-speed)": "Transmission_AS9",
    "AS10 (Auto Select 10-speed)": "Transmission_AS10",
    "AV (CVT)": "Transmission_AV",
    "AV6 (CVT 6-speed)": "Transmission_AV6",
    "AV7 (CVT 7-speed)": "Transmission_AV7",
    "AV8 (CVT 8-speed)": "Transmission_AV8",
    "M5 (Manual 5-speed)": "Transmission_M5",
    "M6 (Manual 6-speed)": "Transmission_M6",
    "M7 (Manual 7-speed)": "Transmission_M7"
}
transmission_choice = st.selectbox("Transmission Type", list(transmission_options.keys()))
transmission_feature = transmission_options[transmission_choice]
transmission_features = {col: False for col in transmission_options.values()}
transmission_features[transmission_feature] = True

# Fuel type options
fuel_type_options = {
    "Regular Gasoline (X)": "Fuel Type_X",
    "Premium Gasoline (Z)": "Fuel Type_Z",
    "Diesel (D)": "Fuel Type_D",
    "Ethanol (E85) (E)": "Fuel Type_E",
    "Natural Gas (N)": "Fuel Type_N"
}
fuel_type_choice = st.selectbox("Fuel Type", list(fuel_type_options.keys()))
fuel_type_feature = fuel_type_options[fuel_type_choice]
fuel_type_features = {col: False for col in fuel_type_options.values()}
fuel_type_features[fuel_type_feature] = True

# Process Engine Size with PowerTransformer
transformed_engine_size = power_transformer.transform([[engine_size]])[0][0]

# Combine all features into one input dictionary
input_dict = {
    "Cylinders": cylinders,
    "Fuel_Consumption_Comb_L_per_100km": fuel_consumption,
    "Engine_Size_L": transformed_engine_size,
    **transmission_features,
    **fuel_type_features
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Ensure all expected model columns exist
for col in model_cols:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing column with 0
input_df = input_df[model_cols]  # Ensure correct column order

# Scale the input
scaled_input = scaler.transform(input_df)

# Predict
if st.button("Predict CO₂ Emissions"):
    prediction = model.predict(scaled_input)[0]
    st.success(f"Estimated CO₂ Emissions: **{prediction:.2f} g/km**")
