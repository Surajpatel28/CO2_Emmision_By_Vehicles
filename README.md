# 🚗 CO₂ Emission Predictor for Vehicles

Predict the CO₂ emissions of a vehicle based on its specifications using a trained machine learning model.

### 🌐 Live Demo  
**Deployed on Streamlit:**  
👉 [Click here to try it out](https://co2emmisionbyvehicles.streamlit.app/)

---

## 📌 Features

- Predicts CO₂ emissions using vehicle specifications like:
  - Engine Size
  - Number of Cylinders
  - Fuel Type
  - Transmission Type
  - Fuel Consumption (Combined)
- Interactive UI with real-time predictions
- Built with **Streamlit** for fast and responsive deployment

---

## 🧠 Model Info

- **Algorithm**: Linear Regression
- **Preprocessing**:
  - `PowerTransformer (yeo-johnson)` applied on engine size
  - `StandardScaler` for normalization
- Trained on a cleaned dataset with vehicle specifications and CO₂ emissions

---

---

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/surajpatel28/co2_emmision_by_vehicles.git
   cd co2_emmision_by_vehicles
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the Streamlit app
```bash
streamlit run app.py
```
