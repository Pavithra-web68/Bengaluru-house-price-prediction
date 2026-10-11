from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.joblib"

st.set_page_config(
    page_title="Bengaluru House Price Prediction",
    page_icon="🏠",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


st.title("🏠 Bengaluru House Price Prediction")
st.write(
    "Estimate a house price using a machine learning model "
    "trained on Bengaluru housing data."
)

bundle = load_model()

if bundle is None:
    st.error("Model not found. Run train_house_price_model.py first.")
    st.stop()

model = bundle["model"]
locations = bundle["locations"]
area_types = bundle["area_types"]

st.subheader("Enter house details")

location = st.selectbox("Location", locations)
area_type = st.selectbox("Area type", area_types)
bhk = st.number_input("Bedrooms (BHK)", min_value=1, max_value=20, value=2)
bath = st.number_input("Bathrooms", min_value=1, max_value=20, value=2)
total_sqft = st.number_input(
    "Area (square feet)",
    min_value=100,
    max_value=100000,
    value=1200,
    step=50,
)

if st.button("Predict House Price"):
    input_data = pd.DataFrame([{
        "location": location,
        "area_type": area_type,
        "total_sqft": float(total_sqft),
        "bath": float(bath),
        "bhk": float(bhk),
    }])

    predicted_price = float(model.predict(input_data)[0])

    st.success(f"Estimated price: ₹{predicted_price:.2f} lakhs")
    