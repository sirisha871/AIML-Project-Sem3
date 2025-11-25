import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# --------------------------------------------------------
# LOAD MODEL + ENCODERS + DATA
# --------------------------------------------------------
@st.cache_resource
def load_model():
    model = xgb.Booster()
    model.load_model("crop_yield_xgboost_model.json")
    return model

@st.cache_resource
def load_encoders():
    le_area = joblib.load("encoder_country.pkl")
    le_crop = joblib.load("encoder_crop.pkl")
    return le_area, le_crop

@st.cache_resource
def load_data():
    df = pd.read_csv("yield_df.csv")
    return df

model = load_model()
le_area, le_crop = load_encoders()
df = load_data()

# --------------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------------
st.set_page_config(page_title="Smart Agriculture Yield Predictor", layout="centered")
st.title("🌱 Smart Agriculture & Crop Yield Prediction")
st.write("Predict crop yield based on country, crop type, and environmental factors.")

st.markdown("---")

# --------------------------------------------------------
# INPUT FIELDS
# --------------------------------------------------------
country = st.selectbox("Select Country", sorted(df['Area'].unique()))
crop = st.selectbox("Select Crop", sorted(df['Item'].unique()))
year = st.number_input("Year", min_value=1960, max_value=2050, value=2025)

st.subheader("Optional Environmental Inputs")
rainfall = st.number_input("Rainfall (mm/year)", min_value=0.0, value=0.0)
pesticides = st.number_input("Pesticides (tonnes)", min_value=0.0, value=0.0)
temp = st.number_input("Average Temperature (°C)", min_value=-10.0, value=0.0)

# --------------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------------
def predict_yield(country, crop, year, rainfall, pesticides, temp):
    try:
        area_code = le_area.transform([country])[0]
        item_code = le_crop.transform([crop])[0]
    except:
        return None, "❌ Error: Country or crop not found in training data."

    # Auto-fill missing values
    if rainfall == 0:
        rainfall = df[df['Area'] == country]['average_rain_fall_mm_per_year'].mean()

    if pesticides == 0:
        pesticides = df[df['Area'] == country]['pesticides_tonnes'].mean()

    if temp == 0:
        temp = df['avg_temp'].mean()

    # DataFrame with feature names
    input_df = pd.DataFrame([{
        'Area_encoded': area_code,
        'Item_encoded': item_code,
        'Year': year,
        'average_rain_fall_mm_per_year': rainfall,
        'pesticides_tonnes': pesticides,
        'avg_temp': temp
    }])

    dmatrix = xgb.DMatrix(input_df)
    prediction = model.predict(dmatrix)[0]
    
    return prediction, None

# --------------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------------
if st.button("Predict Yield"):
    pred, error = predict_yield(country, crop, year, rainfall, pesticides, temp)

    if error:
        st.error(error)
    else:
        st.success(f"🌾 **Predicted Yield:** {pred:,.0f} hg/ha")
        st.write("Higher values indicate better productivity per hectare.")

        st.markdown("---")
        st.info("""
### Interpretation  
- **hg/ha (hectogram per hectare)** is a standard FAO unit.  
- More rainfall, balanced pesticides, and proper climate generally improve yield.
        """)

# Footer
st.markdown("---")
st.caption("Developed using Machine Learning + XGBoost + Streamlit 🌱")
