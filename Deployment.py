import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Air Quality Predictor", page_icon="🌬️", layout="centered")

st.title("🌬️ Air Quality Classification App")
st.write("""
This application utilizes a Machine Learning model to predict ambient air quality 
levels based on environmental and pollutant metrics.
""")

@st.cache_resource
def load_model():
    return joblib.load("Air_Quality_Prediction.pkl")

model = load_model()

st.subheader("📊 Enter Environmental & Pollutant Metrics")

col1, col2 = st.columns(2)

with col1:
    temperature = st.slider("Temperature (°C)", min_value=10.0, max_value=60.0, value=30.0, step=0.1)
    humidity = st.slider("Humidity (%)", min_value=30.0, max_value=130.0, value=70.0, step=0.1)
    pm25 = st.number_input("PM2.5 Concentration (µg/m³)", min_value=0.0, max_value=300.0, value=20.0)
    pm10 = st.number_input("PM10 Concentration (µg/m³)", min_value=0.0, max_value=320.0, value=30.0)
    no2 = st.number_input("NO2 Concentration (ppb)", min_value=0.0, max_value=70.0, value=26.0)

with col2:
    so2 = st.number_input("SO2 Concentration (ppb)", min_value=0.0, max_value=50.0, value=10.0)
    co = st.number_input("CO Concentration (ppm)", min_value=0.0, max_value=5.0, value=1.5, step=0.01)
    industrial_prox = st.slider("Proximity to Industrial Areas (km)", min_value=0.0, max_value=30.0, value=8.5, step=0.1)
    pop_density = st.number_input("Population Density (people/km²)", min_value=100, max_value=1000, value=500, step=1)

input_data = pd.DataFrame({
    'Temperature': [temperature],
    'Humidity': [humidity],
    'PM2.5': [pm25],
    'PM10': [pm10],
    'NO2': [no2],
    'SO2': [so2],
    'CO': [co],
    'Proximity_to_Industrial_Areas': [industrial_prox],
    'Population_Density': [pop_density]
})

st.markdown("---")

if st.button("🔮 Predict Air Quality Level", use_container_width=True):
    prediction = model.predict(input_data)[0]
    with st.spinner("Calculating..."):        
        st.success(f"🔮 **Predicted Value:** {prediction}")