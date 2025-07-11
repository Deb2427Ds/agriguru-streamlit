import streamlit as st  
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AgriGuru Lite", layout="centered")
st.title("🌾 AgriGuru Lite – Smart Farming Assistant")

# ---------------- WEATHER FORECAST ----------------
st.subheader("🌦️ 5-Day Weather Forecast")
weather_api_key = "0a16832edf4445ce698396f2fa890ddd"

location = st.text_input("Enter your City/District (for weather)")

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={weather_api_key}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()['list'][:5]
    return None

if location:
    forecast = get_weather(location)
    if forecast:
        for day in forecast:
            st.write(f"{day['dt_txt']} | 🌡️ {day['main']['temp']}°C | {day['weather'][0]['description']}")
    else:
        st.warning("Couldn't fetch weather. Please check the city name.")

# ---------------- RULE-BASED CROP RECOMMENDATION ----------------
st.subheader("🧠 Rule-Based Crop Recommendation")

season = st.selectbox("Select the Crop Season", ["Kharif", "Rabi", "Zaid"])
soil = st.selectbox("Select Soil Type", ["Alluvial", "Black", "Red", "Laterite", "Sandy", "Clayey"])

def recommend_crops(season, soil):
    if season == "Kharif" and soil == "Alluvial":
        return ["Paddy", "Maize", "Jute"]
    elif season == "Rabi" and soil == "Black":
        return ["Wheat", "Barley", "Gram"]
    elif season == "Zaid":
        return ["Watermelon", "Cucumber", "Bitter Gourd"]
    else:
        return ["Millets", "Pulses", "Sunflower"]

if season and soil:
    rule_based = recommend_crops(season, soil)
    st.success("Recommended Crops: " + ", ".join(rule_based))

# ---------------- USER INPUTS FOR ML MODELS ----------------
st.markdown("**Enter Soil and Climate Data for ML Models**")
n = st.number_input("Nitrogen (N)", min_value=0.0)
p = st.number_input("Phosphorus (P)", min_value=0.0)
k = st.number_input("Potassium (K)", min_value=0.0)
temp = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("Soil pH", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

# ---------------- ADVANCED MODEL: SOIL TYPE INCLUDED ----------------
st.subheader("🧪 Advanced Prediction (from data_core.csv)")

@st.cache_data
def load_soil_dataset():
    df = pd.read_csv("data_core.csv")
    le = LabelEncoder()
    df["soil_encoded"] = le.fit_transform(df["soil_type"])
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall", "soil_encoded"]
    X = df[features]
    y = df["label"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, le, df

try:
    soil_model, soil_encoder, soil_df = load_soil_dataset()
    soil_input = st.selectbox("Select Soil Type for ML Model", soil_df["soil_type"].unique())

    if st.button("Predict Crop (Soil-Aware Model)"):
        encoded_soil = soil_encoder.transform([soil_input])[0]
        input_data = [[n, p, k, temp, humidity, ph, rainfall, encoded_soil]]
        soil_prediction = soil_model.predict(input_data)[0]
        st.success(f"🌿 Predicted Crop (with Soil Type): **{soil_prediction}**")
except FileNotFoundError:
    st.warning("Please make sure data_core.csv is uploaded.")

# ---------------- PRODUCTION DATA VIEWER (crop_production.csv) ----------------
st.subheader("📊 Crop Production Insights (from crop_production.csv)")

@st.cache_data
def load_production_data():
    return pd.read_csv("crop_production.csv")

try:
    prod_df = load_production_data()

    state_filter = st.selectbox("Filter by State", prod_df["State"].dropna().unique())
    season_filter = st.selectbox("Filter by Season", prod_df["Season"].dropna().unique())

    filtered = prod_df[
        (prod_df["State"] == state_filter) &
        (prod_df["Season"] == season_filter)
    ]

    if not filtered.empty:
        st.success(f"Showing crops produced in **{state_filter}** during **{season_filter}**:")
        st.dataframe(filtered[["District", "Crop", "Area", "Production", "Yield"]])
    else:
        st.warning("No data found for the selected filters.")
except FileNotFoundError:
    st.warning("Please make sure crop_production.csv is uploaded.")
