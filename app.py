import streamlit as st  
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AgriGuru Lite", layout="centered")
st.title("🌾 AgriGuru Lite – Smart Farming Assistant")

# ---------------- CROP PRODUCTION DATA ----------------
@st.cache_data
def load_production_data():
    return pd.read_csv("crop_production.csv")

try:
    prod_df = load_production_data()

    state_filter = st.selectbox("🌍 Select State", prod_df["State_Name"].dropna().unique())
    district_filter = st.selectbox("🏞️ Select District", prod_df[prod_df["State_Name"] == state_filter]["District_Name"].dropna().unique())
    season_filter = st.selectbox("🗓️ Select Season", prod_df["Season"].dropna().unique())

    st.markdown(f"### 📍 Selected Region: **{district_filter}, {state_filter}** | Season: **{season_filter}**")

    filtered = prod_df[
        (prod_df["State_Name"] == state_filter) &
        (prod_df["District_Name"] == district_filter) &
        (prod_df["Season"] == season_filter)
    ]

    if not filtered.empty:
        st.dataframe(filtered[["Crop_Year", "Crop", "Area", "Production"]])
    else:
        st.warning("No production data found for selected location.")
except FileNotFoundError:
    st.warning("Please make sure crop_production.csv is uploaded.")

# ---------------- WEATHER FORECAST ----------------
st.subheader("🌦️ 5-Day Weather Forecast")
weather_api_key = "0a16832edf4445ce698396f2fa890ddd"

def get_weather(place):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={place}&appid={weather_api_key}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()['list'][:5]
    return None

if district_filter:
    forecast = get_weather(district_filter)
    if forecast:
        for day in forecast:
            st.write(f"{day['dt_txt']} | 🌡️ {day['main']['temp']}°C | {day['weather'][0]['description']}")
    else:
        st.warning("Couldn't fetch weather. Please check the district name.")

# ---------------- USER INPUT FOR ML ----------------
st.markdown("### 📥 Enter Soil and Climate Data (for ML Prediction)")
n = st.number_input("Nitrogen", min_value=0.0)
p = st.number_input("Phosphorous", min_value=0.0)
k = st.number_input("Potassium", min_value=0.0)
temp = st.number_input("Temparature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
moisture = st.number_input("Moisture (%)", min_value=0.0)

# ---------------- ML MODEL USING data_core.csv ----------------
st.subheader("🌿 Smart Crop Prediction (from data_core.csv)")

@st.cache_data
def load_soil_dataset():
    df = pd.read_csv("data_core.csv")
    le = LabelEncoder()
    df["soil_encoded"] = le.fit_transform(df["Soil Type"])
    features = ["Nitrogen", "Phosphorous", "Potassium", "Temparature", "Humidity", "Moisture", "soil_encoded"]
    X = df[features]
    y = df["Crop Type"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, le, df

try:
    soil_model, soil_encoder, soil_df = load_soil_dataset()
    soil_input = st.selectbox("🧪 Select Soil Type for Prediction", soil_df["Soil Type"].unique())

    if st.button("Predict Best Crop"):
        encoded_soil = soil_encoder.transform([soil_input])[0]
        input_data = [[n, p, k, temp, humidity, moisture, encoded_soil]]
        crop_prediction = soil_model.predict(input_data)[0]
        st.success(f"🌱 ML Predicted Crop: **{crop_prediction}**")
except FileNotFoundError:
    st.warning("Please make sure data_core.csv is uploaded.")
