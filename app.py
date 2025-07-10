import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AgriGuru Lite", layout="centered")

st.title("🌾 AgriGuru Lite – Smart Farming Assistant")

# --- Weather Forecast Section ---
st.subheader("🌦️ 5-Day Weather Forecast")
api_key = "0a16832edf4445ce698396f2fa890ddd"  # Your OpenWeatherMap API Key

location = st.text_input("Enter your City/District (for weather)")

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={api_key}&units=metric"
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

# --- Basic Rule-Based Crop Suggestion ---
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
    st.success("Recommended Crops: " + ", ".join(recommend_crops(season, soil)))

# --- ML-Based Crop Recommendation ---
st.subheader("🤖 ML-Based Crop Recommendation (via CSV + Random Forest)")

@st.cache_data
def load_crop_data():
    return pd.read_csv("Crop_recommendation.csv")

df = load_crop_data()

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier()
model.fit(X, y)

# Take user inputs
n = st.number_input("Nitrogen (N)", min_value=0.0)
p = st.number_input("Phosphorus (P)", min_value=0.0)
k = st.number_input("Potassium (K)", min_value=0.0)
temp = st.number_input("Temperature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
ph = st.number_input("Soil pH", min_value=0.0)
rainfall = st.number_input("Rainfall (mm)", min_value=0.0)

if st.button("Predict Best Crop"):
    input_data = [[n, p, k, temp, humidity, ph, rainfall]]
    prediction = model.predict(input_data)
    st.success(f"🌱 Recommended Crop: {prediction[0]}")
