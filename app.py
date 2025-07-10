import streamlit as st
import pandas as pd
import requests

st.set_page_config(page_title="AgriGuru Lite", layout="centered")

st.title("🌾 AgriGuru Lite – Smart Farming Assistant")

# --- User Input ---
location = st.text_input("Enter your City/District (for weather)")
season = st.selectbox("Select the Crop Season", ["Kharif", "Rabi", "Zaid"])
soil = st.selectbox("Select Soil Type", ["Alluvial", "Black", "Red", "Laterite", "Sandy", "Clayey"])

# --- Crop Suggestion Logic ---
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
    st.subheader("✅ Recommended Crops")
    crops = recommend_crops(season, soil)
    st.success(", ".join(crops))

# --- Weather Forecast Section ---
st.subheader("🌦️ 5-Day Weather Forecast")
api_key = "0a16832edf4445ce698396f2fa890ddd"  # Replace this with your real API key

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

# --- Mandi Prices Section ---
st.subheader("📈 Sample Mandi Prices")

@st.cache_data

