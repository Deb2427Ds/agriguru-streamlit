import streamlit as st  
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier

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

# ---------------- ML-BASED CROP RECOMMENDATION ----------------
st.subheader("🤖 ML-Based Crop Recommendation (via CSV + Random Forest)")

@st.cache_data
def load_crop_data():
    return pd.read_csv("Crop_recommendation.csv")

df = load_crop_data()

X = df.drop("label", axis=1)
y = df["label"]

model = RandomForestClassifier()
model.fit(X, y)

# Crop-to-Season Mapping
crop_seasons = {
    "rice": "Kharif", "maize": "Kharif", "jute": "Kharif", "cotton": "Kharif",
    "kidneybeans": "Kharif", "pigeonpeas": "Kharif", "blackgram": "Kharif", 
    "mothbeans": "Kharif", "mungbean": "Kharif",
    "wheat": "Rabi", "gram": "Rabi", "lentil": "Rabi", "chickpea": "Rabi",
    "grapes": "Rabi", "apple": "Rabi", "orange": "Rabi", "pomegranate": "Rabi",
    "watermelon": "Zaid", "muskmelon": "Zaid", "cucumber": "Zaid",
    "banana": "All Season", "mango": "All Season", "papaya": "All Season",
    "coconut": "All Season", "coffee": "All Season"
}

st.markdown("**Enter Soil and Climate Data for ML Prediction**")
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
    predicted_crop = prediction[0]
    season = crop_seasons.get(predicted_crop, "Unknown")
    st.success(f"🌱 Predicted Crop: **{predicted_crop}** ({season} season)")

# ---------------- PRICE-BASED CROP RECOMMENDATION ----------------
st.subheader("💰 Price-Based Crop Recommendation (Mandi Price API)")

user_price = st.number_input("Enter your expected crop price (₹ per quintal)", min_value=0)

def get_crop_prices():
    url = "https://api.data.gov.in/resource/f9efb243-4f43-4941-a181-0a6e54c5f295"
    params = {
        "api-key": "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b",
        "format": "json",
        "limit": 500
    }
    try:
        res = requests.get(url, params=params)
        data = res.json()
        prices = []
        for entry in data['records']:
            try:
                modal_price = int(entry['modal_price'])
                crop = entry['commodity']
                if abs(modal_price - user_price) <= 500:
                    prices.append((crop, modal_price, entry['state'], entry['market']))
            except:
                continue
        return prices
    except:
        return []

if st.button("Suggest Crops by Price"):
    if user_price <= 0:
        st.warning("Please enter a valid price.")
    else:
        matching_crops = get_crop_prices()
        if matching_crops:
            st.success("🌾 Crops with prices within ₹500 of your input:")
            for crop, price, state, market in matching_crops[:10]:
                st.write(f"**{crop}** – ₹{price} (State: {state}, Market: {market})")
        else:
            st.warning("No crops found within the given price range.")

# ---------------- ML + BUDGET FILTER ----------------
st.subheader("🧪 Predict Crops That Fit Your Budget (from uploaded dataset)")

@st.cache_data
def load_priced_dataset():
    return pd.read_csv("dataset.csv")

priced_df = load_priced_dataset()

max_budget = st.number_input("Enter your maximum budget per quintal (₹)", min_value=0)

if st.button("Predict Crop Within Budget"):
    input_data = [[n, p, k, temp, humidity, ph, rainfall]]
    predicted_crop = model.predict(input_data)[0]
    
    # Filter dataset based on predicted crop and price
    filtered = priced_df[
        (priced_df['label'] == predicted_crop) &
        (priced_df['price'] <= max_budget)
    ]

    if not filtered.empty:
        st.success(f"✅ You can grow **{predicted_crop}** within your budget!")
        st.dataframe(filtered)
    else:
        st.warning(f"❌ No data found for **{predicted_crop}** within your budget.")
