import streamlit as st
import pandas as pd
import requests
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AgriGuru Lite", layout="centered")
st.title("🌾 AgriGuru Lite – Smart Farming Assistant")

# ---------------- LOAD PRODUCTION DATA ----------------
@st.cache_data
def load_production_data():
    return pd.read_csv("crop_production.csv")

try:
    prod_df = load_production_data()

    state_filter = st.selectbox("🌍 Select State", prod_df["State_Name"].dropna().unique())
    district_filter = st.selectbox("🏞️ Select District", prod_df[prod_df["State_Name"] == state_filter]["District_Name"].dropna().unique())
    season_filter = st.selectbox("🗓️ Select Season", prod_df["Season"].dropna().unique())

    st.markdown(f"### 📍 Selected Region: **{district_filter}, {state_filter}** | Season: **{season_filter}**")
except FileNotFoundError:
    st.warning("Please upload `crop_production.csv`.")

# ---------------- WEATHER FORECAST ----------------
st.subheader("🌦️ 5-Day Weather Forecast")
weather_api_key = "0a16832edf4445ce698396f2fa890ddd"

district_to_city = {
    "MALDAH": "Malda",
    "BARDHAMAN": "Bardhaman",
    "NADIA": "Krishnanagar",
    "24 PARAGANAS NORTH": "Barasat",
    "24 PARAGANAS SOUTH": "Diamond Harbour",
    "HOWRAH": "Howrah",
    "KOLKATA": "Kolkata"
}

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={weather_api_key}&units=metric"
    res = requests.get(url)
    if res.status_code == 200:
        return res.json()['list'][:5]
    return None

if district_filter:
    city_query = district_to_city.get(district_filter.upper(), district_filter)
    forecast = get_weather(city_query)
    if forecast:
        for day in forecast:
            st.write(f"{day['dt_txt']} | 🌡️ {day['main']['temp']}°C | {day['weather'][0]['description']}")
    else:
        st.warning("⚠️ Weather unavailable. Try entering a nearby city manually.")

# ---------------- SOIL TYPE BUTTONS ----------------
st.subheader("🧱 Explore Suitable Crops by Soil Type")

soil_crop_map = {
    "Alluvial": ["Rice", "Sugarcane", "Wheat", "Jute"],
    "Black": ["Cotton", "Soybean", "Sorghum"],
    "Red": ["Millets", "Groundnut", "Potato"],
    "Laterite": ["Cashew", "Tea", "Tapioca"],
    "Sandy": ["Melons", "Pulses", "Groundnut"],
    "Clayey": ["Rice", "Wheat", "Lentil"],
    "Loamy": ["Maize", "Barley", "Sugarcane"]
}

soil_col1, soil_col2, soil_col3 = st.columns(3)
with soil_col1:
    if st.button("Alluvial"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Alluvial"]))
with soil_col2:
    if st.button("Black"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Black"]))
with soil_col3:
    if st.button("Red"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Red"]))

soil_col4, soil_col5, soil_col6 = st.columns(3)
with soil_col4:
    if st.button("Laterite"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Laterite"]))
with soil_col5:
    if st.button("Sandy"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Sandy"]))
with soil_col6:
    if st.button("Clayey"):
        st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Clayey"]))

if st.button("Loamy"):
    st.info("🌾 Suitable Crops: " + ", ".join(soil_crop_map["Loamy"]))

st.divider()

# ---------------- USER INPUT FOR ML ----------------
st.markdown("### 📥 Enter Soil and Climate Data (for ML Prediction)")
n = st.number_input("Nitrogen", min_value=0.0)
p = st.number_input("Phosphorous", min_value=0.0)
k = st.number_input("Potassium", min_value=0.0)
temp = st.number_input("Temparature (°C)", min_value=0.0)
humidity = st.number_input("Humidity (%)", min_value=0.0)
moisture = st.number_input("Moisture (%)", min_value=0.0)

# ---------------- ML MODEL: FILTERED BY DISTRICT CROPS ----------------
st.subheader("🌿 ML-Powered Crop Recommendation (Filtered by District)")

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_soil_dataset():
    df = pd.read_csv("data_core.csv")
    le = LabelEncoder()
    df["soil_encoded"] = le.fit_transform(df["Soil Type"])
    
    features = ["Nitrogen", "Phosphorous", "Potassium", "Temparature", "Humidity", "Moisture", "soil_encoded"]
    X = df[features]
    y = df["Crop Type"]

    # Base XGBoost model
    base_model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )

    # Wrap with CalibratedClassifierCV
    calibrated_model = CalibratedClassifierCV(base_estimator=base_model, method='sigmoid', cv=5)
    calibrated_model.fit(X, y)

    return calibrated_model, le, df


try:
    soil_model, soil_encoder, soil_df = load_soil_dataset()

    # Show translated soil types for selection
    soil_display = [_(s) for s in soil_df["Soil Type"].unique()]
    selected_soil_display = st.selectbox(_("🧪 Select Soil Type for ML"), soil_display)
    selected_soil = soil_df["Soil Type"].unique()[soil_display.index(selected_soil_display)]

    if st.button(_("🌱 Predict Best Crops in District")):
        encoded_soil = soil_encoder.transform([selected_soil])[0]
        input_data = [[n, p, k, temp, humidity, moisture, encoded_soil]]

        # Translate district/state to English (in case user selected in another language)
        try:
            selected_district_en = GoogleTranslator(source=target_lang, target='en').translate(selected_district)
            selected_state_en = GoogleTranslator(source=target_lang, target='en').translate(selected_state)
        except:
            selected_district_en = selected_district
            selected_state_en = selected_state

        # Filter crops grown in selected district
        district_crops = prod_df[
            (prod_df["District_Name"].str.lower() == selected_district_en.lower()) &
            (prod_df["State_Name"].str.lower() == selected_state_en.lower())
        ]["Crop"].dropna().unique()

        # Predict probabilities for all crops
        proba = soil_model.predict_proba(input_data)[0]
        labels = soil_model.classes_
        crop_scores = {label: prob for label, prob in zip(labels, proba)}

        # Keep only crops from this district
        recommended = [(crop, crop_scores[crop]) for crop in district_crops if crop in crop_scores]
        recommended = sorted(recommended, key=lambda x: x[1], reverse=True)[:5]

        if recommended:
            st.success(_("✅ Top Recommended Crops Grown in Your District:"))
            for crop, score in recommended:
                st.write(f"🌿 {_(crop)} — {_('Confidence')}: {score * 100:.1f}%")
        else:
            st.warning(_("❌ No matching crops from prediction found in this district."))

except FileNotFoundError:
    st.warning(_("⚠ Please upload data_core.csv."))
