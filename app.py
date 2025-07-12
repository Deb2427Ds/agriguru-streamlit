import streamlit as st
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from deep_translator import GoogleTranslator

# ---------------- LANGUAGE TRANSLATION SETUP ----------------
languages = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Marathi": "mr",
    "Tamil": "ta"
}

st.set_page_config(page_title="AgriGuru Lite", layout="wide")
st.markdown("""
    <style>
    .stButton button { width: 100%; border-radius: 10px; font-weight: bold; }
    .stSelectbox > div { border-radius: 10px; }
    .block-container { padding-top: 2rem; }
    .css-18e3th9 { background: linear-gradient(to right, #dce35b, #45b649); padding: 1rem; border-radius: 12px; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #2f4f4f; }
    </style>
""", unsafe_allow_html=True)

# Sidebar: Language selection
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/606/606807.png", width=80)
st.sidebar.title("🌐 Language Settings")
selected_lang = st.sidebar.selectbox("Select Language", list(languages.keys()))
target_lang = languages[selected_lang]

translator_cache = {}
def _(text):
    if target_lang == "en":
        return text
    if (text, target_lang) in translator_cache:
        return translator_cache[(text, target_lang)]
    try:
        translated = GoogleTranslator(source='en', target=target_lang).translate(text)
        translator_cache[(text, target_lang)] = translated
        return translated
    except:
        return text

# ---------------- HEADER ----------------
st.markdown(f"<h1 style='text-align: center; color: #006400;'>{_('🌾 AgriGuru Lite – Smart Farming Assistant')}</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #228B22;'>", unsafe_allow_html=True)

# ---------------- ML MODEL LOADING ----------------
@st.cache_data
def load_soil_dataset():
    df = pd.read_csv("data_core.csv")

    # Rename for price clarity
    df.rename(columns={"Production (tonnes)": "Crop Price"}, inplace=True)

    # Clean price values
    df["Crop Price"] = (
        df["Crop Price"].astype(str)
        .str.replace(",", "")
        .str.extract(r"(\d+)")
        .fillna(0)
        .astype(int)
    )

    le = LabelEncoder()
    df["soil_encoded"] = le.fit_transform(df["Soil Type"])
    features = ["Nitrogen", "Phosphorous", "Potassium", "Temparature", "Humidity", "Moisture", "soil_encoded"]
    X = df[features]
    y = df["Crop Type"]
    model = RandomForestClassifier()
    model.fit(X, y)
    return model, le, df

# ---------------- USER INPUT ----------------
st.markdown(f"### 📊 { _('Enter Soil and Climate Data') }")
col1, col2, col3 = st.columns(3)
with col1:
    n = st.number_input("Nitrogen", min_value=0.0)
    p = st.number_input("Phosphorous", min_value=0.0)
with col2:
    k = st.number_input("Potassium", min_value=0.0)
    temp = st.number_input("Temparature (°C)", min_value=0.0)
with col3:
    humidity = st.number_input("Humidity (%)", min_value=0.0)
    moisture = st.number_input("Moisture (%)", min_value=0.0)

st.markdown(f"### 💰 { _('Budget Filter') }")
max_price = st.number_input(_("Enter Maximum Price per Tonne (₹)"), min_value=0)

# ---------------- PREDICTION ----------------
try:
    model, encoder, df = load_soil_dataset()

    soil_display = [_(s) for s in df["Soil Type"].unique()]
    selected_soil_display = st.selectbox("🧪 Select Soil Type", soil_display)
    selected_soil = df["Soil Type"].unique()[soil_display.index(selected_soil_display)]

    if st.button("🌱 Predict Best Crops"):
        encoded_soil = encoder.transform([selected_soil])[0]
        input_data = [[n, p, k, temp, humidity, moisture, encoded_soil]]
        proba = model.predict_proba(input_data)[0]
        labels = model.classes_
        crop_scores = {label: prob for label, prob in zip(labels, proba)}

        top_crops = sorted(crop_scores.items(), key=lambda x: x[1], reverse=True)

        if max_price > 0:
            price_map = dict(zip(df["Crop Type"], df["Crop Price"]))
            top_crops = [(c, s) for c, s in top_crops if price_map.get(c, 1e9) <= max_price]

        if top_crops:
            st.success("✅ Recommended Crops:")
            for crop, score in top_crops[:5]:
                st.write(f"🌿 {crop} — Confidence: {score:.2%} | Price: ₹{price_map[crop]:,}/tonne")
        else:
            st.warning("❌ No crops match your budget or input conditions.")
except FileNotFoundError:
    st.warning("⚠ Please upload 'data_core.csv' file in your project directory.")
