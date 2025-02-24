import streamlit as st
import pandas as pd
import joblib
import time
from deep_translator import GoogleTranslator
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_path = 'crop_production_model13.sav'
with open(model_path, 'rb') as file:
    model = joblib.load(file)

# Load dataset to fit encoders
dataset_path = 'Crop_Production_final_set.csv.csv'
df = pd.read_csv(dataset_path)

# Initialize label encoders
label_encoders = {}
categorical_features = ['District', 'Crop', 'Season']

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Supported languages
languages = {
    "English": "en",
    "తెలుగు": "te",
    "हिन्दी": "hi",
    "தமிழ்": "ta",
}

# Sidebar: Language Selection
selected_language = st.sidebar.selectbox("🌐 Select Language", list(languages.keys()))

def translate(text, target_lang):
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        time.sleep(2)
        return text  # Return original text if translation fails

def t(text):
    return translate(text, languages[selected_language])

# Original values
districts = ['ANANTAPUR', 'EAST GODAVARI', 'KRISHNA', 'VIZIANAGARAM', 'WEST GODAVARI',
             'ADILABAD', 'CHITTOOR', 'GUNTUR', 'KADAPA', 'KARIMNAGAR', 'KHAMMAM',
             'KURNOOL', 'MAHBUBNAGAR', 'MEDAK', 'NALGONDA', 'NIZAMABAD', 'PRAKASAM',
             'RANGAREDDI', 'SPSR NELLORE', 'SRIKAKULAM', 'VISAKHAPATANAM', 'WARANGAL', 'HYDERABAD']

crops = ['Arecanut', 'Arhar/Tur', 'Bajra', 'Banana', 'Cashewnut', 'Castor seed', 'Coconut',
         'Coriander', 'Cotton(lint)', 'Dry chillies', 'Ginger', 'Gram', 'Groundnut', 'Horse-gram',
         'Jowar', 'Linseed', 'Maize', 'Mesta', 'Moong(Green Gram)', 'Niger seed', 'Onion',
         'Other Rabi pulses', 'Other Kharif pulses', 'other oilseeds', 'Potato', 'Ragi',
         'Rapeseed & Mustard', 'Rice', 'Safflower', 'Sesamum', 'Small millets', 'Soyabean',
         'Sugarcane', 'Sunflower', 'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric',
         'Urad', 'Wheat', 'Garlic', 'Cowpea(Lobia)', 'Black pepper', 'Oilseeds total',
         'Sannhamp', 'Guar seed', 'Masoor']

seasons = ['Whole Year', 'Kharif', 'Rabi']

# Translate dropdown values
districts_translated = [t(d) for d in districts]
crops_translated = [t(c) for c in crops]
seasons_translated = [t(s) for s in seasons]

st.title(t("🌾 Crop Production Prediction App"))

# User Inputs
district = st.selectbox(t("🌍 Select District"), districts_translated)
crop = st.selectbox(t("🌱 Select Crop"), crops_translated)
season = st.selectbox(t("📅 Select Season"), seasons_translated)
area = st.number_input(t("📍 Enter Area (in hectares) ( 1 Hectare = 2.47 acres)"), min_value=0.1, format="%.2f")

# Convert translated values back to original for model input
district_original = districts[districts_translated.index(district)]
crop_original = crops[crops_translated.index(crop)]
season_original = seasons[seasons_translated.index(season)]

def predict_production(district, crop, season, area):
    try:
        district_encoded = label_encoders['District'].transform([district])[0]
        crop_encoded = label_encoders['Crop'].transform([crop])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]
    except ValueError:
        return "Error: Input values not in training data"
    
    input_data = pd.DataFrame([[district_encoded, crop_encoded, season_encoded, area]],
                              columns=['District', 'Crop', 'Season', 'Area'])
    prediction = model.predict(input_data)
    return prediction[0]

if st.button(t("🔮 Predict")):
    result = predict_production(district_original, crop_original, season_original, area)
    if isinstance(result, str) and "Error" in result:
        st.error(t(result))
    else:
        st.success(f"⚖️ {t('Predicted Production')}: {result:.2f} {t('Tonnes')}")
