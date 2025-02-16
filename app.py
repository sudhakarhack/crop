import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import joblib
import time  # Add this import to use time.sleep()

# Load the trained model
model_path = 'crop_production_model13.sav'  # Ensure this is the correct path
with open(model_path, 'rb') as file:
    model = joblib.load(model_path)

# Load dataset to fit encoders
dataset_path = r'Crop_Production_final_set.csv.csv'
df = pd.read_csv(dataset_path)

# Initialize label encoders
label_encoders = {}
categorical_features = ['District', 'Crop', 'Season']

for col in categorical_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

def predict_production(district, crop, season, area):
    """Transform inputs and make a prediction"""
    try:
        # Encoding inputs
        district_encoded = label_encoders['District'].transform([district])[0]
        crop_encoded = label_encoders['Crop'].transform([crop])[0]
        season_encoded = label_encoders['Season'].transform([season])[0]
    except ValueError:
        return "Error: Input values not in training data"
    
    input_data = pd.DataFrame([[district_encoded, crop_encoded, season_encoded, area]],
                              columns=['District', 'Crop', 'Season', 'Area'])
    prediction = model.predict(input_data)
    return prediction[0]

# Dropdown values
districts = ['ANANTAPUR', 'EAST GODAVARI', 'KRISHNA', 'VIZIANAGARAM',
             'WEST GODAVARI', 'ADILABAD', 'CHITTOOR', 'GUNTUR', 'KADAPA',
             'KARIMNAGAR', 'KHAMMAM', 'KURNOOL', 'MAHBUBNAGAR', 'MEDAK',
             'NALGONDA', 'NIZAMABAD', 'PRAKASAM', 'RANGAREDDI', 'SPSR NELLORE',
             'SRIKAKULAM', 'VISAKHAPATANAM', 'WARANGAL', 'HYDERABAD']

crops = ['Arecanut', 'Arhar/Tur', 'Bajra', 'Banana', 'Cashewnut',
         'Castor seed', 'Coconut', 'Coriander', 'Cotton(lint)',
         'Dry chillies', 'Ginger', 'Gram', 'Groundnut', 'Horse-gram',
         'Jowar', 'Linseed', 'Maize', 'Mesta', 'Moong(Green Gram)',
         'Niger seed', 'Onion', 'Other Rabi pulses', 'Other Kharif pulses',
         'other oilseeds', 'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice',
         'Safflower', 'Sesamum', 'Small millets', 'Soyabean', 'Sugarcane',
         'Sunflower', 'Sweet potato', 'Tapioca', 'Tobacco', 'Turmeric',
         'Urad', 'Wheat', 'Garlic', 'Cowpea(Lobia)', 'Black pepper',
         'Oilseeds total', 'Sannhamp', 'Guar seed', 'Masoor']

seasons = ['Whole Year', 'Kharif', 'Rabi']

# Streamlit UI with enhancements
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        color: #4CAF50;
        font-size: 45px;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
    }
    .sidebar-header {
        background-color: #4CAF50;
        color: white;
        font-size: 20px;
    }
    .sidebar-selectbox {
        font-size: 18px;
        margin-bottom: 20px;
    }
    .sidebar-button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
    }
    .result {
        font-size: 25px;
        color: #4CAF50;
        text-align: center;
        margin-top: 20px;
    }
    .error {
        font-size: 20px;
        color: red;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True
)

# Header
st.markdown('<div class="title">🌱 Crop Production Prediction App</div>', unsafe_allow_html=True)

# Sidebar with custom styles
st.sidebar.markdown('<div class="sidebar-header">User Input 👨</div>', unsafe_allow_html=True)

district = st.sidebar.selectbox("Select District 🌍", districts, index=0)
crop = st.sidebar.selectbox("Select Crop 🌱", crops, index=0)
season = st.sidebar.selectbox("Select Season 🌞", seasons, index=0)
area = st.sidebar.number_input("Enter Area 📍(Hectares) (1 Hectare = 2.47 acres)", min_value=0.1, step=0.1)

# Show a loading animation before prediction
def show_loading_animation():
    with st.spinner('Making predictions... ⏳'):
        time.sleep(2)  # Simulate a delay for prediction
        st.success('Prediction Done! ✅')

if st.sidebar.button("Predict"):
    show_loading_animation()
    
    result = predict_production(district, crop, season, area)
    
    if isinstance(result, str) and "Error" in result:
        st.markdown(f'<div class="error">{result}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result">Predicted Production (Tonnes) ⚖️: {result:.2f}</div>', unsafe_allow_html=True)

    # Optional: Add reset button to restart the process
    if st.button("Reset Inputs"):
        st.experimental_rerun()
