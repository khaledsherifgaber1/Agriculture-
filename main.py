import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PowerTransformer

# Functions for data processing
def SQ(Data, SQ_Columns):
    for i in SQ_Columns:
        Data[f'SQ_{i}'] = np.sqrt(Data[i].clip(lower=0))
        Data.drop(i, axis=1, inplace=True)
    return Data

def Log(Data, log_Columns):
    for i in log_Columns:
        Data[f'Log_{i}'] = np.log1p(Data[i])
        Data.drop(i, axis=1, inplace=True)
    return Data

def PT(Data, PT_Columns):
    transformers = {}
    for col in PT_Columns:
        if Data[col].nunique() > 1:
            pt = PowerTransformer(method='yeo-johnson')
            transformers[col] = pt.fit(Data[[col]])
            Data[f'PT_{col}'] = transformers[col].transform(Data[[col]])
        else:
            Data[f'PT_{col}'] = Data[[col]]  # Pass through if not transformable
        Data.drop(col, axis=1, inplace=True)
    return Data

def Feature_Engineering(Train_Set):
    def PH_Classification(PH):
        if PH < 5.5:
            return "Acidic"
        elif 5.5 <= PH <= 7.5:
            return "Neutral"
        else:
            return "Alkaline"

    Train_Set['NP_Ratio'] = Train_Set['Nitrogen'] / Train_Set['Phosphorus']
    Train_Set['NK_Ratio'] = Train_Set['Nitrogen'] / Train_Set['Potassium']
    Train_Set['PK_Ratio'] = Train_Set['Phosphorus'] / Train_Set['Potassium']
    Train_Set["NPK_Average"] = (Train_Set['Nitrogen'] + Train_Set["Phosphorus"] + Train_Set["Potassium"]) / 3
    Train_Set["Temp_Humididty_Index"] = Train_Set["Temperature"] * Train_Set['Humidity']
    Train_Set["Rainfall_Humidity_Index"] = Train_Set['Rainfall'] * Train_Set['Humidity']
    Train_Set["PH_Categories"] = Train_Set['pH_Value'].apply(PH_Classification)
    return Train_Set

# Streamlit App
st.set_page_config(page_title="Ghosn App", page_icon="🌾", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://github.com/khaledsherifgaber1/Agriculture-/blob/main/crops-growing-in-thailand.jpg?raw=true');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        position: relative;
    }

    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5); /* Dark overlay for better readability */
        z-index: 1;
    }

    .stApp > div {
        position: relative;
        z-index: 2;
    }

    .header {
        text-align: center;
        background-color: #4CAF50;
        padding: 20px;
        color: white;
        font-size: 2.5em;
        font-weight: bold;
        border-bottom: 2px solid #388E3C;
        margin-bottom: 40px;
        border-radius: 10px;
    }

    .footer {
        text-align: center;
        padding: 10px;
        color: #4CAF50;
        font-size: 1em;
        border-top: 1px solid #ddd;
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: white;
        border-radius: 10px 10px 0 0; /* Rounded top corners */
    }

    .title {
        font-size: 3em;
        font-weight: bold;
        color: #fff;
        text-align: center;
        margin-top: 20px;
    }

    .description {
        font-size: 1.2em;
        color: #ddd;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 40px;
    }

    .input-section {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 30px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        max-width: 700px;
        margin: 0 auto;
    }

    .input-field {
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<div class="header">Crop Recommendation System</div>', unsafe_allow_html=True)

# Title and description
st.markdown(
    '<p class="description">Enter your soil and climate conditions to get recommendations for the best crops to grow.</p>',
    unsafe_allow_html=True)

# Input Fields in a styled container
with st.container():
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        nitrogen = st.number_input('Nitrogen (kg/ha)', min_value=0.0, value=0.0, step=0.01, key='nitrogen', help="Enter the amount of Nitrogen in kg per hectare.")
        phosphorus = st.number_input('Phosphorus (kg/ha)', min_value=0.0, value=0.0, step=0.01, key='phosphorus', help="Enter the amount of Phosphorus in kg per hectare.")
        potassium = st.number_input('Potassium (kg/ha)', min_value=0.0, value=0.0, step=0.01, key='potassium', help="Enter the amount of Potassium in kg per hectare.")
        temperature = st.number_input('Temperature (°C)', min_value=-50.0, value=0.0, step=0.01, key='temperature', help="Enter the temperature in Celsius.")

    with col2:
        humidity = st.number_input('Humidity (%)', min_value=0.0, value=0.0, step=0.01, key='humidity', help="Enter the humidity percentage.")
        ph_value = st.number_input('pH Value', min_value=0.0, value=0.0, step=0.01, key='ph_value', help="Enter the pH value of the soil.")
        rainfall = st.number_input('Rainfall (mm)', min_value=0.0, value=0.0, step=0.01, key='rainfall', help="Enter the amount of rainfall in mm.")

    st.markdown('</div>', unsafe_allow_html=True)

# Create DataFrame from user input
user_data = pd.DataFrame({
    'Nitrogen': [nitrogen],
    'Phosphorus': [phosphorus],
    'Potassium': [potassium],
    'Temperature': [temperature],
    'Humidity': [humidity],
    'pH_Value': [ph_value],
    'Rainfall': [rainfall]
})

# Apply Feature Engineering
user_data = Feature_Engineering(user_data)

# Load Encoders
encoder_1 = joblib.load("Ordinal_Encoder.pkl")
encoder_2 = joblib.load("label_Encoder (1).pkl")

# Apply Encoding
user_data['PH_Cat'] = encoder_1.transform(user_data[['PH_Categories']])
user_data.drop(['PH_Categories'], axis=1, inplace=True)

# Define Columns for Transformation
log_Columns = np.array(
    ['Phosphorus', 'Humidity', 'Rainfall', 'NK_Ratio', 'PK_Ratio', 'NPK_Average', 'Rainfall_Humidity_Index'],
    dtype=object)
PT_Columns = np.array(['Potassium', 'NP_Ratio'], dtype=object)
SQ_Columns = np.array(['Nitrogen'], dtype=object)

# Apply Transformations
Log(user_data, log_Columns)
SQ(user_data, SQ_Columns)
user_data = PT(user_data, PT_Columns)

# Load Scaler and Scale Data
Scaller = joblib.load("FeatureScaler.pkl")
Scaller_Data = Scaller.transform(user_data)
user_data = pd.DataFrame(data=Scaller_Data, columns=user_data.columns)

# Load the Model
model = joblib.load('Extra_Tree_model.pkl')

# Predict the Crop
prediction = model.predict(user_data)[0]

# Load Crop Images
crop_images = {
    'Rice': 'https://example.com/images/rice.jpg',
    'Maize': 'https://example.com/images/maize.jpg',
    'Jute': 'https://example.com/images/jute.jpg',
    'Cotton': 'https://example.com/images/cotton.jpg',
    'Coconut': 'https://example.com/images/coconut.jpg',
    'Papaya': 'https://example.com/images/papaya.jpg',
    'Orange': 'https://example.com/images/orange.jpg',
    'Apple': 'https://example.com/images/apple.jpg',
    'Muskmelon': 'https://example.com/images/muskmelon.jpg',
    'Watermelon': 'https://example.com/images/watermelon.jpg',
    'Grapes': 'https://example.com/images/grapes.jpg',
    'Mango': 'https://example.com/images/mango.jpg',
    'Banana': 'https://example.com/images/banana.jpg',
    'Pomegranate': 'https://example.com/images/pomegranate.jpg',
    'Lentil': 'https://example.com/images/lentil.jpg',
    'Blackgram': 'https://example.com/images/blackgram.jpg',
    'MungBean': 'https://example.com/images/mungbean.jpg',
    'MothBeans': 'https://example.com/images/mothbeans.jpg',
    'PigeonPeas': 'https://example.com/images/pigeonpeas.jpg',
    'KidneyBeans': 'https://example.com/images/kidneybeans.jpg',
    'ChickPea': 'https://example.com/images/chickpea.jpg',
    'Coffee': 'https://example.com/images/coffee.jpg'
}

# Display Results
st.markdown('<h2 style="text-align: center;">Recommended Crop</h2>', unsafe_allow_html=True)
st.markdown(f'<h3 style="text-align: center; color: #4CAF50;">{prediction}</h3>', unsafe_allow_html=True)
st.image(crop_images.get(prediction, 'https://example.com/images/default.jpg'), use_column_width=True)

# Footer
st.markdown('<div class="footer">Developed with ❤️ by [Khaled Sherif]</div>', unsafe_allow_html=True)
