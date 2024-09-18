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
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="wide")

# Add a custom background image
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://github.com/khaledsherifgaber1/Agriculture-/blob/cc52ea76e869e8731820783d7a2862eba842e39f/Gemini_Generated_Image_w5z0g0w5z0g0w5z0.jpeg?raw=true');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
    .header {
        text-align: center;
        background-color: #4CAF50;
        padding: 10px;
        color: white;
        font-size: 2em;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 10px;
        color: #4CAF50;
        font-size: 1em;
        border-top: 1px solid #ddd;
    }
    </style>
    <div class="header">Crop Recommendation System</div>
    """, unsafe_allow_html=True)

# Add footer at the end of your code
st.markdown("""
    <div class="footer">
        <p>&copy; 2024 Your Name | <a href="https://github.com/yourusername" target="_blank" style="color: #4CAF50;">GitHub</a></p>
    </div>
    """, unsafe_allow_html=True)


# Title and description
st.markdown('<h1 class="title">🌿 Crop Recommendation System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="description">Enter your soil and climate conditions to get recommendations for the best crops to grow.</p>',
    unsafe_allow_html=True)

# Input Fields in a styled container
with st.container():
    st.markdown('<div class="input-section">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        nitrogen = st.number_input('Nitrogen (kg/ha)', min_value=0.0, value=0.0, step=0.01)
        phosphorus = st.number_input('Phosphorus (kg/ha)', min_value=0.0, value=0.0, step=0.01)
        potassium = st.number_input('Potassium (kg/ha)', min_value=0.0, value=0.0, step=0.01)
        temperature = st.number_input('Temperature (°C)', min_value=-50.0, value=0.0, step=0.01)

    with col2:
        humidity = st.number_input('Humidity (%)', min_value=0.0, value=0.0, step=0.01)
        ph_value = st.number_input('pH Value', min_value=0.0, value=0.0, step=0.01)
        rainfall = st.number_input('Rainfall (mm)', min_value=0.0, value=0.0, step=0.01)

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
encoder_1 = joblib.load("../../OneDrive/Desktop/Crop Recommendation System/Ordinal_Encoder.pkl")
encoder_2 = joblib.load("../../OneDrive/Desktop/Crop Recommendation System/label_Encoder (1).pkl")

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
Scaller = joblib.load("../../OneDrive/Desktop/Crop Recommendation System/FeatureScaler.pkl")
Scaller_Data = Scaller.transform(user_data)
user_data = pd.DataFrame(data=Scaller_Data, columns=Scaller.get_feature_names_out(), index=user_data.index)

# Ensure columns match those expected by the scaler
Final = user_data[['Log_Humidity', 'Log_Rainfall', 'Log_Rainfall_Humidity_Index',
                   'PT_Potassium', 'Log_Phosphorus', 'Log_NPK_Average',
                   'Temp_Humididty_Index', 'SQ_Nitrogen', 'Log_PK_Ratio', 'Temperature']]

# Load the Model
model = joblib.load("../../OneDrive/Desktop/Crop Recommendation System/Extra_Tree_model (1).pkl")

# Prediction Button
if st.button('Predict Crop'):
    try:
        # Predict
        prediction = model.predict(Final)
        predicted_crop = encoder_2.inverse_transform(prediction)

        # Crop Icons
        crop_icons = {
            'Rice': 'https://example.com/icons/rice.png',
            'Maize': 'https://example.com/icons/maize.png',
            'Jute': 'https://example.com/icons/jute.png',
            'Cotton': 'https://example.com/icons/cotton.png',
            'Coconut': 'https://example.com/icons/coconut.png',
            'Papaya': 'https://example.com/icons/papaya.png',
            'Orange': 'https://example.com/icons/orange.png',
            'Apple': 'https://example.com/icons/apple.png',
            'Muskmelon': 'https://example.com/icons/muskmelon.png',
            'Watermelon': 'https://example.com/icons/watermelon.png',
            'Grapes': 'https://example.com/icons/grapes.png',
            'Mango': 'https://example.com/icons/mango.png',
            'Banana': 'https://example.com/icons/banana.png',
            'Pomegranate': 'https://example.com/icons/pomegranate.png',
            'Lentil': 'https://example.com/icons/lentil.png',
            'Blackgram': 'https://example.com/icons/blackgram.png',
            'MungBean': 'https://example.com/icons/mungbean.png',
            'MothBeans': 'https://example.com/icons/mothbeans.png',
            'PigeonPeas': 'https://example.com/icons/pigeonpeas.png',
            'KidneyBeans': 'https://example.com/icons/kidneybeans.png',
            'ChickPea': 'https://example.com/icons/chickpea.png',
            'Coffee': 'https://example.com/icons/coffee.png'
        }

        crop_name = predicted_crop[0]
        icon_url = crop_icons.get(crop_name, 'https://example.com/icons/default.png')

        st.image(icon_url, width=150, caption=f"Recommended Crop: {crop_name}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown('<div class="footer"><p>&copy; 2024 Crop Recommendation System. All rights reserved.</p></div>',
            unsafe_allow_html=True)