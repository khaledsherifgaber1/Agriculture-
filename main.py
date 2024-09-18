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



# Title and description
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




# Check for non-numeric values
if not pd.api.types.is_numeric_dtype(user_data):
    st.error("Input data contains non-numeric values. Please provide valid numerical inputs.")
    st.stop()

# Ensure columns match those expected by the scaler
expected_columns = Scaller.get_feature_names_out()
if not set(expected_columns).issubset(user_data.columns):
    st.error("Mismatch between input data columns and expected columns for scaling.")
    st.stop()

# Print debug information
st.write("User data before scaling:")
st.write(user_data)

# Apply Scaler
try:
    Scaller_Data = Scaller.transform(user_data)
except Exception as e:
    st.error(f"An error occurred while scaling the data: {e}")
    st.stop()
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
user_data = pd.DataFrame(data=Scaller_Data, columns=Scaller.get_feature_names_out(), index=user_data.index)

# Ensure columns match those expected by the scaler
Final = user_data[['Log_Humidity', 'Log_Rainfall', 'Log_Rainfall_Humidity_Index',
                   'PT_Potassium', 'Log_Phosphorus', 'Log_NPK_Average',
                   'Temp_Humididty_Index', 'SQ_Nitrogen', 'Log_PK_Ratio', 'Temperature']]

# Load the Model
model = joblib.load("Extra_Tree_model (1).pkl")

# Prediction Button
if st.button('Predict Crop'):
    try:
        # Predict
        prediction = model.predict(Final)
        predicted_crop = encoder_2.inverse_transform(prediction)

        # Crop Icons
        crop_icons = {
    'Rice': 'https://www.pngkey.com/png/full/137-1378768_rice-png.png',
    'Maize': 'https://www.pngkey.com/png/full/84-849452_corn-maize-grain-plant-clipart.png',
    'Jute': 'https://www.pngfind.com/pngs/m/11-113628_jute-png-image-jute-plant-png.png',
    'Cotton': 'https://www.pngkey.com/png/full/274-2746382_cotton-cotton-plant-png.png',
    'Coconut': 'https://www.pngkit.com/png/full/27-279157_coconut-icon-png-coconut-icon-transparent.png',
    'Papaya': 'https://www.pngkey.com/png/full/49-496342_papaya-fruit-papaya-fruit-image-png.png',
    'Orange': 'https://www.pngkey.com/png/full/274-2742951_orange-fruit-png.png',
    'Apple': 'https://www.pngkit.com/png/full/1-15184_apple-fruit-png-transparent-background-apple-fruit.png',
    'Muskmelon': 'https://www.pngfind.com/pngs/m/72-720453_muskmelon-png-image-musk-melon-png.png',
    'Watermelon': 'https://www.pngkey.com/png/full/38-380874_watermelon-fruit-png-watermelon-png-image.png',
    'Grapes': 'https://www.pngkit.com/png/full/201-2017066_grapes-png-transparent-background-grape.png',
    'Mango': 'https://www.pngkit.com/png/full/63-635708_mango-mango-fruit-png-transparent-image.png',
    'Banana': 'https://www.pngkit.com/png/full/55-558080_banana-png-image-banana-png-transparent.png',
    'Pomegranate': 'https://www.pngkit.com/png/full/54-547457_pomegranate-fruit-png.png',
    'Lentil': 'https://www.pngkit.com/png/full/203-2035965_lentil-png-lentils.png',
    'Blackgram': 'https://www.pngkit.com/png/full/0-7555_blackgram-png.png',
    'MungBean': 'https://www.pngkit.com/png/full/0-7664_mung-bean-mung-bean-png.png',
    'MothBeans': 'https://www.pngkit.com/png/full/14-146668_beans-png-transparent-beans-png.png',
    'PigeonPeas': 'https://www.pngkit.com/png/full/23-232837_pigeon-peas-png-image-pigeon-peas.png',
    'KidneyBeans': 'https://www.pngkit.com/png/full/203-2034582_red-kidney-beans-png-image.png',
    'ChickPea': 'https://www.pngkit.com/png/full/74-741289_chickpea-png-image-chickpea.png',
    'Coffee': 'https://www.pngkit.com/png/full/161-161731_coffee-beans-png-coffee-beans.png'
}

        crop_name = predicted_crop[0]
        icon_url = crop_icons.get(crop_name, 'https://example.com/icons/default.png')

        st.image(icon_url, width=150, caption=f"Recommended Crop: {crop_name}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer
st.markdown('<div class="footer"><p>&copy; 2024 Crop Recommendation System. All rights reserved.</p></div>',
            unsafe_allow_html=True)
