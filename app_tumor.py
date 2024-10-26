import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import MinMaxScaler

# Configuring the page with Title, icon, and layout
st.set_page_config(
    page_title="Tumor Prediction",
    page_icon="/home/hdoop//U5MRIcon.png",
    layout="wide",
    menu_items={
        'Get Help': 'https://helppage.ethipiau5m.com',
        'Report a Bug': 'https://bugreport.ethipiau5m.com',
        'About': 'https://ethiopiau5m.com',
    },
)

# Custom CSS to adjust spacing and background color
st.markdown("""
    <style>
        div.stApp { margin-top: -90px !important; }
        body { background-color: #f5f5f5; }
        h1 { color: #800080; font-family: 'Helvetica', sans-serif; }
    </style>
""", unsafe_allow_html=True)

st.image("cancer.jpg", width=800)  # Change "cancer.jpg" to the path of your image
st.markdown('<h1 style="text-align: center;">Predicting Cancer Tumor Type</h1>', unsafe_allow_html=True)

def horizontal_line(height=1, color="blue", margin="0.5em 0"):
    return f'<hr style="height: {height}px; margin: {margin}; background-color: {color};">'

# Load the model and preprocessing objects
loaded_model = joblib.load('catboost_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
minmax_scalers = joblib.load('scaler_params.pkl')

# Reconstruct the MinMaxScaler
scaler = MinMaxScaler()
scaler.scale_ = minmax_scalers['scale']
scaler.min_ = minmax_scalers['min']

# Feature names and types
features = {
    'Age': 'numerical',
    'Sex': 'categorical',
    'Occupation': 'categorical',
    'Education_Level': 'categorical',
    'Residence': 'categorical',
    'Region': 'categorical',
    'Zone': 'categorical',
    'City': 'categorical',
    'SubCity': 'categorical',
    'Woreda': 'categorical',
    'Kebel': 'categorical',
    'Diagnosis': 'categorical',
    'Group diagonsis': 'categorical',
    'Type diagnosis': 'categorical',
    'Status': 'categorical',
    'Unit': 'categorical',
    'Pacient Weight': 'categorical',
    'BMI': 'categorical',
    'Laboratory Service ': 'categorical',
    'HistoryType': 'categorical',
    'History value ': 'categorical',
    'Prescrption Type': 'categorical',
    'Prescribed Item': 'categorical',
    'Stage': 'categorical',
    'Price': 'numerical',
    'Is Paid': 'numerical',
    'Is Available': 'numerical',
}

# Sidebar for file upload
st.sidebar.title("Input Parameters")
uploaded_file = st.sidebar.file_uploader("Upload XLSX file", type=["XLSX"])

# Dictionary for grouping labels
group_labels = {
    'Demographic Data': ['Age', 'Sex', 'Occupation', 'Education_Level', 'Residence', 'Region', 'Zone', 'City', 'SubCity', 'Woreda', 'Kebel'],
    'Clinical Data': ['Diagnosis', 'Group diagonsis', 'Type diagnosis', 'Status', 'Unit', 'Pacient Weight', 'BMI', 'Laboratory Service ', 'HistoryType', 'History value ', 'Prescrption Type', 'Prescribed Item',  'Stage'],
    'Financial Data': ['Price', 'Is Paid', 'Is Available'],
}

# Initialize input dataframe
input_df = pd.DataFrame(index=[0])

# Define columns for Demographic, Clinical, and Financial data
demographic_col, clinical_col, financial_col = st.columns(3)

# Demographic Data Inputs
with demographic_col:
    st.subheader("Demographic Data")
    for feature in group_labels['Demographic Data']:
        widget_key = f"Demographic_{feature}"
        if features[feature] == 'categorical':
            input_df[feature] = st.selectbox(feature.replace('_', ' '), label_encoders[feature].classes_, key=widget_key)
        else:
            input_val = st.text_input(feature.replace('_', ' '), key=widget_key)
            input_df[feature] = pd.to_numeric(input_val, errors='coerce')

# Clinical Data Inputs
with clinical_col:
    st.subheader("Clinical Data")
    for feature in group_labels['Clinical Data']:
        widget_key = f"Clinical_{feature}"
        if features[feature] == 'categorical':
            input_df[feature] = st.selectbox(feature.replace('_', ' '), label_encoders[feature].classes_, key=widget_key)
        else:
            input_val = st.text_input(feature.replace('_', ' '), key=widget_key)
            input_df[feature] = pd.to_numeric(input_val, errors='coerce')

# Financial Data Inputs
with financial_col:
    st.subheader("Financial Data")
    for feature in group_labels['Financial Data']:
        widget_key = f"Financial_{feature}"
        if features[feature] == 'categorical':
            input_df[feature] = st.selectbox(feature.replace('_', ' '), label_encoders[feature].classes_, key=widget_key)
        else:
            input_val = st.text_input(feature.replace('_', ' '), key=widget_key)
            input_df[feature] = pd.to_numeric(input_val, errors='coerce')

# Display input data before encoding and normalization
st.write("Input Data (Before Encoding and Normalization):")
st.write(input_df)

# Predict button
if st.sidebar.button("Predict"):
    # Apply label encoding to categorical features
    for feature, encoder in label_encoders.items():
        if feature != 'Tumor_Type':
            input_df[feature] = encoder.transform(input_df[feature])

    # Apply Min-Max scaling to numerical features
    for feature, scaler in minmax_scalers.items():
        if feature in input_df.columns:
            input_df[feature] = scaler.transform(input_df[feature].values.reshape(-1, 1))

    # Display input data after encoding and normalization
    st.write("Input Data (After Encoding and Normalization):")
    st.write(input_df)

    # Make predictions
    prediction = loaded_model.predict(input_df)

    # Assuming it's a classification problem, and the prediction array is valid
    if isinstance(prediction, np.ndarray) and prediction.ndim > 0:
        predicted_label = prediction[0]
        Tumor_Type = np.array(['Benign', 'Malignant'])
        prediction_index = int(predicted_label) if isinstance(predicted_label, (int, np.integer)) else np.where(Tumor_Type == predicted_label)[0][0]

        # Output prediction
        st.sidebar.write("Prediction:", Tumor_Type[prediction_index])

        # Show prediction probabilities if available
        if hasattr(loaded_model, 'predict_proba'):
            prediction_proba = loaded_model.predict_proba(input_df)
            st.subheader('Prediction (Is cancer tumor Benign or Malignant?)')
            st.write(f"Type of cancer Tumor: {Tumor_Type[prediction_index]}")
            st.subheader('Prediction Probability')
            probability_df = pd.DataFrame(prediction_proba, columns=Tumor_Type)
            st.write(probability_df)
    else:
        st.sidebar.write("Prediction could not be made.")
