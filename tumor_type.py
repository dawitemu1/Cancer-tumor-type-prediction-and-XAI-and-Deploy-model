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
    #initial_sidebar_state="collapsed",  # Optional, collapses the sidebar by default
    menu_items={
        'Get Help': 'https://helppage.ethipiau5m.com',
        'Report a Bug': 'https://bugreport.ethipiau5m.com',
        'About': 'https://ethiopiau5m.com',
    },
)

# Custom CSS to adjust spacing
custom_css = """
<style>
    div.stApp {
        margin-top: -90px !important;  /* We can adjust this value as needed */
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)
st.image("cancer.jpg", width=800)  # Change "logo.png" to the path of your logo image file
# # Setting the title with Markdown and center-aligning
st.markdown('<h1 style="text-align: center;">Predicting Cancer Tumor Type  </h1>', unsafe_allow_html=True)

# Defining background color
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Defining  header color and font
st.markdown(
    """
    <style>
    h1 {
        color: #800080;  /* Blue color */
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def horizontal_line(height=1, color="blue", margin="0.5em 0"):
    return f'<hr style="height: {height}px; margin: {margin}; background-color: {color};">'

# # Load the XGBoost model
# model_path = "Tumor_type_CatBoost_model.sav"
# loaded_model = pickle.load(open(model_path, "rb"))

# # Load the label encoders
# label_encoders_path = "label_encoders.pkl"
# label_encoders = pickle.load(open(label_encoders_path, "rb"))

# # Load the MinMax scalers
# scalers_path = "minmax_scalers.pkl"
# minmax_scalers = pickle.load(open(scalers_path, "rb"))

# Load the CatBoost model
loaded_model = joblib.load('catboost_model2.pkl')

# Load the label encoders
label_encoders = joblib.load('label_encoders2.pkl')

# Load the MinMax scaler parameters
minmax_scalers = joblib.load('scaler_params2.pkl')

# Reconstruct the MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Create a new MinMaxScaler instance
scaler = MinMaxScaler()
scaler.scale_ = minmax_scalers['scale']  # Assign the scale
scaler.min_ = minmax_scalers['min']      # Assign the min

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
    'Imagereport': 'categorical',
    'Stage': 'categorical',
    'Price': 'numerical',
    'Is Paid': 'numerical',
    'Is Available': 'numerical',
}

# Sidebar title
st.sidebar.title("Input Parameters")
st.sidebar.markdown("""
[Example XLSX input file](https://master/penguins_example.csv)
""")

# Create dictionary for grouping labels
group_labels = {
    'Demographic Data': ['Age', 'Sex', 'Occupation', 'Education_Level', 'Residence', 'Region', 'Zone', 'City', 'SubCity', 'Woreda', 'Kebel'],
    'Clinical Data': ['Diagnosis', 'Group diagonsis', 'Type diagnosis',
       'Status', 'Unit', 'Pacient Weight', 'BMI', 'Laboratory Service ', 'HistoryType', 'History value ',
       'Prescrption Type', 'Prescribed Item',  'Stage'],
    'Imagereport Data': ['Imagereport'],
    'financial Data': ['Price', 'Is Paid', 'Is Available'],
}

# Option for CSV file upload
uploaded_file = st.sidebar.file_uploader("Upload XLSX file", type=["XLSX"])

# If CSV file is uploaded, read the file
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

# If CSV file is not uploaded, allow manual input
else:
    # Create empty dataframe to store input values
    input_df = pd.DataFrame(index=[0])

    # Loop through features and get user input
    for group, features_in_group in group_labels.items():
        st.sidebar.markdown(horizontal_line(), unsafe_allow_html=True)
        st.sidebar.subheader(group)
        for feature in features_in_group:
            # Ensure each widget has a unique key
            widget_key = f"{group}_{feature}"

            # Display more descriptive labels
            if features[feature] == 'categorical':
                label = f"{feature.replace('_', ' ')}"
                input_df[feature] = st.sidebar.selectbox(label, label_encoders[feature].classes_, key=widget_key)
            else:
                label = f"{feature.replace('_', ' ')}"
                input_val = st.sidebar.text_input(label, key=widget_key)
                input_df[feature] = pd.to_numeric(input_val, errors='coerce')

st.markdown(
    """
    ### Welcome to Cancer tumor type prediction Tool!

    #### What You Can Do:
    1. Know the cancer pacients' Tumor and make next clinical decision .
    2. Pacients know the type of tumor for given cancer, before reciving any medical treatment
    3. It Give clear information about type of tumor for pythicans 
    4. Based on the type of tumor pythicans make accoedingly the decision which is data driven decision 

    Dive into the rich data of Tikur Anebassa hospital from 2020 to 2024, interact, and uncover valuable insights for decision making!
    """
)

# Display the input dataframe

st.write("Input Data (Before Encoding and Normalization):")
st.write(input_df)

# Make predictions using the loaded model
if st.sidebar.button("Predict"):
    # Apply label encoding to categorical features
    for feature, encoder in label_encoders.items():
        if feature != 'Tumor_Type':
            input_df[feature] = encoder.transform(input_df[feature])

    # Apply Min-Max scaling to numerical features
    for feature, scaler in minmax_scalers.items():
        if feature in input_df.columns:  # Check if feature exists in input_df
            input_df[feature] = scaler.transform(input_df[feature].values.reshape(-1, 1))

    # Display the input data after encoding and normalization
    st.write("Input Data (After Encoding and Normalization):")
    st.write(input_df)

    # Make predictions
    prediction = loaded_model.predict(input_df)

    # Ensure prediction is a valid array with expected shape
    if isinstance(prediction, np.ndarray) and prediction.ndim > 0:
        # Assuming it's a classification problem, take the first prediction
        predicted_label = prediction[0]  # Get the first prediction
        
        # Check if it's a string label
        Tumor_Type = np.array(['Benign', 'Malignant'])
        if isinstance(predicted_label, str):
            # Find the index of the predicted label
            prediction_index = np.where(Tumor_Type == predicted_label)[0][0]
        else:
            # Otherwise assume it's an index
            prediction_index = int(predicted_label)

        # Output the prediction
        st.sidebar.write("Prediction:", Tumor_Type[prediction_index])

        # Show prediction probabilities if applicable
        if hasattr(loaded_model, 'predict_proba'):
            prediction_proba = loaded_model.predict_proba(input_df)
            st.subheader('Prediction (Is cancer tumor Benign or Malignant?)')
            st.write(f"Type of cancer Tumor: {Tumor_Type[prediction_index]}")

            st.subheader('Prediction Probability')
            probability_df = pd.DataFrame(prediction_proba, columns=Tumor_Type)
            st.write(probability_df)
    else:
        st.sidebar.write("Prediction could not be made.")
