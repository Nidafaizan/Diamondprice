import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Set page config
st.set_page_config(page_title="Diamond Price Classifier", layout="wide")

# Load the trained model and scaler


@st.cache_resource
def load_model():
    with open('best_classification_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


model, scaler = load_model()

# Title and description
st.title("💎 Diamond Price Classifier")
st.markdown("""
This app predicts diamond price categories (Low, Medium, High) based on diamond characteristics.
The model achieves **97.21% accuracy** using a Random Forest classifier with MinMax scaling.
""")

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Diamond Characteristics")
    carat = st.slider("Carat Weight", 0.2, 5.0, 1.0, 0.1)
    cut_options = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
    cut = st.selectbox("Cut Quality", cut_options)

    color_options = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
    color = st.selectbox("Color", color_options)

    clarity_options = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    clarity = st.selectbox("Clarity", clarity_options)

with col2:
    st.subheader("Dimensions")
    depth = st.slider("Depth (mm)", 43.0, 80.0, 62.0, 0.1)
    table = st.slider("Table Width (%)", 43.0, 95.0, 57.0, 0.1)
    x = st.slider("Length (mm)", 0.0, 11.0, 4.0, 0.1)
    y = st.slider("Width (mm)", 0.0, 11.0, 4.0, 0.1)
    z = st.slider("Height (mm)", 0.0, 6.5, 2.5, 0.1)

# Encode categorical variables
cut_encoder = LabelEncoder()
cut_encoder.fit(['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
cut_encoded = cut_encoder.transform([cut])[0]

color_encoder = LabelEncoder()
color_encoder.fit(['J', 'I', 'H', 'G', 'F', 'E', 'D'])
color_encoded = color_encoder.transform([color])[0]

clarity_encoder = LabelEncoder()
clarity_encoder.fit(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
clarity_encoded = clarity_encoder.transform([clarity])[0]

# Prepare features for prediction
features = np.array(
    [[carat, cut_encoded, color_encoded, clarity_encoded, depth, table, x, y, z]])
features_scaled = scaler.transform(features)

# Make prediction
if st.button("🔮 Predict Price Category", use_container_width=True):
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    # Map prediction to category
    price_categories = {0: 'Low', 1: 'Medium', 2: 'High'}
    predicted_category = price_categories[prediction]

    # Display prediction
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        if predicted_category == 'Low':
            st.success(f"### 💰 Price Category: **{predicted_category}**")
        elif predicted_category == 'Medium':
            st.info(f"### 💰 Price Category: **{predicted_category}**")
        else:
            st.warning(f"### 💰 Price Category: **{predicted_category}**")

    with col2:
        st.metric("Confidence", f"{max(probabilities)*100:.2f}%")

    # Show probability breakdown
    st.subheader("Prediction Probabilities")
    prob_df = pd.DataFrame({
        'Category': ['Low', 'Medium', 'High'],
        'Probability': [f"{p*100:.2f}%" for p in probabilities]
    })
    st.bar_chart(pd.DataFrame({
        'Category': ['Low', 'Medium', 'High'],
        'Probability': probabilities
    }).set_index('Category'))

# Model information
st.markdown("---")
st.markdown("""
### 📊 Model Information
- **Algorithm**: Random Forest Classifier
- **Accuracy**: 97.21%
- **Features**: 9 (carat, cut, color, clarity, depth, table, x, y, z)
- **Scaling**: MinMax Scaler (0-1 normalization)
- **Top Features**: Diamond dimensions (y, x, z), clarity, and carat weight
""")
