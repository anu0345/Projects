import streamlit as st
import pickle
import numpy as np

# Load the Pickle model
model = pickle.load(open('label(2).pkl', 'rb'))

# Create Streamlit app
st.title("Fruit Variety Prediction")

st.header("Enter Soil Nutrient Parameters:")
potassium = st.number_input("Potassium (K) value (60-250):", min_value=60, max_value=250, step=1)
electrical_conductivity = st.number_input("Electrical Conductivity (EC) value (0.2-6):", min_value=0.2, max_value=6.0, step=0.1)
zinc = st.number_input("Zinc (Zn) value (20-50):", min_value=20.0, max_value=50.0, step=0.1)
boron = st.number_input("Boron (B) value (10-50):", min_value=10.0, max_value=50.0, step=0.1)

# Prediction
if st.button("Predict"):
    input_features = np.array([[potassium, electrical_conductivity, zinc, boron]])
    prediction = model.predict(input_features)[0]
    fruit_classes = ['Grapes', 'Mango', 'Mulberry', 'Pomegranate', 'Potato', 'Ragi']
    predicted_fruit = fruit_classes[prediction]
    st.success(f"The predicted fruit variety is: {predicted_fruit}")

