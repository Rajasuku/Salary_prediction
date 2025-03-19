import streamlit as st
import numpy as np
import joblib

# Load the trained model
def load_model():
    try:
        model = joblib.load("salary_model.pkl")
        return model
    except FileNotFoundError:
        return None

# Streamlit UI
st.title("Salary Prediction App")

# Load model
model = load_model()
if model:
    st.write("### Predict Salary")
    years_experience = st.number_input("Enter Years of Experience:", min_value=0.0, step=0.1)
    if st.button("Predict"):
        try:
            prediction = model.predict(np.array([[years_experience, 0]]))  # Add an extra placeholder if needed
            st.success(f"Predicted Salary: ${prediction[0]:,.2f}")
        except ValueError as e:
            st.error(f"Model expects a different input shape: {e}")
