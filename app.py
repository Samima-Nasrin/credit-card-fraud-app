import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(page_title="Credit Card Fraud Detector",
                   layout="wide",
                   page_icon="💳")

# Load the trained model
with open("trained_model.sav", "rb") as f:
    model = pickle.load(f)

st.title("Credit Card Fraud Detection App")

st.write("Enter all feature values exactly like in the training data:")

# You have 30 features in the dataset (V1 to V28, plus Time and Amount)
features = [f"V{i}" for i in range(1, 29)]
features = ["Time"] + features + ["Amount"]

# Create input fields
input_data = []
for feature in features:
    val = st.number_input(f"{feature}", value=0.0, format="%.5f")
    input_data.append(val)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)[0]
    if prediction == 0:
        st.success("Transaction is Legit")
    else:
        st.error("Transaction is Fraudulent")
