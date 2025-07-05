import streamlit as st
import numpy as np
import joblib

FEATURE_NAMES = [ 'V2','V3','V4','V11','V12','V14','V16','V17','V18','V7','V10' ]

@st.cache(allow_output_mutation=True)
def load_artifacts():
    model   = joblib.load("/Users/rohith/Desktop/fraud-detection-ml/Models/rf_champion.joblib")
    return model  # , scaler

model = load_artifacts()

st.title("🔍 Credit-Card Fraud Detector")
st.markdown("Enter transaction features on the left and hit **Run Fraud Check**.")

# Sidebar inputs
st.sidebar.header("Transaction Features")
inputs = {}
for feat in FEATURE_NAMES:
    inputs[feat] = st.sidebar.number_input(feat, value=0.0, format="%.4f")

if st.sidebar.button("Run Fraud Check"):
    X = np.array([list(inputs.values())])
    proba = model.predict_proba(X)[:,1][0]
    pred  = model.predict(X)[0]

    st.subheader("Result")
    st.metric("Fraud Probability", f"{proba:.1%}")
    st.write("🚩 **Fraudulent**" if pred else "✅ **Legitimate**")