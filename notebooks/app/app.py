import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import shap
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier

# ------------------------
# Load dataset reference
# ------------------------
data = load_breast_cancer()

df_reference = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

feature_names = df_reference.columns
# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Breast Cancer Diagnostic System",
    page_icon="🩺",
    layout="wide"
)

# -------------------------------------------------
# LOAD MODEL & SCALER
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "..", "models", "final_svm_model.pkl")
scaler_path = os.path.join(BASE_DIR, "..", "..", "models", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -------------------------------------------------
# LOAD DATA FOR FEATURE INFO
# -------------------------------------------------
data = load_breast_cancer()
feature_names = data.feature_names
df_reference = pd.DataFrame(data.data, columns=feature_names)

# -------------------------------------------------
# TRAIN RF FOR EXPLANATION (FAST + STABLE)
# -------------------------------------------------
rf_explainer_model = RandomForestClassifier(random_state=42)
rf_explainer_model.fit(df_reference, data.target)

explainer = shap.TreeExplainer(rf_explainer_model)

# -------------------------------------------------
# TITLE
# -------------------------------------------------
st.title("🩺 AI-Based Breast Cancer Diagnostic System")
st.markdown("Clinical Decision Support Tool")
st.divider()

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Manual Prediction", "📂 Batch Upload", "📊 Model Info"])

# =================================================
# TAB 1 – MANUAL
# =================================================
with tab1:

    st.subheader("Enter Diagnostic Measurements")

    input_values = []
    col1, col2, col3 = st.columns(3)
    columns = [col1, col2, col3]

    for i, feature in enumerate(feature_names):
        col = columns[i % 3]

        min_val = float(df_reference[feature].min())
        max_val = float(df_reference[feature].max())
        mean_val = float(df_reference[feature].mean())

        value = col.slider(
            feature,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

        input_values.append(value)

    input_data = np.array(input_values).reshape(1, -1)

    if st.button("Run AI Diagnosis"):

        # ---- SVM Prediction ----
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        malignant_prob = probability[0] * 100
        benign_prob = probability[1] * 100

        st.divider()
        st.subheader("Diagnosis Result")

        if prediction == 0:
            st.error("⚠️ Malignant Tumor Detected")
        else:
            st.success("✅ Benign Tumor Detected")

        # Risk Level
        if malignant_prob < 20:
            st.success("🟢 Risk Level: LOW")
        elif malignant_prob < 60:
            st.warning("🟡 Risk Level: MODERATE")
        else:
            st.error("🔴 Risk Level: HIGH")

        st.subheader("Prediction Confidence")
        st.progress(int(max(malignant_prob, benign_prob)))

        colA, colB = st.columns(2)
        colA.metric("Malignant Probability", f"{malignant_prob:.2f}%")
colB.metric("Benign Probability", f"{benign_prob:.2f}%")

# ---------------------------
# SHAP EXPLANATION STARTS HERE
# ---------------------------

import shap

st.divider()
st.subheader("🧠 AI Explanation (Top Contributing Features)")

background = shap.sample(df_reference, 50)

explainer = shap.KernelExplainer(
    model.predict_proba,
    scaler.transform(background)
)

shap_values = explainer.shap_values(input_scaled)

if isinstance(shap_values, list):
    shap_contrib = shap_values[0][0]
else:
    shap_contrib = shap_values[0]

shap_contrib = np.array(shap_contrib).flatten()

min_len = min(len(feature_names), len(shap_contrib))

shap_df = pd.DataFrame({
    "Feature": feature_names[:min_len],
    "Contribution": shap_contrib[:min_len]
})

shap_df = shap_df.reindex(
    shap_df["Contribution"].abs().sort_values(ascending=False).index
).head(10)

st.dataframe(shap_df)
st.bar_chart(shap_df.set_index("Feature"))
# =================================================
# TAB 2 – CSV
# =================================================
with tab2:

    st.subheader("Upload Patient CSV Data")

    template_df = pd.DataFrame(columns=feature_names)
    template_csv = template_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download CSV Template",
        template_csv,
        "template.csv",
        "text/csv"
    )

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:

        df_uploaded = pd.read_csv(uploaded_file)

        if list(df_uploaded.columns) != list(feature_names):
            st.error("CSV columns do not match required features.")
        else:
            scaled_data = scaler.transform(df_uploaded)
            predictions = model.predict(scaled_data)
            probabilities = model.predict_proba(scaled_data)

            df_uploaded["Prediction"] = np.where(
                predictions == 0, "Malignant", "Benign"
            )

            df_uploaded["Malignant Probability (%)"] = (
                probabilities[:, 0] * 100
            ).round(2)

            df_uploaded["Benign Probability (%)"] = (
                probabilities[:, 1] * 100
            ).round(2)

            st.dataframe(df_uploaded)

            csv = df_uploaded.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Results",
                csv,
                "predictions.csv",
                "text/csv"
            )

# =================================================
# TAB 3 – INFO
# =================================================
with tab3:

    st.subheader("Model Overview")

    st.markdown("""
    **Prediction Model:** Support Vector Machine (RBF Kernel)  
    **Explanation Model:** Random Forest (for SHAP analysis)  
    **Test Accuracy:** ~97%  
    **ROC-AUC:** ~0.995  

    This system uses machine learning to assist clinical diagnosis.
    """)

st.divider()
st.warning("⚠️ This system is for educational purposes only.")