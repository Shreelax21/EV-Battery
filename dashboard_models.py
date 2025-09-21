import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

st.set_page_config(page_title="EV Battery Models Dashboard", layout="wide")
st.title("🚗 EV Battery Models Dashboard")

# ==============================
# Load Data / Metrics
# ==============================
@st.cache_data
def load_fault_autoencoder():
    metrics = joblib.load("models/fault_autoencoder_metrics.joblib")
    try:
        errors = np.load("models/fault_autoencoder_errors.npy")
    except:
        errors = None
    return metrics, errors

@st.cache_data
def load_soh_regressor():
    metrics = joblib.load("models/soh_regressor_metrics.joblib")
    df_pred = pd.read_csv("models/soh_predictions.csv")
    model = joblib.load("models/soh_regressor.joblib")
    return metrics, df_pred, model

@st.cache_data
def load_thermal_autoencoder():
    stats = joblib.load("models/thermal_autoencoder_stats.joblib")
    try:
        errors = np.load("models/thermal_autoencoder_errors.npy")
    except:
        errors = None
    return stats, errors

fault_metrics, fault_errors = load_fault_autoencoder()
soh_metrics, soh_df, soh_model = load_soh_regressor()
thermal_stats, thermal_errors = load_thermal_autoencoder()

# ==============================
# Tabs for each model
# ==============================
tab1, tab2, tab3 ,tab4 = st.tabs(["Fault Autoencoder", "SoH Regressor", "Thermal Autoencoder","Summary Metrics"])

# ------------------------------
# Tab 1: Fault Autoencoder
# ------------------------------
with tab1:
    st.header("⚡ Fault Autoencoder")
    st.markdown("""
        **Goal:** Detect anomalies in battery and motor data.  
        **Method:** Autoencoder reconstruction error; anomalies occur when error exceeds threshold.
    """)
    
    st.subheader("Metrics")
    st.write(fault_metrics)
    
    if fault_errors is not None:
        st.subheader("Reconstruction Error Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.histplot(fault_errors, bins=50, kde=True, ax=ax)
        threshold = np.percentile(fault_errors, 95)
        ax.axvline(threshold, color='red', linestyle='--', label='95th percentile threshold')
        ax.set_xlabel("Reconstruction Error")
        ax.set_ylabel("Count")
        ax.set_title("Fault Autoencoder: Reconstruction Error Distribution")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Anomalies Over Time (Interactive)")
        start_idx, end_idx = st.slider(
            "Select sample range",
            min_value=0,
            max_value=len(fault_errors)-1,
            value=(0, min(200, len(fault_errors)-1)),
            step=1
        )
        fig2 = px.line(
            x=np.arange(start_idx, end_idx),
            y=fault_errors[start_idx:end_idx],
            labels={"x":"Sample Index", "y":"Reconstruction Error"},
            title="Fault Autoencoder: Reconstruction Error over Selected Range"
        )
        fig2.add_hline(y=threshold, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("⚠️ Reconstruction errors not found. Please save errors as 'fault_autoencoder_errors.npy'.")

# ------------------------------
# Tab 2: SoH Regressor
# ------------------------------
with tab2:
    st.header("🔋 SoH Regressor")
    st.markdown("""
        **Goal:** Predict battery State of Health (SoH).  
        **Method:** Random Forest regression; predict % battery health.
    """)
    
    st.subheader("Model Metrics")
    st.write(soh_metrics)
    
    st.subheader("Predicted vs True SoH")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(soh_df["True_SoH(%)"], soh_df["Predicted_SoH(%)"], c='blue')
    ax.plot([0, 100], [0, 100], 'r--')
    ax.set_xlabel("True SoH (%)")
    ax.set_ylabel("Predicted SoH (%)")
    ax.set_title("SoH Regressor: True vs Predicted")
    st.pyplot(fig)
    
    st.subheader("Error Distribution (Residuals)")
    residuals = soh_df["Predicted_SoH(%)"] - soh_df["True_SoH(%)"]
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(residuals, bins=30, kde=True, ax=ax2)
    ax2.set_xlabel("Residuals")
    ax2.set_title("Prediction Errors")
    st.pyplot(fig2)
    
    st.subheader("Battery Health Status")
    # Add slider to filter by predicted SoH
    min_soh, max_soh = st.slider("Filter Predicted SoH Range (%)", 0, 100, (0, 100))
    filtered_df = soh_df[(soh_df["Predicted_SoH(%)"] >= min_soh) & (soh_df["Predicted_SoH(%)"] <= max_soh)]
    st.dataframe(filtered_df[["Sample","True_SoH(%)","Predicted_SoH(%)","Status"]])

# ------------------------------
# Tab 3: Thermal Autoencoder
# ------------------------------
with tab3:
    st.header("🔥 Thermal Autoencoder")
    st.markdown("""
        **Goal:** Detect thermal anomalies in battery data (fire risk).  
        **Method:** Sequence autoencoder; anomalies detected via reconstruction error.
    """)
    
    st.subheader("Stats")
    st.write(thermal_stats)
    
    if thermal_errors is not None:
        st.subheader("Thermal Reconstruction Errors (Interactive)")
        start_idx, end_idx = st.slider(
            "Select sample range",
            min_value=0,
            max_value=len(thermal_errors),
            value=(0, min(200, len(thermal_errors)-1)),
            step=1,
            key="thermal_slider"
        )
        threshold = thermal_stats.get("THRESHOLD_95", np.percentile(thermal_errors, 95))
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(np.arange(start_idx, end_idx), thermal_errors[start_idx:end_idx], label="Reconstruction Error")
        ax.axhline(threshold, color='red', linestyle='--', label="Threshold")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Reconstruction Error")
        ax.set_title("Thermal Autoencoder: Errors over Selected Range")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Anomalies in Selected Range")
        anomalies_idx = np.where(thermal_errors[start_idx:end_idx] > threshold)[0] + start_idx
        st.write(f"Total anomalies in range: {len(anomalies_idx)}")
        st.write(anomalies_idx[:10])
    else:
        st.info("⚠️ Thermal reconstruction errors not found. Please save as 'thermal_autoencoder_errors.npy'.")

st.success("✅ Dashboard Loaded Successfully")

with tab4:
    st.header("📊 Summary of All Models")
    st.markdown("""
        This tab consolidates the key metrics of all three models for a quick overview.
    """)
    
    # Fault Autoencoder metrics
    fa_metrics = fault_metrics.copy()
    fa_metrics_display = {
        "Model": "Fault Autoencoder",
        "Precision": fa_metrics.get("PREC", "N/A"),
        "Recall": fa_metrics.get("REC", "N/A"),
        "F1 Score": fa_metrics.get("F1", "N/A"),
        "AUC": fa_metrics.get("AUC", "N/A"),
        "Threshold (95th percentile)": fa_metrics.get("THRESHOLD_95", "N/A"),
        "Num Anomalies Detected": fa_metrics.get("NUM_ANOMALIES_DETECTED", "N/A")
    }
    
    # SoH Regressor metrics (take test set)
    sr_metrics = soh_metrics.get("test", {})
    sr_metrics_display = {
        "Model": "SoH Regressor",
        "MAE": sr_metrics.get("MAE", "N/A"),
        "RMSE": sr_metrics.get("RMSE", "N/A"),
        "R²": sr_metrics.get("R2", "N/A")
    }
    
    # Thermal Autoencoder metrics
    ta_metrics_display = {
        "Model": "Thermal Autoencoder",
        "Threshold (95th percentile)": thermal_stats.get("THRESHOLD_95", "N/A"),
        "Num Anomalies Detected": thermal_stats.get("NUM_ANOMALIES_DETECTED", "N/A")
    }

    # Combine all metrics into a DataFrame
    summary_df = pd.DataFrame([
        fa_metrics_display,
        sr_metrics_display,
        ta_metrics_display
    ])
    
    st.dataframe(summary_df, use_container_width=True)

    # Optional: Add bar plots for comparison
    st.subheader("Metric Comparison")
    fig, ax = plt.subplots(figsize=(10, 4))
    # Only include numeric columns for plotting
    numeric_cols = summary_df.select_dtypes(include=np.number).columns
    summary_df[numeric_cols].plot(kind="bar", ax=ax)
    ax.set_title("Comparison of Key Metrics Across Models")
    ax.set_xticklabels(summary_df["Model"], rotation=0)
    ax.set_ylabel("Metric Value")
    st.pyplot(fig)
