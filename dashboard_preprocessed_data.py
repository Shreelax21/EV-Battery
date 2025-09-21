import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    return pd.read_parquet("data/processed/merged_enhanced.parquet")

df = load_data()
df = df.reset_index()   # ensure timestamp is a column
if "timestamp" not in df.columns:
    df["timestamp"] = range(len(df))

st.set_page_config(page_title="EV Battery Dashboard", layout="wide")
st.title("🔋 EV Battery & Motor Data Dashboard")

# ======================
# SIDEBAR FILTERS
# ======================
st.sidebar.header("Controls")
start_time, end_time = st.sidebar.slider(
    "Select Time Range",
    min_value=int(df.index.min()),
    max_value=int(df.index.max()),
    value=(int(df.index.min()), int(df.index.max()))
)
df_filtered = df.iloc[start_time:end_time]

# ======================
# SOC Over Time
# ======================
if "soc" in df_filtered.columns:
    fig_soc = px.line(df_filtered, x="timestamp", y="soc",
                      title="Battery State of Charge (%) Over Time",
                      labels={"soc": "SoC (%)"})
    st.plotly_chart(fig_soc, use_container_width=True)

# ======================
# Pack Power
# ======================
if "pack_power_w" in df_filtered.columns:
    fig_power = px.line(df_filtered, x="timestamp", y="pack_power_w",
                        title="Battery Pack Power (W)")
    st.plotly_chart(fig_power, use_container_width=True)

# ======================
# Charging vs Discharging
# ======================
if "pack_current_a" in df_filtered.columns and "is_charging" in df_filtered.columns:
    fig_charge = px.scatter(
        df_filtered, x="timestamp", y="pack_current_a",
        color="is_charging", color_continuous_scale=["red", "green"],
        title="Charging vs Discharging (Current)"
    )
    st.plotly_chart(fig_charge, use_container_width=True)

# ======================
# Temperature
# ======================
if "battery_temp_c" in df_filtered.columns and "ambient_temp_c" in df_filtered.columns:
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Line(x=df_filtered["timestamp"], y=df_filtered["battery_temp_c"], name="Battery Temp"))
    fig_temp.add_trace(go.Line(x=df_filtered["timestamp"], y=df_filtered["ambient_temp_c"], name="Ambient Temp"))
    fig_temp.update_layout(title="Battery vs Ambient Temperature (°C)")
    st.plotly_chart(fig_temp, use_container_width=True)

# ======================
# Voltage Fluctuation
# ======================
if "voltage_fluctuation" in df_filtered.columns:
    fig_volt = px.line(df_filtered, x="timestamp", y="voltage_fluctuation",
                       title="Voltage Fluctuation (30s Window)")
    st.plotly_chart(fig_volt, use_container_width=True)

# ======================
# Energy Throughput
# ======================
if "energy_Wh" in df_filtered.columns:
    fig_energy = px.line(df_filtered, x="timestamp", y="energy_Wh",
                         title="Cumulative Energy Throughput (Wh)")
    st.plotly_chart(fig_energy, use_container_width=True)

# ======================
# Correlation Heatmap
# ======================
corr_cols = df_filtered.select_dtypes(include=["float64", "int64"])
if not corr_cols.empty:
    fig_corr = px.imshow(corr_cols.corr(), text_auto=True, aspect="auto",
                         title="Feature Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)

st.success("✅ Dashboard Loaded Successfully")
