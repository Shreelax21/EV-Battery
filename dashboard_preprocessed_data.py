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
df = df.reset_index()  # Ensure timestamp is a column
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
    st.subheader("🔹 Battery State of Charge (SoC) Over Time")
    st.markdown(
        """
        **Overview:**  
        State of Charge (SoC) is the percentage of battery capacity remaining.  
        - 100% = fully charged, 0% = fully discharged.  
        - SoC indicates how much energy is available in the battery at any time.
        
        **Insights:**  
        - Smooth decreases during driving indicate normal usage.  
        - Rapid drops may signal high load or abnormal energy consumption.  
        - Flat lines near 100% or 0% indicate periods when the vehicle is idle or fully charged/discharged.

        **Applications:**  
        - Planning charging cycles.  
        - Detecting unusual energy drains.  
        - Estimating remaining driving range.
        """
    )
    fig_soc = px.line(df_filtered, x="timestamp", y="soc", labels={"soc": "SoC (%)"})
    st.plotly_chart(fig_soc, use_container_width=True)

# ======================
# Pack Power
# ======================
if "pack_power_w" in df_filtered.columns:
    st.subheader("🔹 Battery Pack Power Over Time")
    st.markdown(
        """
        **Overview:**  
        Pack Power (W) = Voltage × Current. Shows energy flow at each moment.  

        **Insights:**  
        - Positive power = discharging (vehicle consuming battery).  
        - Negative or low power = charging (battery being replenished).  
        - Sudden spikes indicate high acceleration, heavy load, or regenerative braking.

        **Applications:**  
        - Monitoring high-load events.  
        - Understanding energy usage patterns.  
        - Detecting anomalies in power delivery or regeneration.
        """
    )
    fig_power = px.line(df_filtered, x="timestamp", y="pack_power_w")
    st.plotly_chart(fig_power, use_container_width=True)

# ======================
# Charging vs Discharging
# ======================
if "pack_current_a" in df_filtered.columns and "is_charging" in df_filtered.columns:
    st.subheader("🔹 Charging vs Discharging Periods")
    st.markdown(
        """
        **Overview:**  
        Shows current (A) over time, highlighting whether the battery is charging or discharging.  

        **Insights:**  
        - Green = charging, Red = discharging.  
        - Patterns show the frequency and duration of charging cycles.  
        - Sudden changes in current indicate fast acceleration, heavy load, or abnormal events.

        **Applications:**  
        - Evaluate charging strategy efficiency.  
        - Detect unusual battery usage.  
        - Diagnose potential BMS issues.
        """
    )
    fig_charge = px.scatter(
        df_filtered, x="timestamp", y="pack_current_a",
        color="is_charging", color_continuous_scale=["red", "green"]
    )
    st.plotly_chart(fig_charge, use_container_width=True)

# ======================
# Temperature
# ======================
if "battery_temp_c" in df_filtered.columns and "ambient_temp_c" in df_filtered.columns:
    st.subheader("🔹 Battery vs Ambient Temperature")
    st.markdown(
        """
        **Overview:**  
        Compares battery temperature with ambient temperature. Temperature management is critical for battery life and safety.  

        **Insights:**  
        - Gradual rise in battery temp during use is normal.  
        - Rapid increase may indicate thermal runaway risk or cooling system failure.  
        - Very low battery temps may reduce efficiency and capacity.

        **Applications:**  
        - Prevent overheating and thermal runaway.  
        - Optimize battery cooling and heating strategies.  
        - Evaluate environmental impact on battery performance.
        """
    )
    fig_temp = go.Figure()
    fig_temp.add_trace(go.Line(x=df_filtered["timestamp"], y=df_filtered["battery_temp_c"], name="Battery Temp"))
    fig_temp.add_trace(go.Line(x=df_filtered["timestamp"], y=df_filtered["ambient_temp_c"], name="Ambient Temp"))
    st.plotly_chart(fig_temp, use_container_width=True)

# ======================
# Voltage Fluctuation
# ======================
if "voltage_fluctuation" in df_filtered.columns:
    st.subheader("🔹 Voltage Fluctuation (30s Window)")
    st.markdown(
        """
        **Overview:**  
        Rolling voltage fluctuation = max - min voltage in a 30-second window.  

        **Insights:**  
        - High fluctuation may indicate unstable power delivery.  
        - Can be caused by high acceleration, regenerative braking, or BMS issues.  
        - Low fluctuation = stable operation.

        **Applications:**  
        - Identify potential electrical issues.  
        - Assess battery and motor system stability.  
        - Detect abnormal voltage spikes.
        """
    )
    fig_volt = px.line(df_filtered, x="timestamp", y="voltage_fluctuation")
    st.plotly_chart(fig_volt, use_container_width=True)

# ======================
# Energy Throughput
# ======================
if "energy_Wh" in df_filtered.columns:
    st.subheader("🔹 Cumulative Energy Throughput")
    st.markdown(
        """
        **Overview:**  
        Energy throughput (Wh) is the cumulative energy delivered or consumed by the battery.  

        **Insights:**  
        - Shows total energy usage over time.  
        - Sudden jumps indicate high power events or rapid charging/discharging cycles.  

        **Applications:**  
        - Track total energy efficiency.  
        - Understand driving or load patterns.  
        - Evaluate battery performance and aging.
        """
    )
    fig_energy = px.line(df_filtered, x="timestamp", y="energy_Wh")
    st.plotly_chart(fig_energy, use_container_width=True)

# ======================
# Correlation Heatmap
# ======================
corr_cols = df_filtered.select_dtypes(include=["float64", "int64"])
if not corr_cols.empty:
    st.subheader("🔹 Feature Correlation Heatmap")
    st.markdown(
        """
        **Overview:**  
        Heatmap showing correlation between numeric features in the dataset.  

        **Insights:**  
        - Values near 1 → strong positive correlation.  
        - Values near -1 → strong negative correlation.  
        - Close to 0 → weak/no correlation.  

        **Applications:**  
        - Identify dependent variables for modeling.  
        - Detect redundant or highly correlated features.  
        - Understand relationships, e.g., voltage-current vs SoC or power.
        """
    )
    fig_corr = px.imshow(corr_cols.corr(), text_auto=True, aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

st.success("✅ Dashboard Loaded Successfully")
# import pandas as pd
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go

# # ======================
# # LOAD DATA
# # ======================
# @st.cache_data
# def load_data():
#     return pd.read_parquet("data/processed/merged_enhanced.parquet")

# df = load_data()
# df = df.reset_index()   # ensure timestamp is a column
# if "timestamp" not in df.columns:
#     df["timestamp"] = range(len(df))

# st.set_page_config(page_title="EV Battery Dashboard", layout="wide")
# st.title("🔋 EV Battery & Motor Data Dashboard")

# # ======================
# # SIDEBAR FILTERS
# # ======================
# st.sidebar.header("Controls")
# start_time, end_time = st.sidebar.slider(
#     "Select Time Range",
#     min_value=int(df.index.min()),
#     max_value=int(df.index.max()),
#     value=(int(df.index.min()), int(df.index.max()))
# )
# df_filtered = df.iloc[start_time:end_time]

# # ======================
# # SOC Over Time
# # ======================
# if "soc" in df_filtered.columns:
#     fig_soc = px.line(df_filtered, x="timestamp", y="soc",
#                       title="Battery State of Charge (%) Over Time",
#                       labels={"soc": "SoC (%)"})
#     st.plotly_chart(fig_soc, use_container_width=True)

# # ======================
# # Pack Power
# # ======================
# if "pack_power_w" in df_filtered.columns:
#     fig_power = px.line(df_filtered, x="timestamp", y="pack_power_w",
#                         title="Battery Pack Power (W)")
#     st.plotly_chart(fig_power, use_container_width=True)

# # ======================
# # Charging vs Discharging
# # ======================
# if "pack_current_a" in df_filtered.columns and "is_charging" in df_filtered.columns:
#     fig_charge = px.scatter(
#         df_filtered, x="timestamp", y="pack_current_a",
#         color="is_charging", color_continuous_scale=["red", "green"],
#         title="Charging vs Discharging (Current)"
#     )
#     st.plotly_chart(fig_charge, use_container_width=True)

# # ======================
# # Temperature
# # ======================
# if "battery_temp_c" in df_filtered.columns and "ambient_temp_c" in df_filtered.columns:
#     fig_temp = go.Figure()
#     fig_temp.add_trace(go.Line(x=df_filtered["timestamp"], y=df_filtered["battery_temp_c"], name="Battery Temp"))
#     fig_temp.add_trace(go.Line(x=df_filtered["timestamp"], y=df_filtered["ambient_temp_c"], name="Ambient Temp"))
#     fig_temp.update_layout(title="Battery vs Ambient Temperature (°C)")
#     st.plotly_chart(fig_temp, use_container_width=True)

# # ======================
# # Voltage Fluctuation
# # ======================
# if "voltage_fluctuation" in df_filtered.columns:
#     fig_volt = px.line(df_filtered, x="timestamp", y="voltage_fluctuation",
#                        title="Voltage Fluctuation (30s Window)")
#     st.plotly_chart(fig_volt, use_container_width=True)

# # ======================
# # Energy Throughput
# # ======================
# if "energy_Wh" in df_filtered.columns:
#     fig_energy = px.line(df_filtered, x="timestamp", y="energy_Wh",
#                          title="Cumulative Energy Throughput (Wh)")
#     st.plotly_chart(fig_energy, use_container_width=True)

# # ======================
# # Correlation Heatmap
# # ======================
# corr_cols = df_filtered.select_dtypes(include=["float64", "int64"])
# if not corr_cols.empty:
#     fig_corr = px.imshow(corr_cols.corr(), text_auto=True, aspect="auto",
#                          title="Feature Correlation Heatmap")
#     st.plotly_chart(fig_corr, use_container_width=True)

# st.success("✅ Dashboard Loaded Successfully")
