import os
import re
import pandas as pd
import numpy as np

# ============ CONFIG ============
DATA_DIR = "data"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

battery_file = os.path.join(DATA_DIR, "battery_bms_dataset.csv")
motor_file = os.path.join(DATA_DIR, "motor_dataset.csv")
tele_file = os.path.join(DATA_DIR, "tele_dataset_.csv")

# ============ STEP 1: LOAD ============
print("Loading CSV files...")
battery_df = pd.read_csv(battery_file)
motor_df = pd.read_csv(motor_file)
tele_df = pd.read_csv(tele_file)


print("\nShapes:")
print(f"Battery: {battery_df.shape}, Motor: {motor_df.shape}, Telemetry: {tele_df.shape}")

# ============ STEP 2: CLEAN COLUMN NAMES ============
def clean_column_names(df):
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[\(\)%°]+", "", regex=True)
                  .str.replace(" ", "_")
    )
    return df

battery_df = clean_column_names(battery_df)
motor_df = clean_column_names(motor_df)
tele_df = clean_column_names(tele_df)

print("\nCleaned Columns:")
print("Battery:", battery_df.columns.tolist())
print("Motor:", motor_df.columns.tolist())
print("Telemetry:", tele_df.columns.tolist())

# ============ STEP 3: DROP DUPLICATES & HANDLE NA ============
battery_df = battery_df.drop_duplicates()
motor_df = motor_df.drop_duplicates()
tele_df = tele_df.drop_duplicates()

# Fill small gaps if needed (optional)
battery_df = battery_df.fillna(method="ffill")
motor_df = motor_df.fillna(method="ffill")
tele_df = tele_df.fillna(method="ffill")

# ============ STEP 4: CREATE TIMESTAMP IF MISSING ============
if "timestamp" not in battery_df.columns:
    battery_df["timestamp"] = pd.date_range("2023-01-01", periods=len(battery_df), freq="1S")
if "timestamp" not in motor_df.columns:
    motor_df["timestamp"] = pd.date_range("2023-01-01", periods=len(motor_df), freq="1S")
if "timestamp" not in tele_df.columns:
    tele_df["timestamp"] = pd.date_range("2023-01-01", periods=len(tele_df), freq="1S")

# ============ STEP 5: RESAMPLE TO 1 SECOND ============
def resample_numeric(df):
    df = df.set_index("timestamp")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_resampled = df[numeric_cols].resample("1s").mean()
    return df_resampled

battery_df = resample_numeric(battery_df)
motor_df = resample_numeric(motor_df)
tele_df = resample_numeric(tele_df)

# ============ STEP 5.1: RENAME DUPLICATE COLUMNS ============
if "soc_" in battery_df.columns:
    battery_df.rename(columns={"soc_": "battery_soc"}, inplace=True)
if "soc_" in tele_df.columns:
    tele_df.rename(columns={"soc_": "telemetry_soc"}, inplace=True)

# ============ STEP 6: MERGE ALL ============
merged_df = battery_df.join([motor_df, tele_df], how="outer").interpolate()

print("\nMerged shape:", merged_df.shape)
print("Columns after merge:", merged_df.columns.tolist())

# ============ STEP 7: FEATURE ENGINEERING ============
# --- Pack Power ---
if "pack_voltage_v" in merged_df.columns and "pack_current_a" in merged_df.columns:
    merged_df["pack_power_w"] = merged_df["pack_voltage_v"] * merged_df["pack_current_a"]

# --- Rolling Features (multi-window) ---
for w in [5, 30, 60, 300]:  # seconds
    if "pack_current_a" in merged_df.columns:
        merged_df[f"current_mean_{w}s"] = merged_df["pack_current_a"].rolling(window=w).mean()
        merged_df[f"current_std_{w}s"] = merged_df["pack_current_a"].rolling(window=w).std()
    if "pack_voltage_v" in merged_df.columns:
        merged_df[f"voltage_mean_{w}s"] = merged_df["pack_voltage_v"].rolling(window=w).mean()
        merged_df[f"voltage_std_{w}s"] = merged_df["pack_voltage_v"].rolling(window=w).std()

# --- Temperature Difference & Rate ---
if "battery_temp_c" in merged_df.columns and "ambient_temp_c" in merged_df.columns:
    merged_df["temp_diff"] = merged_df["battery_temp_c"] - merged_df["ambient_temp_c"]
    merged_df["battery_temp_rate"] = merged_df["battery_temp_c"].diff()

# --- SoC change per minute ---
if "soc" in merged_df.columns:
    merged_df["soc_delta_1m"] = merged_df["soc"].diff().rolling(window=60).sum()

# --- Energy Throughput ---
if "pack_power_w" in merged_df.columns:
    merged_df["energy_Wh"] = merged_df["pack_power_w"].cumsum() / 3600.0

# --- Charge/Discharge flag ---
if "pack_current_a" in merged_df.columns:
    merged_df["is_charging"] = (merged_df["pack_current_a"] > 0).astype(int)

# --- Voltage Fluctuation ---
if "pack_voltage_v" in merged_df.columns:
    merged_df["voltage_fluctuation"] = merged_df["pack_voltage_v"].rolling(window=30).apply(lambda x: x.max() - x.min())

#--- Normalize SoH if available ---
for soh_col in ["soh", "soh_"]:
    if soh_col in merged_df.columns:
        soh_min, soh_max = merged_df[soh_col].min(), merged_df[soh_col].max()
        merged_df["soh_normalized"] = (merged_df[soh_col] - soh_min) / (soh_max - soh_min)
        break

# ============ STEP 8: SAVE ============
output_file = os.path.join(OUT_DIR, "merged_enhanced.parquet")
merged_df.to_parquet(output_file)
print(f"\n✅ Enhanced dataset saved at: {output_file}")
print(f"Final dataset shape: {merged_df.shape}")

# ============ STEP 9: SUMMARY STATS ============
summary = merged_df.describe().T
summary["missing_pct"] = merged_df.isna().mean() * 100
summary["dtype"] = merged_df.dtypes
summary = summary[["dtype", "count", "missing_pct", "mean", "std", "min", "25%", "50%", "75%", "max"]]

summary_file = os.path.join(OUT_DIR, "data_summary.csv")
summary.to_csv(summary_file)
print(f"Data summary saved at: {summary_file}")

print("\nData Summary:")
print(summary)

# import os
# import re
# import pandas as pd
# import numpy as np

# # ============ CONFIG ============
# DATA_DIR = "data"
# OUT_DIR = "data/processed"
# os.makedirs(OUT_DIR, exist_ok=True)

# battery_file = os.path.join(DATA_DIR, "battery_bms_dataset.csv")
# motor_file = os.path.join(DATA_DIR, "motor_dataset.csv")
# tele_file = os.path.join(DATA_DIR, "tele_dataset_.csv")

# # ============ STEP 1: LOAD ============
# print("Loading CSV files...")
# battery_df = pd.read_csv(battery_file)
# motor_df = pd.read_csv(motor_file)
# tele_df = pd.read_csv(tele_file)

# print("\nShapes:")
# print(f"Battery: {battery_df.shape}, Motor: {motor_df.shape}, Telemetry: {tele_df.shape}")

# # ============ STEP 2: CLEAN COLUMN NAMES ============
# def clean_column_names(df):
#     df.columns = (
#         df.columns.str.strip()
#                   .str.lower()
#                   .str.replace(r"[\(\)%°]+", "", regex=True)
#                   .str.replace(" ", "_")
#     )
#     return df

# battery_df = clean_column_names(battery_df)
# motor_df = clean_column_names(motor_df)
# tele_df = clean_column_names(tele_df)

# print("\nCleaned Columns:")
# print("Battery:", battery_df.columns.tolist())
# print("Motor:", motor_df.columns.tolist())
# print("Telemetry:", tele_df.columns.tolist())

# # ============ STEP 3: DROP DUPLICATES & HANDLE NA ============
# battery_df = battery_df.drop_duplicates()
# motor_df = motor_df.drop_duplicates()
# tele_df = tele_df.drop_duplicates()

# # Fill small gaps if needed (optional)
# battery_df = battery_df.fillna(method="ffill")
# motor_df = motor_df.fillna(method="ffill")
# tele_df = tele_df.fillna(method="ffill")

# # ============ STEP 4: CREATE TIMESTAMP IF MISSING ============
# if "timestamp" not in battery_df.columns:
#     battery_df["timestamp"] = pd.date_range("2023-01-01", periods=len(battery_df), freq="1S")
# if "timestamp" not in motor_df.columns:
#     motor_df["timestamp"] = pd.date_range("2023-01-01", periods=len(motor_df), freq="1S")
# if "timestamp" not in tele_df.columns:
#     tele_df["timestamp"] = pd.date_range("2023-01-01", periods=len(tele_df), freq="1S")

# # ============ STEP 5: RESAMPLE TO 1 SECOND ============
# # Exclude SoH from resampling, keep raw
# soh_col = "soh_" if "soh_" in battery_df.columns else None
# if soh_col:
#     battery_soh = battery_df[[soh_col]]  # keep raw SoH

# def resample_numeric(df, exclude_cols=[]):
#     df = df.set_index("timestamp")
#     numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]
#     df_resampled = df[numeric_cols].resample("1s").mean()
#     return df_resampled

# battery_df_resampled = resample_numeric(battery_df, exclude_cols=[soh_col] if soh_col else [])
# motor_df = resample_numeric(motor_df)
# tele_df = resample_numeric(tele_df)

# # Merge SoH back
# if soh_col:
#     battery_df = battery_df_resampled.join(battery_soh, how="left")
# else:
#     battery_df = battery_df_resampled

# # ============ STEP 5.1: RENAME DUPLICATE COLUMNS ============
# if "soc_" in battery_df.columns:
#     battery_df.rename(columns={"soc_": "battery_soc"}, inplace=True)
# if "soc_" in tele_df.columns:
#     tele_df.rename(columns={"soc_": "telemetry_soc"}, inplace=True)

# # ============ STEP 6: MERGE ALL ============
# merged_df = battery_df.join([motor_df, tele_df], how="outer").interpolate()
# print("\nMerged shape:", merged_df.shape)
# print("Columns after merge:", merged_df.columns.tolist())

# import numpy as np

# # ===== STEP 6.1: SIMULATE SoH IF MISSING OR LOW VARIANCE =====
# soh_col = "soh_" if "soh_" in battery_df.columns else None
# if soh_col:
#     # Check if SoH is missing or flat
#     if battery_df[soh_col].isna().all() or battery_df[soh_col].std() < 0.01:
#         print("Simulating realistic SoH decay...")
#         n = len(merged_df)
#         start_soh = 100.0
#         end_soh = 70.0
#         merged_df[soh_col] = np.linspace(start_soh, end_soh, n)
#         noise = np.random.normal(0, 0.2, n)
#         merged_df[soh_col] += noise
#         merged_df[soh_col] = merged_df[soh_col].clip(lower=50.0, upper=100.0)
#     else:
#         print("Using existing SoH column.")

# # Optional: check distribution
# import matplotlib.pyplot as plt
# plt.figure(figsize=(8,4))
# merged_df[soh_col].plot(kind='hist', bins=50, title='SoH Distribution After Simulation')
# plt.xlabel('SoH (%)')
# plt.show()

# # ===== STEP 6.2: OPTIONAL: SAVE ORIGINAL SoH BEFORE FEATURE ENGINEERING =====
# if soh_col:
#     merged_df[f"{soh_col}_raw"] = merged_df[soh_col]

# # ============ STEP 7: FEATURE ENGINEERING ============
# # --- Pack Power ---
# if "pack_voltage_v" in merged_df.columns and "pack_current_a" in merged_df.columns:
#     merged_df["pack_power_w"] = merged_df["pack_voltage_v"] * merged_df["pack_current_a"]

# # --- Rolling Features (multi-window) ---
# for w in [5, 30, 60, 300]:  # seconds
#     if "pack_current_a" in merged_df.columns:
#         merged_df[f"current_mean_{w}s"] = merged_df["pack_current_a"].rolling(window=w).mean()
#         merged_df[f"current_std_{w}s"] = merged_df["pack_current_a"].rolling(window=w).std()
#     if "pack_voltage_v" in merged_df.columns:
#         merged_df[f"voltage_mean_{w}s"] = merged_df["pack_voltage_v"].rolling(window=w).mean()
#         merged_df[f"voltage_std_{w}s"] = merged_df["pack_voltage_v"].rolling(window=w).std()

# # --- Temperature Difference & Rate ---
# if "battery_temp_c" in merged_df.columns and "ambient_temp_c" in merged_df.columns:
#     merged_df["temp_diff"] = merged_df["battery_temp_c"] - merged_df["ambient_temp_c"]
#     merged_df["battery_temp_rate"] = merged_df["battery_temp_c"].diff()

# # --- SoC change per minute ---
# if "soc" in merged_df.columns:
#     merged_df["soc_delta_1m"] = merged_df["soc"].diff().rolling(window=60).sum()

# # --- Energy Throughput ---
# if "pack_power_w" in merged_df.columns:
#     merged_df["energy_Wh"] = merged_df["pack_power_w"].cumsum() / 3600.0

# # --- Charge/Discharge flag ---
# if "pack_current_a" in merged_df.columns:
#     merged_df["is_charging"] = (merged_df["pack_current_a"] > 0).astype(int)

# # --- Voltage Fluctuation ---
# if "pack_voltage_v" in merged_df.columns:
#     merged_df["voltage_fluctuation"] = merged_df["pack_voltage_v"].rolling(window=30).apply(lambda x: x.max() - x.min())

# # ============ STEP 8: SAVE ============
# output_file = os.path.join(OUT_DIR, "merged_enhanced.parquet")
# merged_df.to_parquet(output_file)
# print(f"\n✅ Enhanced dataset saved at: {output_file}")
# print(f"Final dataset shape: {merged_df.shape}")

# # ============ STEP 9: SUMMARY STATS ============
# summary = merged_df.describe().T
# summary["missing_pct"] = merged_df.isna().mean() * 100
# summary["dtype"] = merged_df.dtypes
# summary = summary[["dtype", "count", "missing_pct", "mean", "std", "min", "25%", "50%", "75%", "max"]]
# summary_file = os.path.join(OUT_DIR, "data_summary.csv")
# summary.to_csv(summary_file)
# print(f"Data summary saved at: {summary_file}")
# print("\nData Summary:")
# print(summary)

# # ===== STEP 10: OPTIONAL TRAIN/VAL SPLIT CHECK =====
# from sklearn.model_selection import train_test_split

# if soh_col:
#     train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42, shuffle=True)
#     print(f"Train SoH range: {train_df[soh_col].min():.2f} - {train_df[soh_col].max():.2f}")
#     print(f"Test SoH range: {test_df[soh_col].min():.2f} - {test_df[soh_col].max():.2f}")

# import os
# import re
# import pandas as pd
# import numpy as np

# # ============ CONFIG ============
# DATA_DIR = "data"
# OUT_DIR = "data/processed"
# os.makedirs(OUT_DIR, exist_ok=True)

# battery_file = os.path.join(DATA_DIR, "battery_bms_dataset.csv")
# motor_file = os.path.join(DATA_DIR, "motor_dataset.csv")
# tele_file = os.path.join(DATA_DIR, "tele_dataset_.csv")

# # ============ STEP 1: LOAD ============
# print("Loading CSV files...")
# battery_df = pd.read_csv(battery_file)
# motor_df = pd.read_csv(motor_file)
# tele_df = pd.read_csv(tele_file)

# print("\nShapes:")
# print(f"Battery: {battery_df.shape}, Motor: {motor_df.shape}, Telemetry: {tele_df.shape}")

# # ============ STEP 2: CLEAN COLUMN NAMES ============
# def clean_column_names(df):
#     df.columns = (
#         df.columns.str.strip()
#                   .str.lower()
#                   .str.replace(r"[\(\)%°]+", "", regex=True)
#                   .str.replace(" ", "_")
#     )
#     return df

# battery_df = clean_column_names(battery_df)
# motor_df = clean_column_names(motor_df)
# tele_df = clean_column_names(tele_df)

# print("\nCleaned Columns:")
# print("Battery:", battery_df.columns.tolist())
# print("Motor:", motor_df.columns.tolist())
# print("Telemetry:", tele_df.columns.tolist())

# # ============ STEP 3: DROP DUPLICATES & HANDLE NA ============
# battery_df = battery_df.drop_duplicates()
# motor_df = motor_df.drop_duplicates()
# tele_df = tele_df.drop_duplicates()

# # Fill small gaps if needed (optional)
# battery_df = battery_df.fillna(method="ffill")
# motor_df = motor_df.fillna(method="ffill")
# tele_df = tele_df.fillna(method="ffill")

# # ============ STEP 4: CREATE TIMESTAMP IF MISSING ============
# if "timestamp" not in battery_df.columns:
#     battery_df["timestamp"] = pd.date_range("2023-01-01", periods=len(battery_df), freq="1S")
# if "timestamp" not in motor_df.columns:
#     motor_df["timestamp"] = pd.date_range("2023-01-01", periods=len(motor_df), freq="1S")
# if "timestamp" not in tele_df.columns:
#     tele_df["timestamp"] = pd.date_range("2023-01-01", periods=len(tele_df), freq="1S")

# # ============ STEP 5: RESAMPLE TO 1 SECOND ============
# def resample_numeric(df):
#     df = df.set_index("timestamp")
#     numeric_cols = df.select_dtypes(include=[np.number]).columns
#     df_resampled = df[numeric_cols].resample("1s").mean()
#     return df_resampled

# battery_df = resample_numeric(battery_df)
# motor_df = resample_numeric(motor_df)
# tele_df = resample_numeric(tele_df)

# # ============ STEP 5.1: RENAME DUPLICATE COLUMNS ============
# if "soc_" in battery_df.columns:
#     battery_df.rename(columns={"soc_": "battery_soc"}, inplace=True)
# if "soc_" in tele_df.columns:
#     tele_df.rename(columns={"soc_": "telemetry_soc"}, inplace=True)



# # ============ STEP 6: MERGE ALL ============

# merged_df = battery_df.join([motor_df, tele_df], how="outer").interpolate()
# print("\nMerged shape:", merged_df.shape)
# print("Columns after merge:", merged_df.columns.tolist())


# # ============ STEP 7: FEATURE ENGINEERING ============

# # --- Pack Power ---
# if "pack_voltage_v" in merged_df.columns and "pack_current_a" in merged_df.columns:
#     merged_df["pack_power_w"] = merged_df["pack_voltage_v"] * merged_df["pack_current_a"]

# # --- Rolling Features (multi-window) ---
# for w in [5, 30, 60, 300]:  # seconds
#     if "pack_current_a" in merged_df.columns:
#         merged_df[f"current_mean_{w}s"] = merged_df["pack_current_a"].rolling(window=w).mean()
#         merged_df[f"current_std_{w}s"] = merged_df["pack_current_a"].rolling(window=w).std()
#     if "pack_voltage_v" in merged_df.columns:
#         merged_df[f"voltage_mean_{w}s"] = merged_df["pack_voltage_v"].rolling(window=w).mean()
#         merged_df[f"voltage_std_{w}s"] = merged_df["pack_voltage_v"].rolling(window=w).std()

# # --- Temperature Difference & Rate ---
# if "battery_temp_c" in merged_df.columns and "ambient_temp_c" in merged_df.columns:
#     merged_df["temp_diff"] = merged_df["battery_temp_c"] - merged_df["ambient_temp_c"]
#     merged_df["battery_temp_rate"] = merged_df["battery_temp_c"].diff()

# # --- SoC change per minute ---
# if "soc" in merged_df.columns:
#     merged_df["soc_delta_1m"] = merged_df["soc"].diff().rolling(window=60).sum()

# # --- Energy Throughput ---
# if "pack_power_w" in merged_df.columns:
#     merged_df["energy_Wh"] = merged_df["pack_power_w"].cumsum() / 3600.0

# # --- Charge/Discharge flag ---
# if "pack_current_a" in merged_df.columns:
#     merged_df["is_charging"] = (merged_df["pack_current_a"] > 0).astype(int)

# # --- Voltage Fluctuation ---
# if "pack_voltage_v" in merged_df.columns:
#     merged_df["voltage_fluctuation"] = merged_df["pack_voltage_v"].rolling(window=30).apply(lambda x: x.max() - x.min())

# # --- Normalize SoH if available ---
# # for soh_col in ["soh", "soh_"]:
# #     if soh_col in merged_df.columns:
# #         soh_min, soh_max = merged_df[soh_col].min(), merged_df[soh_col].max()
# #         merged_df["soh_normalized"] = (merged_df[soh_col] - soh_min) / (soh_max - soh_min)
# #         break


# # ============ STEP 8: SAVE ============
# output_file = os.path.join(OUT_DIR, "merged_enhanced.parquet")
# merged_df.to_parquet(output_file)
# print(f"\n✅ Enhanced dataset saved at: {output_file}")
# print(f"Final dataset shape: {merged_df.shape}")
# # ============ STEP 9: SUMMARY STATS ============       
# summary = merged_df.describe().T
# summary["missing_pct"] = merged_df.isna().mean()        * 100
# summary["dtype"] = merged_df.dtypes         
# summary = summary[["dtype", "count", "missing_pct", "mean", "std", "min", "25%", "50%", "75%", "max"]]
# summary_file = os.path.join(OUT_DIR, "data_summary.csv")            
# summary.to_csv(summary_file)
# print(f"Data summary saved at: {summary_file}") 
# print("\nData Summary:")
# print(summary)


