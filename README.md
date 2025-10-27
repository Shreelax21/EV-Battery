# AI-Based EV Battery Health & Fault Detection System

## **Project Overview**

This project leverages AI to monitor **EV battery health, detect anomalies, and predict thermal risks**, enabling safer and more efficient EV operation.
It integrates **tabular and sequence-based models** to provide predictive insights on battery State of Health (SoH), motor/battery faults, and thermal anomalies.

**Key Features:**

* Detects motor and battery anomalies using **autoencoders**.
* Predicts battery SoH using **Random Forest Regression**.
* Identifies thermal runaway risks using **LSTM Autoencoder**.
* Generates **sliding windows** for AI models from raw telemetry.
* Provides a **real-time control loop** to decide charging mode: FAST, SLOW, or HOLD.

---

## **Project Structure**

```
EV-Battery/
│
├─ data/
│   ├─ raw/
│   │   ├─ battery_bms_dataset.csv
│   │   ├─ motor_dataset.csv
│   │   └─ tele_dataset_.csv
│   └─ processed/
│       ├─ merged_enhanced.parquet
│       ├─ data_summary.csv
│       └─ windows/
│           ├─ windows_seq.npz
│           ├─ windows_tabular.npz
│           ├─ feature_cols.npy
│           ├─ window_starts.npy
│           └─ splits.joblib
│
├─ data_scripts/
│   ├─ 01_load_data.py         # Load & clean datasets
│   └─ 02_make_windows.py      # Create tabular & sequence windows
│
├─ training/
│   ├─ train_fault_encoder.py  # Autoencoder for fault detection
│   ├─ train_soh_regressor.py  # Random Forest for battery SoH
│   └─ train_thermal_autoencoder.py # LSTM for thermal anomaly detection
│
├─ inference/
│   └─ control_loop.py         # Real-time inference & charging decision
│
├─ models/                     # Trained model outputs
├─ requirements.txt
└─ README.md
```

---

## **Task A – Environment Setup**

1. **Install prerequisites:**

```bash
python --version
git --version
pip install -r requirements.txt
```

2. **Create and activate virtual environment (Windows):**

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. **VS Code extensions recommended:** Python, Pylance, Jupyter, GitLens.

4. **Test environment:**

```python
import numpy as np
print("Environment ready")
```

---

## **Task B – Data Processing**

### Step 1: Load & Clean Datasets

* Input CSVs: battery, motor, telemetry.
* Merge, clean, and resample to 1-second intervals.
* Add engineered features like rolling stats, timestamp windows.
* Save outputs to Parquet and CSV for ML models.

Run:

```bash
python data_scripts/01_load_data.py
```

### Step 2: Create Sliding Windows

* Generate **tabular windows** for Random Forest and autoencoder.
* Generate **sequence windows** for LSTM thermal model.
* Outputs saved to `data/processed/windows/`.

Run:

```bash
python data_scripts/02_make_windows.py
```

**Explanation:** Sliding windows capture temporal patterns in battery and motor behavior for AI models.

---

## **Task C – AI Model Training**

| Task                      | Script                         | Model            | Output                      |
| ------------------------- | ------------------------------ | ---------------- | --------------------------- |
| Fault Detection           | `train_fault_encoder.py`       | Autoencoder      | Detects anomalies & metrics |
| SoH Regression            | `train_soh_regressor.py`       | Random Forest    | Predicts battery SoH (%)    |
| Thermal Anomaly Detection | `train_thermal_autoencoder.py` | LSTM Autoencoder | Detects thermal anomalies   |

### **1. Fault Detection – Autoencoder**

* Learns **normal battery/motor behavior**.
* High reconstruction error → anomaly.
* Metrics: precision, recall, F1, AUC.

Run:

```bash
python training/train_fault_encoder.py
```

### **2. Battery Health Regression – Random Forest**

* Predicts **battery State of Health (SoH)**.
* SoH < 80% → battery unhealthy.
* Metrics: MAE, RMSE, R².

Run:

```bash
python training/train_soh_regressor.py
```

### **3. Thermal Anomaly Detection – LSTM Autoencoder**

* Detects **potential overheating events**.
* Uses sequence windows (temporal patterns).
* Threshold-based detection (95th percentile).

Run:

```bash
python training/train_thermal_autoencoder.py
```

---

## **Task D – Real-Time Inference & Control Loop**

Script: `inference/control_loop.py`

### Features:

1. Loads **tabular & sequence windows**.
2. Loads trained models (SoH, fault, thermal).
3. Computes:

   * Battery SoH prediction
   * Fault reconstruction score
   * Thermal anomaly probability
4. Decides **charging mode**:

   * **FAST**: All checks healthy
   * **SLOW**: Minor issues detected
   * **HOLD**: Multiple issues detected

Run demo:

```bash
python inference/control_loop.py
```

**Example Output:**

```
⚡ Decision: SLOW | Reason: SoH low (74.3%), Fault detected (score=0.015)
```

---

## **Methodology Overview**

1. **Data Collection:** EV battery, motor, and telemetry datasets.
2. **Preprocessing & Feature Engineering:** Cleaning, merging, rolling statistics.
3. **Windowing:** Tabular & sequence windows for model input.
4. **Model Training:**

   * Fault detection → Autoencoder
   * Battery health → Random Forest
   * Thermal anomaly → LSTM Autoencoder
5. **Evaluation & Thresholding:** Metrics and 95th percentile anomaly thresholds.
6. **Control Loop:** Integrates model outputs to decide EV charging mode.

---

## **Benefits**

* ⚡ **Safety:** Detect faults and overheating early.
* 💰 **Cost-saving:** Prevent expensive EV failures.
* ⚙️ **Scalable:** Works across different EV datasets.
* 🔄 **Flexible:** Models can be retrained on new data easily.

---

## **Team 2 Summary**

| Model                    | Script                         | Purpose                         |
| ------------------------ | ------------------------------ | ------------------------------- |
| Fault Autoencoder        | `train_fault_encoder.py`       | Detect battery/motor anomalies  |
| SoH Regressor            | `train_soh_regressor.py`       | Predict battery State of Health |
| Thermal LSTM Autoencoder | `train_thermal_autoencoder.py` | Detect thermal risks            |

---
Absolutely! Using all the details you’ve provided so far (Task A, B, C, models, inference/control loop, methodology, team summary, outputs), I’ve drafted a **full, detailed, professional README.md** suitable for GitHub. I’ve structured it logically so anyone can follow from environment setup to running the full EV AI pipeline.

Here’s the complete version:

---

# **AI-Powered EV Battery Health Prediction & Fault Detection System**

**Team:** AIML-TEAM 2 – Analyze, Predict, and Detect Anomalies

---

## **Project Overview**

This project aims to **predict the health of electric vehicle (EV) batteries, detect potential faults, and identify thermal anomalies** using AI/ML models. It provides **real-time decision support for charging and safety management**.

Key functionalities include:

* **Battery State-of-Health (SoH) Estimation** – Predict whether the battery is healthy or degraded.
* **Fault Detection** – Detect motor or battery anomalies using Autoencoders.
* **Thermal Anomaly Detection** – Identify risk of overheating using LSTM Autoencoder.
* **Control Loop / Charging Decision** – Suggest safe charging modes (FAST, SLOW, HOLD) based on predictions.

---

## **Methodology**

The pipeline is divided into **Tasks A, B, C, and D**, forming a complete AI workflow for EV battery analysis:

### **Task A – Data Loading & Preprocessing**

1. **Environment Setup**

   * Install software: Python >=3.9, VS Code, Git
   * Initialize project folder and Git:

     ```bash
     mkdir ev-aiml
     cd ev-aiml
     git init
     python -m venv .venv
     .venv\Scripts\activate
     pip install -r requirements.txt
     ```
   * Verify installations:

     ```bash
     python --version
     git --version
     code --version
     ```

2. **Data Preparation**

   * Create `data/` folder and upload datasets:

     * `battery_bms_dataset.csv`
     * `motor_dataset.csv`
     * `tele_dataset_.csv`

3. **Load and Clean Data** – `data_scripts/01_load_data.py`

   * Standardizes column names
   * Fills missing data
   * Adds timestamps
   * Resamples to 1-second intervals
   * Merges datasets into a single DataFrame
   * Generates advanced features: power, energy throughput, rolling statistics, temperature deltas, SoH normalization
   * Outputs:

     * `data/processed/merged_enhanced.parquet`
     * `data/processed/data_summary.csv`

**Parquet Format:** Optimized for ML pipelines; fast read/write, compressed, preserves column types.

---

### **Task B – Windowing**

**Script:** `data_scripts/02_make_windows.py`

* Converts time-series data into **fixed-length windows** for ML models.
* Outputs:

  * `windows_seq.npz` → 3D array for sequence models
  * `windows_tabular.npz` → Aggregated features for tree models
  * `feature_cols.npy` → List of numeric features
  * `window_starts.npy` → Start timestamps
  * `splits.joblib` → Train/validation/test indices
  * `scaler.joblib` → StandardScaler for features

**Why Windowing:** ML models require fixed-size inputs; EV battery anomalies depend on temporal patterns.

---

### **Task C – Model Training**

#### **1. Battery Fault Detection – Autoencoder**

**Script:** `training/train_fault_encoder.py`

* Trains an unsupervised Autoencoder on normal samples
* Detects faults using reconstruction error
* Outputs:

  * `models/fault_autoencoder.keras`
  * `models/fault_autoencoder_metrics.joblib`

**Key Points:**

* Encoder compresses input → bottleneck
* Decoder reconstructs input
* High reconstruction error → anomaly/fault
* Threshold-based anomaly detection using 95th percentile

---

#### **2. Battery SoH Estimation – Random Forest Regressor**

**Script:** `training/train_soh_regressor.py`

* Predicts battery health (% SoH)
* Outputs:

  * `models/soh_regressor.joblib`
  * `models/soh_regressor_metrics.joblib`

**Interpretation:**

* SoH < 80% → Battery considered unhealthy
* High accuracy on training/validation; low generalization on small test sets may occur

---

#### **3. Thermal Anomaly Detection – LSTM Autoencoder**

**Script:** `training/train_thermal_autoencoder.py`

* Detects abnormal temperature patterns in battery/motor
* Outputs:

  * `models/thermal_autoencoder.keras`
  * `models/thermal_autoencoder_stats.joblib`

**Key Points:**

* Sequence-to-sequence LSTM Autoencoder
* High reconstruction error → thermal anomaly
* 95th percentile threshold identifies risky windows

---

### **Task D – Inference & Control Loop**

**Script:** `inference/control_loop.py`

* Loads trained models & processed windows
* Predicts SoH, fault score, thermal probability
* Decides **charging mode**:

  * **FAST** → Battery healthy
  * **SLOW** → Minor issues
  * **HOLD** → Multiple issues detected

**Sample Output:**

```
⚡ Decision: SLOW | Reason: SoH low (74.3%), Fault detected (score=0.015)
```

**Custom Thresholds:**

* `soh_thresh_low=0.75`, `fault_thresh=0.01`, `thermal_thresh=0.6`
* Adjustable for safer or more aggressive operation

---

## **Team Summary – AIML TEAM 2**

| Member Name        | Role              | Responsibility                                     |
| ------------------ | ----------------- | -------------------------------------------------- |
| Shreelakshmi Hegde | AI/ML Developer   | Data preprocessing, model training, SoH prediction |
| Akshay M           | Backend Developer | Control loop, inference integration                |
| Pratham            | Data Engineer     | Windowing, feature engineering                     |

---

## **Quick Start – Running the Full Pipeline**

1. **Clone Repository**

```bash
git clone https://github.com/Shreelax21/EV-Battery.git
cd EV-Battery
```

2. **Set Up Environment**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. **Preprocess Data**

```bash
python data_scripts/01_load_data.py
python data_scripts/02_make_windows.py
```

4. **Train Models**

```bash
python training/train_fault_encoder.py
python training/train_soh_regressor.py
python training/train_thermal_autoencoder.py
```

5. **Run Inference / Charging Decision**

```bash
python inference/control_loop.py
``

---
**Conclusion**

This AI-based EV Battery Health & Fault Detection System provides a comprehensive solution for EV battery monitoring and safety management. By integrating fault detection, SoH prediction, and thermal anomaly identification, it ensures informed charging decisions, enhances vehicle safety, and reduces operational costs.

The modular, scalable design allows for retraining with new data, adaptation to different EV platforms, and continuous monitoring for performance optimization. This project highlights the critical role of AI in advancing electric vehicle reliability and efficiency.

