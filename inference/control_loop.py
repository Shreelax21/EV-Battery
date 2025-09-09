# inference/control_loop.py
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

TAB_PATH = os.path.join("data", "processed", "windows", "windows_tabular.npz")
SEQ_PATH = os.path.join("data", "processed", "windows", "windows_seq.npz")
SPLITS_PATH = os.path.join("data", "processed", "windows", "splits.joblib")

MODELS_DIR = "models"

def load_artifacts():
    tab = np.load(TAB_PATH, allow_pickle=True)
    Xagg = tab["Xagg"]
    seq = np.load(SEQ_PATH, allow_pickle=True)
    Xseq = seq["X"]
    splits = joblib.load(SPLITS_PATH)

    soh_model = joblib.load(os.path.join(MODELS_DIR, "soh_regressor.joblib")) if os.path.exists(os.path.join(MODELS_DIR, "soh_regressor.joblib")) else None
    fault_model = load_model(os.path.join(MODELS_DIR, "fault_autoencoder.keras")) if os.path.exists(os.path.join(MODELS_DIR, "fault_autoencoder.keras")) else None
    thermal_model = load_model(os.path.join(MODELS_DIR, "thermal_lstm.keras")) if os.path.exists(os.path.join(MODELS_DIR, "thermal_lstm.keras")) else None

    return Xagg, Xseq, splits, soh_model, fault_model, thermal_model

def anomaly_score(fault_model, x_tab):
    recon = fault_model.predict(x_tab)
    err = np.mean(np.square(x_tab - recon), axis=1)
    return float(err[0])

def decide_charge_mode(soh_pred, fault_score, thermal_prob,
                       soh_thresh_low=0.75, fault_thresh=0.01, thermal_thresh=0.6):
    reasons = []
    if soh_pred is not None and soh_pred < soh_thresh_low * 100:
        reasons.append(f"SoH low ({soh_pred:.1f}%)")
    if fault_score is not None and fault_score >= fault_thresh:
        reasons.append(f"Fault detected (score={fault_score:.3f})")
    if thermal_prob is not None and thermal_prob >= thermal_thresh:
        reasons.append(f"Thermal risk high (p={thermal_prob:.2f})")

    if any(reasons):
        mode = "HOLD" if len(reasons) >= 2 else "SLOW"
        return mode, "; ".join(reasons)
    return "FAST", "All checks healthy"

def demo_on_test_window():
    Xagg, Xseq, splits, soh_model, fault_model, thermal_model = load_artifacts()
    te_s, te_e = splits["test"]
    if te_e - te_s <= 0:
        raise RuntimeError("No test windows available.")

    i = te_s
    x_tab = Xagg[i:i+1]
    x_seq = Xseq[i:i+1]

    soh_pred = float(soh_model.predict(x_tab)[0]) if soh_model else None
    fault_score = anomaly_score(fault_model, x_tab) if fault_model else None
    thermal_prob = float(thermal_model.predict(x_seq).ravel()[0]) if thermal_model else None

    mode, why = decide_charge_mode(soh_pred, fault_score, thermal_prob)
    print(f"⚡ Decision: {mode} | Reason: {why}")

if __name__ == "__main__":
    demo_on_test_window()
