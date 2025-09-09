# training/train_all.py
"""
Runs all training scripts (SoH regressor, Fault Autoencoder, Thermal LSTM)
and prints a summary report.

Usage:
    python training/train_all.py
"""

import os
import subprocess
import joblib

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

SCRIPTS = [
    "training/train_soh_regressor.py",
    "training/train_fault_autoencoder.py",
    "training/train_thermal_runaway.py",
]

def run_script(script):
    print(f"\n🚀 Running {script} ...")
    result = subprocess.run(["python", script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("⚠️ ERRORS/WARNINGS:")
        print(result.stderr)

def load_metrics():
    metrics_files = {
        "SoH Regressor": "soh_regressor_metrics.joblib",
        "Fault Autoencoder": "fault_autoencoder_metrics.joblib",
        "Thermal LSTM": "thermal_lstm_metrics.joblib",
    }
    summary = {}
    for name, f in metrics_files.items():
        path = os.path.join(MODELS_DIR, f)
        if os.path.exists(path):
            try:
                summary[name] = joblib.load(path)
            except Exception as e:
                summary[name] = f"Could not load ({e})"
        else:
            summary[name] = "Not trained"
    return summary

def main():
    for script in SCRIPTS:
        run_script(script)

    print("\n📊 === Training Summary ===")
    summary = load_metrics()
    for model_name, metrics in summary.items():
        print(f"\n🔹 {model_name}")
        print(metrics)

if __name__ == "__main__":
    main()

