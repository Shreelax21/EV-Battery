# inference_thermal_autoencoder.py

import os
import numpy as np
import tensorflow as tf
import joblib
from training.utils_io import load_seq, load_splits  # Adjust imports if needed

MODELS_DIR = "models"

def main():
    # Load the trained autoencoder
    model_path = os.path.join(MODELS_DIR, "thermal_autoencoder.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")

    model = tf.keras.models.load_model(model_path)
    print("✅ Thermal Autoencoder loaded successfully.")

    # Load the dataset
    X, y, feat_cols = load_seq()  # y may be None for unsupervised
    splits = load_splits()
    te_s, te_e = splits["test"]
    Xte = X[te_s:te_e]

    # Predict reconstruction
    X_pred = model.predict(Xte)
    errors = np.mean(np.square(Xte - X_pred), axis=(1, 2))  # MSE per sequence

    # Compute threshold (95th percentile)
    threshold = np.percentile(errors, 95)

    # Detect anomalies
    anomalies = np.where(errors > threshold)[0]

    # Print results
    print("✅ Thermal Autoencoder inference completed")
    print(f"Threshold (95th percentile): {threshold}")
    print(f"Number of anomalies detected in test set: {len(anomalies)}")
    print(f"Anomaly indices: {anomalies}")

    # Optional: save results
    joblib.dump({"errors": errors, "threshold": threshold, "anomalies": anomalies},
                os.path.join(MODELS_DIR, "thermal_autoencoder_inference.joblib"))

if __name__ == "__main__":
    main()
