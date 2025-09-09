import os
import joblib
import numpy as np
import tensorflow as tf
from training.utils_io import load_tabular

MODELS_DIR = "models"

def load_model_and_metrics():
    """Load the trained autoencoder and metrics."""
    model_path = os.path.join(MODELS_DIR, "fault_autoencoder.keras")
    metrics_path = os.path.join(MODELS_DIR, "fault_autoencoder_metrics.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics not found at {metrics_path}")

    model = tf.keras.models.load_model(model_path)
    metrics = joblib.load(metrics_path)
    return model, metrics

def detect_anomalies(X, model, threshold=None):
    """Detect anomalies in dataset X using the trained model."""
    recon = model.predict(X)
    errors = np.mean(np.square(X - recon), axis=1)

    if threshold is None:
        # fallback to 95th percentile
        threshold = np.percentile(errors, 95)

    anomalies = errors >= threshold
    num_anomalies = np.sum(anomalies)

    return anomalies, num_anomalies, errors, threshold

def main():
    # Load dataset
    Xagg, y, feature_cols = load_tabular()

    # Load trained model
    model, metrics = load_model_and_metrics()

    # Use threshold from training metrics if available
    threshold = metrics.get("THRESHOLD_95", None)

    # Detect anomalies on the entire dataset (or split if desired)
    anomalies, num_anomalies, errors, threshold_used = detect_anomalies(Xagg, model, threshold)

    print("✅ Anomaly detection completed")
    print(f"Threshold used: {threshold_used}")
    print(f"Number of anomalies detected: {num_anomalies}")
    print(f"Total samples: {Xagg.shape[0]}")
    print(f"Anomaly indices: {np.where(anomalies)[0]}")

    # Optional: return errors array for further analysis
    return errors, anomalies

if __name__ == "__main__":
    main()
