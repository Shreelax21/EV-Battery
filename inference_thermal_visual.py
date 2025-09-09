import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from training.utils_io import load_seq, load_splits

# Paths
MODEL_PATH = "models/thermal_autoencoder.keras"  # your saved autoencoder

# Load model
model = load_model(MODEL_PATH)
print("✅ Thermal Autoencoder loaded successfully.")

# Load test data
X, y, _ = load_seq()
splits = load_splits()
_, te_e = splits["test"]
Xte = X[splits["test"][0]:te_e]

# Predict reconstruction
X_pred = model.predict(Xte)

# Compute reconstruction error per sample
recon_error = np.mean(np.square(Xte - X_pred), axis=(1, 2))

# Threshold (95th percentile)
threshold = np.percentile(recon_error, 95)
print("Threshold (95th percentile):", threshold)

# Detect anomalies
anomalies = np.where(recon_error > threshold)[0]
print("Anomaly indices:", anomalies)

# Plot reconstruction error with anomalies
plt.figure(figsize=(12,6))
plt.plot(recon_error, label="Reconstruction Error")
plt.axhline(y=threshold, color='r', linestyle='--', label="Threshold (95th percentile)")
plt.scatter(anomalies, recon_error[anomalies], color='red', label="Anomalies")
plt.xlabel("Test Sample Index")
plt.ylabel("Reconstruction Error")
plt.title("Thermal Autoencoder Anomaly Detection")
plt.legend()
plt.show()

# Optional: plot actual vs reconstructed for anomalies
for idx in anomalies:
    plt.figure(figsize=(10,4))
    plt.plot(Xte[idx].ravel(), label="Actual")
    plt.plot(X_pred[idx].ravel(), label="Reconstructed", linestyle='--')
    plt.title(f"Sample {idx} - Anomaly")
    plt.xlabel("Time Step / Feature Index")
    plt.ylabel("Thermal Reading")
    plt.legend()
    plt.show()
