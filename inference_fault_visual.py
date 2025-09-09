import os
import numpy as np
import tensorflow as tf
from training.utils_io import load_fault_data  # you should have a loader function for fault data

MODELS_DIR = "models"
FAULT_MODEL_PATH = os.path.join(MODELS_DIR, "fault_autoencoder.keras")

# Load the trained fault autoencoder
model = tf.keras.models.load_model(FAULT_MODEL_PATH)
print("✅ Fault Autoencoder loaded successfully.")

# Load test data
X_test = load_fault_data(split="test")  # returns your fault test dataset
print(f"Test samples: {X_test.shape[0]}")

# Get reconstruction errors
reconstructions = model.predict(X_test)
errors = np.mean(np.square(X_test - reconstructions), axis=1)

# Threshold using 95th percentile
threshold = np.percentile(errors, 95)
anomalies = np.where(errors > threshold)[0]

print("✅ Fault Autoencoder inference completed")
print(f"Threshold (95th percentile): {threshold}")
print(f"Number of anomalies detected in test set: {len(anomalies)}")
print(f"Anomaly indices: {anomalies.tolist()}")
