import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from training.utils_io import load_tabular, load_splits

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def build_autoencoder(input_dim):
    inp = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation="relu")(inp)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    bottleneck = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(bottleneck)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    out = tf.keras.layers.Dense(input_dim, activation="linear")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

def main():
    # Load data
    Xagg, y, feature_cols = load_tabular()
    splits = load_splits()

    tr_s, tr_e = splits["train"]
    va_s, va_e = splits["val"]
    te_s, te_e = splits["test"]

    # Safe normal mask
    if y is not None and y.size == Xagg.shape[0] and (y == 0).any():
        normal_mask = (y == 0)
    else:
        normal_mask = np.ones(Xagg.shape[0], dtype=bool)

    # Split data
    Xtr = Xagg[tr_s:tr_e][normal_mask[tr_s:tr_e]]
    Xva = Xagg[va_s:va_e]
    Xte = Xagg[te_s:te_e]

    # Fallback if training set empty
    if Xtr.shape[0] == 0:
        print("⚠️ Warning: No normal samples in training split. Using all samples instead.")
        Xtr = Xagg[tr_s:tr_e]

    print(f"Training samples: {Xtr.shape[0]}")
    print(f"Validation samples: {Xva.shape[0]}")
    print(f"Test samples: {Xte.shape[0]}")

    # Build model
    model = build_autoencoder(Xagg.shape[1])

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "fault_autoencoder.keras"),
            save_best_only=True, monitor="val_loss"
        ),
    ]

    # Safe batch size
    batch_size = min(64, max(1, Xtr.shape[0]))
    train_ds = tf.data.Dataset.from_tensor_slices((Xtr, Xtr)).batch(batch_size).repeat()
    val_ds = tf.data.Dataset.from_tensor_slices((Xva, Xva)).batch(batch_size)

    steps_per_epoch = max(1, Xtr.shape[0] // batch_size)
    validation_steps = max(1, Xva.shape[0] // batch_size)

    # Train
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=50,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )

    # Predict and calculate reconstruction error
    recon = model.predict(Xte)
    errors = np.mean(np.square(Xte - recon), axis=1)

    # Threshold-based anomaly detection
    thresh = np.percentile(errors, 95)
    preds = (errors >= thresh).astype(int)

    metrics = {}

    if y is not None and y.size == Xagg.shape[0]:
        yte = y[te_s:te_e].astype(int)

        # If only one class exists, skip sklearn metrics and report threshold info
        if len(np.unique(yte)) < 2:
            print("⚠️ Warning: yte contains only one class. Reporting threshold-based anomaly statistics.")
            n_anomalies = np.sum(preds)
            metrics = {
                "THRESHOLD_95": float(thresh),
                "NUM_ANOMALIES_DETECTED": int(n_anomalies),
                "TOTAL_TEST_SAMPLES": int(len(yte))
            }
        else:
            prec, rec, f1, _ = precision_recall_fscore_support(yte, preds, average="binary")
            try:
                auc = roc_auc_score(yte, errors)
            except Exception:
                auc = float("nan")
            metrics = {"PREC": prec, "REC": rec, "F1": f1, "AUC": auc}

    # Save metrics
    joblib.dump(metrics, os.path.join(MODELS_DIR, "fault_autoencoder_metrics.joblib"))
    print("✅ Fault Autoencoder saved in models/")
    print(metrics)

if __name__ == "__main__":
    main()
