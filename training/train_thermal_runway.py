# training/train_thermal_autoencoder.py
import os
import numpy as np
import tensorflow as tf
import joblib
from training.utils_io import load_seq  # Make sure this returns your sequence data

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def build_autoencoder(seq_len, n_features):
    inp = tf.keras.Input(shape=(seq_len, n_features))
    # Encoder
    x = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
    x = tf.keras.layers.LSTM(32)(x)
    # Latent representation
    encoded = tf.keras.layers.Dense(16, activation="relu")(x)
    # Decoder
    x = tf.keras.layers.RepeatVector(seq_len)(encoded)
    x = tf.keras.layers.LSTM(32, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=True)(x)
    out = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(x)

    model = tf.keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model

def main():
    X, _, feat_cols = load_seq()  # ignore labels if any

    # Split into train, val, test (80-10-10)
    n_samples = X.shape[0]
    tr_end = int(0.8 * n_samples)
    va_end = int(0.9 * n_samples)

    Xtr, Xva, Xte = X[:tr_end], X[tr_end:va_end], X[va_end:]

    model = build_autoencoder(X.shape[1], X.shape[2])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "thermal_autoencoder.keras"),
            save_best_only=True,
            monitor="val_loss",
        ),
    ]

    model.fit(Xtr, Xtr, validation_data=(Xva, Xva), epochs=50, batch_size=64, callbacks=callbacks, verbose=1)

    # Compute reconstruction error on test set
    reconstructions = model.predict(Xte)
    errors = np.mean((Xte - reconstructions) ** 2, axis=(1, 2))
    threshold = np.percentile(errors, 95)  # 95th percentile threshold

    anomalies = np.where(errors > threshold)[0]

    # Save model and threshold
    print("✅ Thermal Autoencoder saved in models/")
    print(f"Threshold (95th percentile): {threshold}")
    print(f"Number of anomalies detected in test set: {len(anomalies)}")
    joblib.dump({"THRESHOLD_95": threshold, "NUM_ANOMALIES_DETECTED": len(anomalies)}, 
                os.path.join(MODELS_DIR, "thermal_autoencoder_stats.joblib"))

if __name__ == "__main__":
    main()
# # training/train_thermal_runway.py
# import os
# import numpy as np
# import joblib
# import tensorflow as tf
# from training.utils_io import load_seq, load_splits

# MODELS_DIR = "models"
# os.makedirs(MODELS_DIR, exist_ok=True)

# def build_model(seq_len, n_features):
#     inp = tf.keras.Input(shape=(seq_len, n_features))
#     x = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
#     x = tf.keras.layers.LSTM(32)(x)
#     x = tf.keras.layers.Dense(32, activation="relu")(x)
#     out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
#     model = tf.keras.Model(inp, out)
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     return model

# def main():
#     # Load data
#     X, y, feat_cols = load_seq()
#     splits = load_splits()

#     if y is None or y.size == 0:
#         print("⚠️ No labels found. Training will continue, but metrics won't be computed.")
#         compute_metrics = False
#         y = np.zeros(X.shape[0])  # dummy labels
#     else:
#         compute_metrics = True
#         y = y.astype(np.float32)

#     # Split indices
#     tr_s, tr_e = splits["train"]
#     va_s, va_e = splits["val"]
#     te_s, te_e = splits["test"]

#     Xtr, ytr = X[tr_s:tr_e], y[tr_s:tr_e]
#     Xva, yva = X[va_s:va_e], y[va_s:va_e]
#     Xte, yte = X[te_s:te_e], y[te_s:te_e]

#     # Build and train model
#     model = build_model(X.shape[1], X.shape[2])
#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
#         tf.keras.callbacks.ModelCheckpoint(os.path.join(MODELS_DIR, "thermal_lstm.keras"),
#                                            save_best_only=True, monitor="val_loss"),
#     ]

#     model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=64, callbacks=callbacks, verbose=1)

#     # Predict
#     proba = model.predict(Xte).ravel()
#     preds = (proba >= 0.5).astype(int)

#     # Compute metrics if possible
#     if compute_metrics and len(np.unique(yte)) > 1:
#         from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
#         acc = accuracy_score(yte, preds)
#         prec, rec, f1, _ = precision_recall_fscore_support(yte, preds, average="binary")
#         try:
#             auc = roc_auc_score(yte, proba)
#         except Exception:
#             auc = float("nan")
#     else:
#         print("⚠️ Metrics not computed due to missing or single-class labels.")
#         acc = prec = rec = f1 = auc = float("nan")

#     metrics = {"ACC": acc, "PREC": prec, "REC": rec, "F1": f1, "AUC": auc}
#     joblib.dump(metrics, os.path.join(MODELS_DIR, "thermal_lstm_metrics.joblib"))

#     print("✅ Thermal LSTM saved in models/")
#     print(metrics)

# if __name__ == "__main__":
#     main()
# # training/train_thermal_runaway.py
# import os
# import numpy as np
# import joblib
# import tensorflow as tf
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
# from training.utils_io import load_seq, load_splits
# from training.utils_io import load_tabular, load_splits

# MODELS_DIR = "models"
# os.makedirs(MODELS_DIR, exist_ok=True)

# def build_model(seq_len, n_features):
#     inp = tf.keras.Input(shape=(seq_len, n_features))
#     x = tf.keras.layers.LSTM(64, return_sequences=True)(inp)
#     x = tf.keras.layers.LSTM(32)(x)
#     x = tf.keras.layers.Dense(32, activation="relu")(x)
#     out = tf.keras.layers.Dense(1, activation="sigmoid")(x)
#     model = tf.keras.Model(inp, out)
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     return model

# def main():
#     X, y, feat_cols = load_seq()
#     splits = load_splits()

#     if y is None or y.size == 0:
#         raise RuntimeError("No labels found for thermal runaway. Add binary labels in preprocessing step.")

#     tr_s, tr_e = splits["train"]
#     va_s, va_e = splits["val"]
#     te_s, te_e = splits["test"]

#     Xtr, ytr = X[tr_s:tr_e], y[tr_s:tr_e]
#     Xva, yva = X[va_s:va_e], y[va_s:va_e]
#     Xte, yte = X[te_s:te_e], y[te_s:te_e]

#     model = build_model(X.shape[1], X.shape[2])
#     callbacks = [
#         tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, monitor="val_loss"),
#         tf.keras.callbacks.ModelCheckpoint(os.path.join(MODELS_DIR, "thermal_lstm.keras"),
#                                            save_best_only=True, monitor="val_loss"),
#     ]

#     model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=64, callbacks=callbacks, verbose=1)

#     proba = model.predict(Xte).ravel()
#     preds = (proba >= 0.5).astype(int)

#     acc = accuracy_score(yte, preds)
#     prec, rec, f1, _ = precision_recall_fscore_support(yte, preds, average="binary")
#     try:
#         auc = roc_auc_score(yte, proba)
#     except Exception:
#         auc = float("nan")

#     metrics = {"ACC": acc, "PREC": prec, "REC": rec, "F1": f1, "AUC": auc}
#     joblib.dump(metrics, os.path.join(MODELS_DIR, "thermal_lstm_metrics.joblib"))

#     print("✅ Thermal LSTM saved in models/")
#     print(metrics)

# if __name__ == "__main__":
#     main()
