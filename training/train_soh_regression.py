# training/train_soh_regressor.py
import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
#utils import

from training.utils_io import load_tabular, load_splits

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    Xagg, y, feature_cols = load_tabular()
    splits = load_splits()

    if y is None or y.size == 0:
        raise RuntimeError("No regression labels found (y empty). Ensure 'soh_' or SOC present.")

    tr_s, tr_e = splits["train"]
    va_s, va_e = splits["val"]
    te_s, te_e = splits["test"]

    Xtr, ytr = Xagg[tr_s:tr_e], y[tr_s:tr_e]
    Xva, yva = Xagg[va_s:va_e], y[va_s:va_e]
    Xte, yte = Xagg[te_s:te_e], y[te_s:te_e]

    model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)

    def metrics(X, y):
        pred = model.predict(X)
        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(((pred - y) ** 2).mean())
        r2 = r2_score(y, pred)
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    allm = {"train": metrics(Xtr, ytr), "val": metrics(Xva, yva), "test": metrics(Xte, yte)}

    joblib.dump(model, os.path.join(MODELS_DIR, "soh_regressor.joblib"))
    joblib.dump(allm, os.path.join(MODELS_DIR, "soh_regressor_metrics.joblib"))

    print("✅ SoH regressor saved in models/")
    print(allm)

if __name__ == "__main__":
    main()
# # training/train_soh_regressor.py
# import os
# import numpy as np
# import joblib
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.model_selection import train_test_split

# # utils import
# from training.utils_io import load_tabular, load_splits

# MODELS_DIR = "models"
# os.makedirs(MODELS_DIR, exist_ok=True)

# # --- SoH status thresholds ---
# def soh_status(val):
#     if val >= 90:
#         return "Healthy ✅"
#     elif val >= 80:
#         return "Moderate ⚠️"
#     else:
#         return "Unhealthy ❌"

# def main():
#     # Load data
#     Xagg, y, feature_cols = load_tabular()
#     splits = load_splits()

#     if y is None or y.size == 0:
#         raise RuntimeError("No regression labels found (y empty). Ensure 'soh_' or SOC present.")

#     tr_s, tr_e = splits["train"]
#     va_s, va_e = splits["val"]
#     te_s, te_e = splits["test"]

#     Xtr, ytr = Xagg[tr_s:tr_e], y[tr_s:tr_e]
#     Xva, yva = Xagg[va_s:va_e], y[va_s:va_e]
#     Xte, yte = Xagg[te_s:te_e], y[te_s:te_e]

#     # --- Check target distribution ---
#     print("Training target range:", ytr.min(), ytr.max())
#     print("Validation target range:", yva.min(), yva.max())
#     print("Test target range:", yte.min(), yte.max())

#     # Train Random Forest Regressor with overfitting prevention
#     model = RandomForestRegressor(
#         n_estimators=200,
#         max_depth=10,             # limits tree depth
#         min_samples_leaf=5,       # avoids leaves with few samples
#         max_features='sqrt',      # reduces correlation
#         random_state=42,
#         n_jobs=-1
#     )
#     model.fit(Xtr, ytr)

#     # Evaluation metrics function
#     def metrics(X, y):
#         pred = model.predict(X)
#         mae = mean_absolute_error(y, pred)
#         rmse = np.sqrt(((pred - y) ** 2).mean())
#         r2 = r2_score(y, pred)
#         return {"MAE": mae, "RMSE": rmse, "R2": r2}

#     # Collect metrics for train, validation, and test
#     allm = {
#         "train": metrics(Xtr, ytr),
#         "val": metrics(Xva, yva),
#         "test": metrics(Xte, yte),
#     }

#     # Save model + metrics
#     joblib.dump(model, os.path.join(MODELS_DIR, "soh_regressor.joblib"))
#     joblib.dump(allm, os.path.join(MODELS_DIR, "soh_regressor_metrics.joblib"))

#     print("✅ SoH regressor saved in models/")
#     print("📈 Model Performance Metrics (MAE, RMSE, R²):")
#     print(allm)

#     # --- Predict & Display Test Results ---
#     print("\n📊 Battery SoH Predictions on Test Set:")
#     y_pred = model.predict(Xte)

#     results = []
#     for i, (true_val, pred_val) in enumerate(zip(yte, y_pred)):
#         status = soh_status(pred_val)
#         results.append({
#             "Sample": i + 1,
#             "True_SoH(%)": round(true_val, 2),
#             "Predicted_SoH(%)": round(pred_val, 2),
#             "Status": status
#         })
#         print(f"Sample {i+1}: True SoH = {true_val:.2f}% | Predicted SoH = {pred_val:.2f}% --> {status}")

#     # Save predictions to CSV for reporting
#     results_df = pd.DataFrame(results)
#     csv_path = os.path.join(MODELS_DIR, "soh_predictions.csv")
#     results_df.to_csv(csv_path, index=False)
#     print(f"\n📂 Predictions saved to: {csv_path}")

#     # --- Optional: Status distribution ---
#     dist = results_df["Status"].value_counts()
#     print("\n📊 Predicted SoH Status Distribution:")
#     print(dist)

# if __name__ == "__main__":
#     main()

# training/train_soh_regressor.py
# import os
# import numpy as np
# import joblib
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score

# # utils import
# from training.utils_io import load_tabular, load_splits

# MODELS_DIR = "models"
# os.makedirs(MODELS_DIR, exist_ok=True)

# # --- SoH status thresholds ---
# def soh_status(val):
#     if val >= 90:
#         return "Healthy ✅"
#     elif val >= 80:
#         return "Moderate ⚠️"
#     else:
#         return "Unhealthy ❌"

# def main():
#     # Load data
#     Xagg, y, feature_cols = load_tabular()
#     splits = load_splits()

#     if y is None or y.size == 0:
#         raise RuntimeError("No regression labels found (y empty). Ensure 'soh_' or SOC present.")

#     tr_s, tr_e = splits["train"]
#     va_s, va_e = splits["val"]
#     te_s, te_e = splits["test"]

#     Xtr, ytr = Xagg[tr_s:tr_e], y[tr_s:tr_e]
#     Xva, yva = Xagg[va_s:va_e], y[va_s:va_e]
#     Xte, yte = Xagg[te_s:te_e], y[te_s:te_e]

#     # Train Random Forest Regressor
#     model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
#     model.fit(Xtr, ytr)

#     # Evaluation metrics function
#     def metrics(X, y):
#         pred = model.predict(X)
#         mae = mean_absolute_error(y, pred)
#         rmse = np.sqrt(((pred - y) ** 2).mean())
#         r2 = r2_score(y, pred)
#         return {"MAE": mae, "RMSE": rmse, "R2": r2}

#     # Collect metrics for train, validation, and test
#     allm = {
#         "train": metrics(Xtr, ytr),
#         "val": metrics(Xva, yva),
#         "test": metrics(Xte, yte),
#     }

#     # Save model + metrics
#     joblib.dump(model, os.path.join(MODELS_DIR, "soh_regressor.joblib"))
#     joblib.dump(allm, os.path.join(MODELS_DIR, "soh_regressor_metrics.joblib"))

#     print("✅ SoH regressor saved in models/")
#     print("📈 Model Performance Metrics (MAE, RMSE, R²):")
#     print(allm)

#     # --- Predict & Display Test Results ---
#     print("\n📊 Battery SoH Predictions on Test Set:")
#     y_pred = model.predict(Xte)

#     results = []
#     for i, (true_val, pred_val) in enumerate(zip(yte, y_pred)):
#         status = soh_status(pred_val)
#         results.append({
#             "Sample": i + 1,
#             "True_SoH(%)": round(true_val, 2),
#             "Predicted_SoH(%)": round(pred_val, 2),
#             "Status": status
#         })
#         print(f"Sample {i+1}: True SoH = {true_val:.2f}% | Predicted SoH = {pred_val:.2f}% --> {status}")

#     # Save predictions to CSV for reporting
#     results_df = pd.DataFrame(results)
#     csv_path = os.path.join(MODELS_DIR, "soh_predictions.csv")
#     results_df.to_csv(csv_path, index=False)
#     print(f"\n📂 Predictions saved to: {csv_path}")

#     # --- Optional: Status distribution ---
#     dist = results_df["Status"].value_counts()
#     print("\n📊 Predicted SoH Status Distribution:")
#     print(dist)

# if __name__ == "__main__":
#     main()
# # training/train_soh_regressor.py
# import os
# import numpy as np
# import joblib
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score

# # utils import
# from training.utils_io import load_tabular, load_splits


# MODELS_DIR = "models"
# os.makedirs(MODELS_DIR, exist_ok=True)

# # Threshold for deciding health status
# HEALTHY_THRESHOLD = 80.0  # % SoH

# def main():
#     # Load data
#     Xagg, y, feature_cols = load_tabular()
#     splits = load_splits()

#     if y is None or y.size == 0:
#         raise RuntimeError("No regression labels found (y empty). Ensure 'soh_' or SOC present.")

#     tr_s, tr_e = splits["train"]
#     va_s, va_e = splits["val"]
#     te_s, te_e = splits["test"]

#     Xtr, ytr = Xagg[tr_s:tr_e], y[tr_s:tr_e]
#     Xva, yva = Xagg[va_s:va_e], y[va_s:va_e]
#     Xte, yte = Xagg[te_s:te_e], y[te_s:te_e]

#     # Train Random Forest Regressor
#     model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
#     model.fit(Xtr, ytr)

#     # Evaluation metrics function
#     def metrics(X, y):
#         pred = model.predict(X)
#         mae = mean_absolute_error(y, pred)
#         rmse = np.sqrt(((pred - y) ** 2).mean())
#         r2 = r2_score(y, pred)
#         return {"MAE": mae, "RMSE": rmse, "R2": r2}

#     # Collect metrics for train, validation, and test
#     allm = {
#         "train": metrics(Xtr, ytr),
#         "val": metrics(Xva, yva),
#         "test": metrics(Xte, yte),
#     }

#     # Save model + metrics
#     joblib.dump(model, os.path.join(MODELS_DIR, "soh_regressor.joblib"))
#     joblib.dump(allm, os.path.join(MODELS_DIR, "soh_regressor_metrics.joblib"))

#     # 🔹 Print the OLD metrics output
#     print("✅ SoH regressor saved in models/")
#     print("📈 Model Performance Metrics (MAE, RMSE, R²):")
#     print(allm)

#     # --- NEW PART: Predict & Display Test Results ---
#     print("\n📊 Battery SoH Predictions on Test Set:")
#     y_pred = model.predict(Xte)

#     results = []
#     for i, (true_val, pred_val) in enumerate(zip(yte, y_pred)):
#         status = "Healthy ✅" if pred_val >= HEALTHY_THRESHOLD else "Unhealthy ⚠️"
#         results.append({
#             "Sample": i+1,
#             "True_SoH(%)": round(true_val, 2),
#             "Predicted_SoH(%)": round(pred_val, 2),
#             "Status": status
#         })
#         print(f"Sample {i+1}: True SoH = {true_val:.2f}% | Predicted SoH = {pred_val:.2f}% --> {status}")

#     # Save predictions to CSV for reporting
#     results_df = pd.DataFrame(results)
#     csv_path = os.path.join(MODELS_DIR, "soh_predictions.csv")
#     results_df.to_csv(csv_path, index=False)
#     print(f"\n📂 Predictions saved to: {csv_path}")


# if __name__ == "__main__":
#     main()
# # training/train_soh_regressor.py
# import os
# import numpy as np
# import joblib
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, r2_score
# #utils import

# from training.utils_io import load_tabular, load_splits


# MODELS_DIR = "models"
# os.makedirs(MODELS_DIR, exist_ok=True)

# def main():
#     Xagg, y, feature_cols = load_tabular()
#     splits = load_splits()

#     if y is None or y.size == 0:
#         raise RuntimeError("No regression labels found (y empty). Ensure 'soh_' or SOC present.")

#     tr_s, tr_e = splits["train"]
#     va_s, va_e = splits["val"]
#     te_s, te_e = splits["test"]

#     Xtr, ytr = Xagg[tr_s:tr_e], y[tr_s:tr_e]
#     Xva, yva = Xagg[va_s:va_e], y[va_s:va_e]
#     Xte, yte = Xagg[te_s:te_e], y[te_s:te_e]

#     model = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
#     model.fit(Xtr, ytr)

#     def metrics(X, y):
#         pred = model.predict(X)
#         mae = mean_absolute_error(y, pred)
#         rmse = np.sqrt(((pred - y) ** 2).mean())
#         r2 = r2_score(y, pred)
#         return {"MAE": mae, "RMSE": rmse, "R2": r2}

#     allm = {"train": metrics(Xtr, ytr), "val": metrics(Xva, yva), "test": metrics(Xte, yte)}

#     joblib.dump(model, os.path.join(MODELS_DIR, "soh_regressor.joblib"))
#     joblib.dump(allm, os.path.join(MODELS_DIR, "soh_regressor_metrics.joblib"))

#     print("✅ SoH regressor saved in models/")
#     print(allm)

# if __name__ == "__main__":
#     main()
