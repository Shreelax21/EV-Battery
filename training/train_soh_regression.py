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
