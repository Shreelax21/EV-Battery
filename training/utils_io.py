# training/utils_io.py
import os
import joblib
import numpy as np

WIN_DIR = os.path.join("data", "processed", "windows")

def load_tabular():
    path = os.path.join(WIN_DIR, "windows_tabular.npz")
    data = np.load(path, allow_pickle=True)
    Xagg = data["Xagg"]
    y = data["y"]
    feature_cols = data["feature_cols"].tolist()
    return Xagg, y, feature_cols

def load_seq():
    path = os.path.join(WIN_DIR, "windows_seq.npz")
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    feature_cols = data["feature_cols"].tolist()
    return X, y, feature_cols

def load_splits():
    path = os.path.join(WIN_DIR, "splits.joblib")
    return joblib.load(path)  # dict: {"train": [s,e], "val": [s,e], "test": [s,e]}

# ================= Fault Autoencoder Loader =================
def load_fault_data(split="test"):
    """
    Load fault autoencoder dataset.
    split: "train", "val", "test"
    """
    path = os.path.join(WIN_DIR, "fault_bms_dataset.npz")  # make sure this file exists
    data = np.load(path, allow_pickle=True)
    X = data["X"]
    y = data["y"] if "y" in data else None

    # Load splits
    splits = load_splits()
    s, e = splits[split]
    X_split = X[s:e]
    y_split = y[s:e] if y is not None else None

    return X_split, y_split
# # training/utils_io.py
# import os
# import joblib
# import numpy as np

# WIN_DIR = os.path.join("data", "processed", "windows")

# def load_tabular():
#     path = os.path.join(WIN_DIR, "windows_tabular.npz")
#     data = np.load(path, allow_pickle=True)
#     Xagg = data["Xagg"]
#     y = data["y"]
#     feature_cols = data["feature_cols"].tolist()
#     return Xagg, y, feature_cols

# def load_seq():
#     path = os.path.join(WIN_DIR, "windows_seq.npz")
#     data = np.load(path, allow_pickle=True)
#     X = data["X"]
#     y = data["y"]
#     feature_cols = data["feature_cols"].tolist()
#     return X, y, feature_cols

# def load_splits():
#     path = os.path.join(WIN_DIR, "splits.joblib")
#     return joblib.load(path)  # dict: {"train": [s,e], "val": [s,e], "test": [s,e]}
