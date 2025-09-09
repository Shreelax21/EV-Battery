import numpy as np
d = np.load("data/processed/windows/windows_seq.npz", allow_pickle=True)
X = d["X"]     # shape (N, W, F)
y = d["y"]     # shape (N,) or empty
fcols = d["feature_cols"]
print("X shape:", X.shape)
print("y shape:", y.shape)
print("features:", fcols[:20])
# load tabular:
t = np.load("data/processed/windows/windows_tabular.npz", allow_pickle=True)
print("Xagg shape:", t["Xagg"].shape)
