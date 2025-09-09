# data_scripts/02_make_windows.py
"""
Creates sliding windows from data/processed/merged.parquet.

Outputs (data/processed/windows/):
 - windows_seq.npz      -> X (float32) shape (N, W, F), y (float32) shape (N,) or None
 - windows_tabular.npz  -> aggregated features for tree-based baselines (N, Fagg)
 - feature_cols.npy     -> list of feature column names (object)
 - window_starts.npy    -> list of window start timestamps (strings)
 - scaler.joblib        -> fitted scaler (StandardScaler)
 
Run: python data_scripts/02_make_windows.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ---- CONFIG ----
# ---- CONFIG ----
INPATH = "data/processed/merged_enhanced.parquet"   # updated file path
OUTDIR = "data/processed/windows"
os.makedirs(OUTDIR, exist_ok=True)

WINDOW_SECONDS = 120   # length of each window (seconds)
STRIDE_SECONDS = 10    # stride (seconds) between windows
SAMPLE_FREQ = "1s"     # assumed sampling frequency in merged.parquet
DTYPE = np.float32     # use float32 to save memory

# INPATH = "data/processed/merged.parquet"
# OUTDIR = "data/processed/windows"
# os.makedirs(OUTDIR, exist_ok=True)

# WINDOW_SECONDS = 120   # length of each window (seconds)
# STRIDE_SECONDS = 10    # stride (seconds) between windows
# SAMPLE_FREQ = "1s"     # assumed sampling frequency in merged.parquet
# DTYPE = np.float32     # use float32 to save memory

# ---- helpers ----
def ensure_index_freq(df, freq="1s"):
    # Ensure index is datetime and has at least this frequency
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")
    # if no freq, try to set it (we won't reindex to new freq here)
    if df.index.freq is None:
        try:
            df = df.asfreq(freq)  # will insert NaNs for missing timestamps
        except Exception:
            # fallback: leave as-is (we assume it's already regular)
            pass
    return df

def make_windows_from_array(arr, window, stride):
    """
    arr: numpy array (T, F)
    returns X shape (n_windows, window, F) and indices of window starts
    """
    T = arr.shape[0]
    starts = list(range(0, T - window + 1, stride))
    n = len(starts)
    if n == 0:
        return np.empty((0, window, arr.shape[1]), dtype=arr.dtype), []
    X = np.empty((n, window, arr.shape[1]), dtype=arr.dtype)
    for i, s in enumerate(starts):
        X[i] = arr[s:s+window]
    return X, starts

def aggregate_window_stats(X):
    """
    X: (N, W, F) -> returns aggregated features per window:
    mean, std, min, max, last, trend (last-first) per feature => 6*F dims
    """
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    last = X[:, -1, :]
    trend = X[:, -1, :] - X[:, 0, :]
    # concatenate
    return np.concatenate([mean, std, mn, mx, last, trend], axis=1)

# ---- main ----
def main():
    print("Loading:", INPATH)
    df = pd.read_parquet(INPATH)
    print("Loaded shape:", df.shape)
    df = ensure_index_freq(df, SAMPLE_FREQ)
    
    # choose numeric features
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print("Numeric feature columns:", feature_cols)
    if len(feature_cols) == 0:
        raise RuntimeError("No numeric features found in merged.parquet")

    # TARGET selection: prefer 'soh_' if present, else try 'battery_soc' or telemetry soc
    target_col = None
    for cand in ["soh_", "battery_soc", "telemetry_soc", "soc"]:
        if cand in df.columns:
            target_col = cand
            break
    print("Detected target column:", target_col)

    # Option: drop target from feature columns so it is not used as input feature
    feature_cols_no_target = [c for c in feature_cols if c != target_col]
    print("Using features (no target):", feature_cols_no_target[:20])

    # fit scaler on entire feature set (we'll save it - ideally fit on train only later)
    scaler = StandardScaler()
    scaler.fit(df[feature_cols_no_target].fillna(method="ffill").fillna(0).values)
    joblib.dump(scaler, os.path.join(OUTDIR, "scaler.joblib"))
    print("Saved scaler to scaler.joblib")

    # convert df to numpy (float32) after filling small gaps
    arr = df[feature_cols_no_target].fillna(method="ffill").fillna(0).astype(np.float32).values
    # scale features (use transform)
    arr_scaled = scaler.transform(arr).astype(DTYPE)

    window = WINDOW_SECONDS
    stride = STRIDE_SECONDS

    X, starts = make_windows_from_array(arr_scaled, window=window, stride=stride)
    print("Windows created:", X.shape)

    # prepare y if target present (value at the end of each window)
    y = None
    if target_col is not None:
        # column might have NaNs — fill
        tgt_arr = df[target_col].fillna(method="ffill").fillna(0).values.astype(np.float32)
        # label is the value at the last row of window
        ys = []
        for s in starts:
            ys.append(tgt_arr[s + window - 1])
        y = np.array(ys, dtype=np.float32)
        print("y shape:", y.shape)

    # Save sequence windows
    seq_out = os.path.join(OUTDIR, "windows_seq.npz")
    # Convert X to float32 (already) to save space
    np.savez_compressed(seq_out, X=X, y=y if y is not None else np.array([]), feature_cols=np.array(feature_cols_no_target, dtype=object))
    print("Saved seq windows to", seq_out)

    # Save aggregated tabular features (for RandomForest baseline)
    if X.shape[0] > 0:
        Xagg = aggregate_window_stats(X)  # (N, 6*F)
        tab_out = os.path.join(OUTDIR, "windows_tabular.npz")
        np.savez_compressed(tab_out, Xagg=Xagg, y=y if y is not None else np.array([]), feature_cols=np.array(feature_cols_no_target, dtype=object))
        print("Saved tabular aggregated windows to", tab_out)
    
    # Save window start timestamps (strings) to help with time-split later
    ts_index = np.array([str(df.index[s]) for s in starts], dtype=object)
    np.save(os.path.join(OUTDIR, "window_starts.npy"), ts_index)
    print("Saved window start timestamps:", os.path.join(OUTDIR, "window_starts.npy"))

    # Save simple split indexes (time-based): train = first 70%, val 15%, test 15%
    N = X.shape[0]
    if N > 0:
        train_end = int(0.7 * N)
        val_end = int(0.85 * N)
        splits = {"train": [0, train_end], "val": [train_end, val_end], "test": [val_end, N]}
        joblib.dump(splits, os.path.join(OUTDIR, "splits.joblib"))
        print("Saved splits:", splits)

    print("All done. Files in:", OUTDIR)

if __name__ == "__main__":
    main()
