import pandas as pd

df = pd.read_parquet("data/processed/merged.parquet")
print(df.shape)
print(df.head())
