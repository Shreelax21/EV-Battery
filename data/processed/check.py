import pandas as pd

df = pd.read_parquet("data/processed/merged_enhanced.parquet")
print(df.shape)
print(df.head())
