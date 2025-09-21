import pandas as pd

df = pd.read_parquet("data/processed/merged_enhanced.parquet")
print(df.columns.tolist())
# import pandas as pd

# df = pd.read_parquet("data/processed/merged_enhanced.parquet")
# print(df["soh_"].describe())

# import pandas as pd
# df = pd.read_parquet("data/processed/merged_enhanced.parquet")
# print(df["soh_"].describe())
# print(df["soh_"].unique()[:20])