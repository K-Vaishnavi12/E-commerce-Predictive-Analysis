"""Quick EDA to understand the dataset before modeling."""
import pandas as pd
import numpy as np

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

train = pd.read_csv("train.csv")
public = pd.read_csv("public_test.csv")
private = pd.read_csv("private_test.csv")
sample = pd.read_csv("sample_submission.csv")

print("=" * 60)
print("SHAPES")
print(f"train:        {train.shape}")
print(f"public_test:  {public.shape}")
print(f"private_test: {private.shape}")
print(f"sample_sub:   {sample.shape}")

print("\n" + "=" * 60)
print("COLUMNS")
print(f"train cols:        {list(train.columns)}")
print(f"public_test cols:  {list(public.columns)}")
print(f"private_test cols: {list(private.columns)}")

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION (train)")
print(train["Converted"].value_counts(normalize=True).rename("pct"))
print(train["Converted"].value_counts().rename("count"))

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION (public_test)")
print(public["Converted"].value_counts(normalize=True).rename("pct"))

print("\n" + "=" * 60)
print("DTYPES")
print(train.dtypes)

print("\n" + "=" * 60)
print("MISSING VALUES (train)")
miss = train.isna().sum()
print(miss[miss > 0].sort_values(ascending=False))
print(f"\nrows with any missing: {train.isna().any(axis=1).sum()} / {len(train)}")

print("\n" + "=" * 60)
print("MISSING VALUES (public_test)")
miss = public.isna().sum()
print(miss[miss > 0].sort_values(ascending=False))

print("\n" + "=" * 60)
print("MISSING VALUES (private_test)")
miss = private.isna().sum()
print(miss[miss > 0].sort_values(ascending=False))

print("\n" + "=" * 60)
print("NUMERIC SUMMARY (train)")
print(train.describe().T)

print("\n" + "=" * 60)
print("CATEGORICAL VALUE COUNTS (train)")
for col in ["City_Tier", "Device_Type", "Traffic_Source", "Discount_Seen"]:
    print(f"\n--- {col} ---")
    print(train[col].value_counts(dropna=False))

print("\n" + "=" * 60)
print("Browser_Version & Campaign_Code cardinality")
for col in ["Browser_Version", "Campaign_Code"]:
    print(f"{col}: nunique={train[col].nunique()} min={train[col].min()} max={train[col].max()}")

print("\n" + "=" * 60)
print("USER_ID overlap check")
tr_ids = set(train["User_ID"])
pub_ids = set(public["User_ID"])
prv_ids = set(private["User_ID"])
print(f"train IDs range: {train['User_ID'].min()} - {train['User_ID'].max()}")
print(f"public IDs range: {public['User_ID'].min()} - {public['User_ID'].max()}")
print(f"private IDs range: {private['User_ID'].min()} - {private['User_ID'].max()}")
print(f"train cap public: {len(tr_ids & pub_ids)}")
print(f"train cap private: {len(tr_ids & prv_ids)}")
print(f"public cap private: {len(pub_ids & prv_ids)}")

print("\n" + "=" * 60)
print("CONVERSION RATE BY CATEGORICAL")
for col in ["City_Tier", "Device_Type", "Traffic_Source", "Discount_Seen"]:
    print(f"\n--- {col} ---")
    grp = train.groupby(col)["Converted"].agg(["mean", "count"])
    print(grp)

print("\n" + "=" * 60)
print("SAMPLE SUBMISSION head/tail")
print(sample.head(3))
print("...")
print(sample.tail(3))
print(f"sum of Converted in sample: {sample['Converted'].sum()}")
