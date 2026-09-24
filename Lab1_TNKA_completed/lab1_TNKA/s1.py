from pathlib import Path
import pandas as pd

BASE = Path(__file__).parent
DATA_PATH = BASE / "headlines_train.csv"
OUT_DIR = BASE / "headlines"
OUT_DIR.mkdir(exist_ok=True)

SAMPLE_SEED = 42
SAMPLE_SIZE = 1000
SELECT_SEED = 7

df = pd.read_csv(DATA_PATH)
n_original = len(df)

df["orig_row"] = df.index
df = df[["orig_row", "text"]].copy()

df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
n_after_empty = len(df)

df = df.drop_duplicates(subset="text")
n_after_dedup = len(df)

print(f"Original rows: {n_original}")
print(f"After removing empty text: {n_after_empty}")
print(f"After removing duplicates: {n_after_dedup}")

n = min(SAMPLE_SIZE, len(df))

sample_df = df.sample(
    n=n,
    random_state=SAMPLE_SEED
).reset_index(drop=True)

sample_df.to_csv(
    OUT_DIR / "sample_collection.csv",
    index=False,
    encoding="utf-8-sig"
)

manual_20 = sample_df.sample(
    n=20,
    random_state=SELECT_SEED
).reset_index(drop=True)

manual_20.insert(
    0,
    "headline_id",
    [f"H{i+1:02d}" for i in range(len(manual_20))]
)

manual_20.to_csv(
    OUT_DIR / "20head.csv",
    index=False,
    encoding="utf-8-sig"
)

print(f"\nSample size: {len(sample_df)}")
print(f"Sample seed: {SAMPLE_SEED}")
print(f"20-headline seed: {SELECT_SEED}")

print("\n20 selected headlines:")
print(manual_20.to_string(index=False))
