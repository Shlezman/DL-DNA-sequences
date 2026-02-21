import pandas as pd

df = pd.read_parquet("/tf/unified_DNA_dataset/DNA_multiclass.parquet")

print("Sequence length statistics:")
print(df["sequence"].str.len().describe())

print("\nOriginal label distribution:")
print(df["label"].value_counts())

min_count = df["label"].value_counts().min()
print(f"\nDownsampling all classes to {min_count} samples")

# Sample from each group separately then concatenate
sampled_groups = []
for label in df["label"].unique():
    group = df[df["label"] == label]
    sampled = group.sample(n=min_count, random_state=42)
    sampled_groups.append(sampled)

df_balanced = pd.concat(sampled_groups, ignore_index=True)
df_balanced = df_balanced.sample(
    frac=1, random_state=42).reset_index(drop=True)  # shuffle
