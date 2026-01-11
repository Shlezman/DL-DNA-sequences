import pandas as pd

df = pd.read_parquet("unified_genomic_dataset/genomic_multiclass.parquet")

print("Original label distribution:")
print(df["label"].value_counts())

min_count = df["label"].value_counts().min()
print(f"\nDownsampling all classes to {min_count} samples")

df_balanced = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(n=min_count, random_state=42))
      .sample(frac=1, random_state=42)  # shuffle
      .reset_index(drop=True)
)

print("\nBalanced label distribution:")
print(df_balanced["label"].value_counts())


output_path = "unified_genomic_dataset/genomic_multiclass_balanced.parquet"
df_balanced.to_parquet(output_path, index=False)

print(f"\nBalanced dataset saved to: {output_path}")
