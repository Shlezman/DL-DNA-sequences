from datasets import load_dataset, concatenate_datasets
from pathlib import Path

DATASETS_CONFIG = [
    {
        "hf_name": "katarinagresova/Genomic_Benchmarks_human_enhancers_ensembl",
        "label": "ENHANCER",
    },
    {
        "hf_name": "katarinagresova/Genomic_Benchmarks_human_nontata_promoters",
        "label": "PROMOTER",
    },
    {
        "hf_name": "katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs",
        "label": "INTERGENIC",
        "filter_label": 0  # Filter for intergenic sequences only
    }
]

LABEL2ID = {
    "ENHANCER": 0,
    "PROMOTER": 1,
    "INTERGENIC": 2
}

def find_sequence_column(column_names):
    for candidate in ["sequence", "seq", "dna"]:
        if candidate in column_names:
            return candidate
    raise ValueError(f"No sequence column found in {column_names}")

def load_and_label_dataset(hf_name, label_str, filter_label=None):
    ds = load_dataset(hf_name)
    
    if isinstance(ds, dict):
        ds = concatenate_datasets(list(ds.values()))
    
    # Filter for specific label if specified (e.g., intergenic only)
    if filter_label is not None:
        ds = ds.filter(lambda x: x.get("label") == filter_label)

    seq_col = find_sequence_column(ds.column_names)
    label_id = LABEL2ID[label_str]

    ds = ds.map(
        lambda x: {
            "sequence": x[seq_col],
            "label": label_id,
            "label_name": label_str,
            "source_dataset": hf_name,
        },
        remove_columns=ds.column_names,
    )

    return ds

all_datasets = []

for cfg in DATASETS_CONFIG:
    print(f"Loading {cfg['hf_name']} ...")
    ds = load_and_label_dataset(
        cfg["hf_name"], 
        cfg["label"],
        filter_label=cfg.get("filter_label")
    )
    print(f"  Loaded {len(ds)} samples for {cfg['label']}")
    all_datasets.append(ds)

unified_dataset = concatenate_datasets(all_datasets)

print("\nUnified dataset created:")
print(unified_dataset)

output_dir = Path("..") / "unified_DNA_dataset"
output_dir.mkdir(exist_ok=True)

parquet_path = output_dir / "DNA_multiclass.parquet"
unified_dataset.to_parquet(parquet_path)

print(f"\nDataset saved to: {parquet_path.resolve()}")