from datasets import load_dataset, concatenate_datasets, Dataset
from pathlib import Path
import random

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

MIN_SEQ_LENGTH = 2
MAX_SEQ_LENGTH = 512

def find_sequence_column(column_names):
    for candidate in ["sequence", "seq", "dna"]:
        if candidate in column_names:
            return candidate
    raise ValueError(f"No sequence column found in {column_names}")

def resize_intergenic_sequences(dataset, label_str):
    """Cut and concatenate intergenic sequences to create variable lengths between 2-512."""
    if label_str != "INTERGENIC":
        return dataset
    
    print(f"  Resizing {len(dataset)} intergenic sequences to variable lengths (2-512bp)...")
    
    resized_samples = []
    buffer = ""
    buffer_metadata = None
    
    # Convert dataset to list for easier iteration
    data_list = list(dataset)
    
    for item in data_list:
        seq = item["sequence"]
        meta = {k: v for k, v in item.items() if k != "sequence"}
        
        # Add sequence to buffer
        buffer += seq
        if buffer_metadata is None:
            buffer_metadata = meta.copy()
        
        # Extract variable-length chunks from buffer
        while len(buffer) >= MAX_SEQ_LENGTH:
            # Random length between MIN and MAX
            chunk_size = random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH)
            resized_samples.append({
                "sequence": buffer[:chunk_size],
                **buffer_metadata
            })
            buffer = buffer[chunk_size:]
            
        # Occasionally extract smaller chunks even if buffer isn't full
        # This creates more variability in lengths
        if len(buffer) >= MIN_SEQ_LENGTH and random.random() < 0.3:
            chunk_size = random.randint(MIN_SEQ_LENGTH, min(len(buffer), MAX_SEQ_LENGTH))
            resized_samples.append({
                "sequence": buffer[:chunk_size],
                **buffer_metadata
            })
            buffer = buffer[chunk_size:]
    
    # Handle remaining buffer
    while len(buffer) >= MIN_SEQ_LENGTH:
        chunk_size = random.randint(MIN_SEQ_LENGTH, min(len(buffer), MAX_SEQ_LENGTH))
        resized_samples.append({
            "sequence": buffer[:chunk_size],
            **buffer_metadata
        })
        buffer = buffer[chunk_size:]
    
    print(f"  Created {len(resized_samples)} resized sequences (original: {len(dataset)})")
    
    # Print length distribution
    lengths = [len(s["sequence"]) for s in resized_samples]
    print(f"  New length range: {min(lengths)}-{max(lengths)}bp, mean: {sum(lengths)/len(lengths):.1f}bp")
    
    return Dataset.from_list(resized_samples)

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
    
    # Resize intergenic sequences
    ds = resize_intergenic_sequences(ds, label_str)

    return ds

# Set random seed for reproducibility
random.seed(42)

all_datasets = []

for cfg in DATASETS_CONFIG:
    print(f"Loading {cfg['hf_name']} ...")
    ds = load_and_label_dataset(
        cfg["hf_name"], 
        cfg["label"],
        filter_label=cfg.get("filter_label")
    )
    print(f"  Final count: {len(ds)} samples for {cfg['label']}")
    all_datasets.append(ds)

unified_dataset = concatenate_datasets(all_datasets)

print("\nUnified dataset created:")
print(unified_dataset)

# Print sequence length statistics by class
from collections import defaultdict
lengths_by_label = defaultdict(list)
for item in unified_dataset:
    lengths_by_label[item["label_name"]].append(len(item["sequence"]))

print(f"\nLength statistics by class:")
for label_name in ["ENHANCER", "PROMOTER", "INTERGENIC"]:
    if label_name in lengths_by_label:
        lens = lengths_by_label[label_name]
        print(f"  {label_name}:")
        print(f"    Count: {len(lens)}")
        print(f"    Min: {min(lens)}, Max: {max(lens)}, Mean: {sum(lens)/len(lens):.1f}")

output_dir = Path("../DNA_dataset")
output_dir.mkdir(exist_ok=True)

parquet_path = output_dir / "DNA_multiclass.parquet"
unified_dataset.to_parquet(parquet_path)

print(f"\nDataset saved to: {parquet_path.resolve()}")