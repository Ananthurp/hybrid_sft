import json
import os
from sklearn.model_selection import train_test_split # type: ignore
import pandas as pd

# --- Configuration ---
INPUT_FILE = "/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_qwen2-1.5b/train.jsonl"
OUTPUT_DIR = "/data/ananthu/gem_project/datasets/ultrafeedback_tokenized_qwen2-1.5b_ssl"
PERCENTAGES = [10, 20, 40, 60, 80]

# --- Main Script ---
print(f"Loading full dataset from {INPUT_FILE}...")
with open(INPUT_FILE, 'r') as f:
    full_data = [json.loads(line) for line in f]

print(f"Loaded {len(full_data)} total samples.")

for p in PERCENTAGES:
    print(f"\nCreating split for {p}% labeled data...")

    split_dir = os.path.join(OUTPUT_DIR, f"labeled_{p}_percent")
    os.makedirs(split_dir, exist_ok=True)

    labeled_path = os.path.join(split_dir, "labeled_train.jsonl")
    unlabeled_path = os.path.join(split_dir, "unlabeled_train.jsonl")

    # Split the data using a fixed seed for reproducibility
    labeled_data, unlabeled_data = train_test_split(
        full_data, 
        train_size=(p / 100.0), 
        random_state=42 
    )

    # Save the labeled set
    with open(labeled_path, 'w') as f:
        for item in labeled_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(labeled_data)} samples to {labeled_path}")

    # Save the unlabeled set
    with open(unlabeled_path, 'w') as f:
        for item in unlabeled_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved {len(unlabeled_data)} samples to {unlabeled_path}")

print("\nAll semi-supervised data splits created successfully.")