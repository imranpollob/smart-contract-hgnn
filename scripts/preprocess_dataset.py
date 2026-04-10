"""
Pre-process all contracts in the aggregated benchmark through Steps 1-5.
Saves processed data to a .pt file that can be loaded on Colab/GPU for training.

Usage:
    uv run python scripts/preprocess_dataset.py
    uv run python scripts/preprocess_dataset.py --output data/processed_dataset.pt
"""

import argparse
import logging
import os
import sys
import time

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.train import generate_cv_splits, process_contract

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_and_save(output_path: str = "data/processed_dataset.pt"):
    """
    Process all contracts and save fold data to a single .pt file.

    The saved file contains:
        - folds: list of 3 dicts, each with 'train' and 'val' lists
        - Each item in train/val is a dict with: X, H_inc, E, node_index, label, n_hyperedges, sol_path
        - metadata: dataset stats
    """
    start = time.time()

    # Generate CV splits
    logger.info("Generating CV splits...")
    folds = generate_cv_splits(n_splits=3, random_state=42)

    # Process all unique contracts (avoid re-processing across folds)
    all_contracts = set()
    for fold in folds:
        for path, label in fold["train"] + fold["val"]:
            all_contracts.add((path, label))

    logger.info(f"Processing {len(all_contracts)} unique contracts...")
    processed_cache = {}
    success = 0
    failed = 0
    no_calls = 0

    for i, (path, label) in enumerate(sorted(all_contracts)):
        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i + 1}/{len(all_contracts)} ({success} ok, {failed} failed, {no_calls} no calls)")

        result = process_contract(path, label)
        if result is not None:
            processed_cache[path] = result
            success += 1
        else:
            failed += 1
            # Check if it was no-calls vs actual failure
            # (we can't easily distinguish here, count both as failed)

    logger.info(f"Processing complete: {success} success, {failed} failed/no-calls out of {len(all_contracts)}")

    # Build fold data from cache
    fold_data = []
    for fold_idx, fold in enumerate(folds):
        train_data = []
        val_data = []

        for path, label in fold["train"]:
            if path in processed_cache:
                train_data.append(processed_cache[path])

        for path, label in fold["val"]:
            if path in processed_cache:
                val_data.append(processed_cache[path])

        fold_data.append({"train": train_data, "val": val_data})
        logger.info(f"Fold {fold_idx + 1}: {len(train_data)} train, {len(val_data)} val")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_data = {
        "fold_data": fold_data,
        "metadata": {
            "total_contracts": len(all_contracts),
            "processed_ok": success,
            "failed": failed,
            "n_folds": 3,
            "random_state": 42,
        },
    }
    torch.save(save_data, output_path)

    elapsed = time.time() - start
    logger.info(f"Saved to {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process dataset for GPU training")
    parser.add_argument("--output", type=str, default="data/processed_dataset.pt",
                        help="Output path for processed data")
    args = parser.parse_args()
    preprocess_and_save(args.output)
