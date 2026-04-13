"""
Pre-process all contracts in the aggregated benchmark through Steps 1-5.
Saves processed data to a .pt file that can be loaded on Colab/GPU for training.

Groups contracts by solc version to minimize version switching (the main
cause of compilation failures).

Usage:
    uv run python scripts/preprocess_dataset.py
    uv run python scripts/preprocess_dataset.py --output data/processed_dataset.pt
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict

import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.extraction.ast_cfg import detect_pragma_version, install_and_use_solc
from src.evaluation.train import generate_cv_splits, process_contract

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def preprocess_and_save(output_path: str = "data/processed_dataset.pt"):
    """
    Process all contracts and save fold data to a single .pt file.

    Groups contracts by their pragma solc version before processing,
    minimizing solc-select version switches and race conditions.

    The saved file contains:
        - fold_data: list of 3 dicts, each with 'train' and 'val' lists
        - metadata: dataset stats
    """
    start = time.time()

    # Generate CV splits
    logger.info("Generating CV splits...")
    folds = generate_cv_splits(n_splits=3, random_state=42)

    # Collect all unique contracts
    all_contracts = {}
    for fold in folds:
        for path, label in fold["train"] + fold["val"]:
            all_contracts[path] = label

    logger.info(f"Total unique contracts: {len(all_contracts)}")

    # Group contracts by solc version to minimize switching
    version_groups = defaultdict(list)
    no_version = []
    for path, label in sorted(all_contracts.items()):
        version = detect_pragma_version(path)
        if version:
            version_groups[version].append((path, label))
        else:
            no_version.append((path, label))

    logger.info(f"Detected {len(version_groups)} unique solc versions:")
    for ver in sorted(version_groups.keys()):
        logger.info(f"  {ver}: {len(version_groups[ver])} contracts")
    if no_version:
        logger.info(f"  No pragma: {len(no_version)} contracts")

    # Process contracts grouped by version
    processed_cache = {}
    success = 0
    failed = 0
    total = len(all_contracts)

    for version in sorted(version_groups.keys()):
        contracts = version_groups[version]
        logger.info(f"\nProcessing {len(contracts)} contracts with solc {version}...")

        # Install and switch solc version once for the whole group
        if not install_and_use_solc(version):
            logger.warning(f"Cannot install solc {version}, skipping {len(contracts)} contracts")
            failed += len(contracts)
            continue

        for path, label in contracts:
            result = process_contract(path, label)
            if result is not None:
                processed_cache[path] = result
                success += 1
            else:
                failed += 1

            if (success + failed) % 50 == 0:
                logger.info(f"  Progress: {success + failed}/{total} ({success} ok, {failed} failed)")

    # Process contracts with no detected version
    if no_version:
        logger.info(f"\nProcessing {len(no_version)} contracts with no detected pragma...")
        for path, label in no_version:
            result = process_contract(path, label)
            if result is not None:
                processed_cache[path] = result
                success += 1
            else:
                failed += 1

    logger.info(f"\nProcessing complete: {success} success, {failed} failed out of {total}")
    logger.info(f"Success rate: {success/total*100:.1f}%")

    # Build fold data from cache
    fold_data = []
    for fold_idx, fold in enumerate(folds):
        train_data = [processed_cache[p] for p, _ in fold["train"] if p in processed_cache]
        val_data = [processed_cache[p] for p, _ in fold["val"] if p in processed_cache]
        fold_data.append({"train": train_data, "val": val_data})
        logger.info(f"Fold {fold_idx + 1}: {len(train_data)} train, {len(val_data)} val")

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_data = {
        "fold_data": fold_data,
        "metadata": {
            "total_contracts": total,
            "processed_ok": success,
            "failed": failed,
            "n_folds": 3,
            "random_state": 42,
        },
    }
    torch.save(save_data, output_path)

    elapsed = time.time() - start
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    logger.info(f"Saved to {output_path} ({size_mb:.1f} MB)")
    logger.info(f"Total time: {elapsed / 60:.1f} minutes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process dataset for GPU training")
    parser.add_argument("--output", type=str, default="data/processed_dataset.pt",
                        help="Output path for processed data")
    args = parser.parse_args()
    preprocess_and_save(args.output)
