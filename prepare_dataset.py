import os
import json
import time
import argparse
from typing import Tuple

import numpy as np
import torchvision


def compute_fixed_split(total_len: int, holdout_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    all_indices = np.arange(total_len)
    rng.shuffle(all_indices)
    holdout_size = int(round(holdout_fraction * total_len))
    holdout_indices = np.sort(all_indices[:holdout_size])
    train_indices = np.sort(all_indices[holdout_size:])
    return train_indices, holdout_indices


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_split(output_dir: str, train_indices: np.ndarray, holdout_indices: np.ndarray, meta: dict) -> None:
    ensure_dir(output_dir)
    # Save as .npy for programmatic loading
    np.save(os.path.join(output_dir, "expert_train_indices.npy"), train_indices)
    np.save(os.path.join(output_dir, "fusion_holdout_indices.npy"), holdout_indices)
    # Also save simple .txt (one index per line) for readability/debugging
    with open(os.path.join(output_dir, "expert_train_indices.txt"), "w", encoding="utf-8") as f:
        for idx in train_indices:
            f.write(f"{int(idx)}\n")
    with open(os.path.join(output_dir, "fusion_holdout_indices.txt"), "w", encoding="utf-8") as f:
        for idx in holdout_indices:
            f.write(f"{int(idx)}\n")
    # Save metadata
    meta_path = os.path.join(output_dir, "split_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_existing_split(output_dir: str) -> Tuple[np.ndarray, np.ndarray] | None:
    train_path = os.path.join(output_dir, "expert_train_indices.npy")
    holdout_path = os.path.join(output_dir, "fusion_holdout_indices.npy")
    if os.path.isfile(train_path) and os.path.isfile(holdout_path):
        train_indices = np.load(train_path)
        holdout_indices = np.load(holdout_path)
        return train_indices, holdout_indices
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare CIFAR-100 dataset split for experts and fusion.")
    parser.add_argument("--data_root", type=str, default="./data", help="Root directory to store/load CIFAR-100")
    parser.add_argument("--output_dir", type=str, default="./splits", help="Directory to save the fixed indices")
    parser.add_argument("--holdout_fraction", type=float, default=0.20, help="Fraction of the full train set to reserve for fusion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    parser.add_argument("--force", action="store_true", help="Overwrite existing saved split if present")
    args = parser.parse_args()

    # If split already exists and not forcing, reuse it
    existing = load_existing_split(args.output_dir)
    if existing is not None and not args.force:
        train_indices, holdout_indices = existing
        print(f"Found existing split in {args.output_dir}. Reusing without changes. Use --force to overwrite.")
        print(f"Train (experts): {len(train_indices)} | Fusion holdout: {len(holdout_indices)}")
        return

    # Ensure dataset is available to get deterministic length (50k for CIFAR-100 train)
    _ = torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True)
    total_len = 50000  # CIFAR-100 training set size

    train_indices, holdout_indices = compute_fixed_split(total_len, args.holdout_fraction, args.seed)

    meta = {
        "dataset": "CIFAR100",
        "subset": "train",
        "total_len": total_len,
        "holdout_fraction": args.holdout_fraction,
        "seed": args.seed,
        "generated_at_unix": int(time.time()),
    }
    save_split(args.output_dir, train_indices, holdout_indices, meta)
    print(f"Saved fixed split to {args.output_dir}")
    print(f"Train (experts): {len(train_indices)} | Fusion holdout: {len(holdout_indices)}")


if __name__ == "__main__":
    main()


