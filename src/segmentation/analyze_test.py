"""Summarize per-image EGE-UNet scores and bootstrap the pooled metrics."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def pooled_scores(counts):
    tp, fp, fn = counts.T
    return np.stack((2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)), axis=-1)


def summarize(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    counts = np.array([[int(row[key]) for key in ("tp", "fp", "fn")] for row in rows])
    dice = np.array([float(row["dice"]) for row in rows])
    ground_truth_sizes = counts[:, 0] + counts[:, 2]
    ordered = np.argsort(ground_truth_sizes, kind="stable")
    size_quartiles = [
        {
            "images": len(group),
            "foreground_pixels_min": int(ground_truth_sizes[group].min()),
            "foreground_pixels_max": int(ground_truth_sizes[group].max()),
            "mean_image_dice": float(dice[group].mean()),
            "dice_below_0_5": int((dice[group] < 0.5).sum()),
        }
        for group in np.array_split(ordered, 4)
    ]
    rng = np.random.default_rng(42)
    resamples = rng.integers(len(rows), size=(5_000, len(rows)))
    intervals = np.percentile(pooled_scores(counts[resamples].sum(axis=1)), [2.5, 97.5], axis=0)

    return {
        "test_images": len(rows),
        "bootstrap_resamples": len(resamples),
        "bootstrap_seed": 42,
        "pooled_dice_iou": pooled_scores(counts.sum(axis=0)).tolist(),
        "pooled_dice_95_percent_interval": intervals[:, 0].tolist(),
        "pooled_iou_95_percent_interval": intervals[:, 1].tolist(),
        "per_image_dice_mean": float(dice.mean()),
        "per_image_dice_quartiles": np.percentile(dice, [25, 50, 75]).tolist(),
        "per_image_dice_below_0_5": int((dice < 0.5).sum()),
        "empty_predictions": int(((counts[:, 0] + counts[:, 1]) == 0).sum()),
        "ground_truth_size_quartiles": size_quartiles,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_image_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()
    args.output_json.write_text(
        json.dumps(summarize(args.per_image_csv), indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
