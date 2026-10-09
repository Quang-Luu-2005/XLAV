"""Metrics for positive binary segmentation queries.

mean_sample_iou differs from class-averaged few-shot mIoU. pixel_ap is averaged
per query; it is not claimed to reproduce the paper's AP protocol.
"""
import numpy as np
from sklearn.metrics import average_precision_score


def measure(probability, target, threshold=0.3):
    probability = np.asarray(probability)
    target = np.asarray(target, dtype=bool)
    if probability.shape != target.shape:
        raise ValueError(f"Shape mismatch: {probability.shape} != {target.shape}")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Expected finite sigmoid probabilities in [0,1].")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0,1].")
    if not target.any():
        raise ValueError("This protocol expects nonempty ground truth masks.")
    prediction = probability >= threshold
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    total = int(prediction.sum() + target.sum())
    return {
        "intersection": intersection, "union": union,
        "iou": intersection / union,
        "dice": 2 * intersection / total,
        "pixel_ap": float(average_precision_score(target.ravel(), probability.ravel())),
    }


def summarize(rows):
    if not rows:
        raise ValueError("No samples to evaluate.")
    return {
        "n_queries": len(rows),
        "mean_sample_iou": float(np.mean([r["iou"] for r in rows])),
        "cumulative_iou": sum(r["intersection"] for r in rows) / sum(r["union"] for r in rows),
        "mean_dice": float(np.mean([r["dice"] for r in rows])),
        "mean_sample_pixel_ap": float(np.mean([r["pixel_ap"] for r in rows])),
        "precision_at_0.5": float(np.mean([r["iou"] >= 0.5 for r in rows])),
        "precision_at_0.7": float(np.mean([r["iou"] >= 0.7 for r in rows])),
    }
