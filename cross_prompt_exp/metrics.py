# -*- coding: utf-8 -*-
"""
Evaluation metrics for cross-prompt experiment.
Computes ACC, QWK, PCC, and MAE between predicted and ground truth scores.
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score, mean_absolute_error


SCORE_DIMS = ["total_score", "relevance_score", "content_score", "expression_score"]


def compute_metrics(pred, truth):
    """Compute ACC, QWK, PCC, MAE for a pair of score arrays.

    Args:
        pred: predicted scores array
        truth: ground truth scores array

    Returns:
        dict: metrics with keys acc, qwk, pcc, mae, n_valid
    """
    # Filter out invalid scores (-1 or NaN)
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    valid_mask = ~(np.isnan(pred) | np.isnan(truth) | (pred == -1) | (truth == -1))
    pred_valid = pred[valid_mask].astype(float)
    truth_valid = truth[valid_mask].astype(float)

    if len(pred_valid) < 2:
        return {
            "acc": None, "qwk": None, "pcc": None, "mae": None,
            "n_valid": int(len(pred_valid)),
        }

    acc = float((pred_valid == truth_valid).mean())
    qwk = float(cohen_kappa_score(pred_valid, truth_valid, weights="quadratic"))
    pcc = float(pearsonr(pred_valid, truth_valid)[0])
    mae = float(mean_absolute_error(truth_valid, pred_valid))

    return {
        "acc": acc, "qwk": qwk, "pcc": pcc, "mae": mae,
        "n_valid": int(len(pred_valid)),
    }


def evaluate_predictions(pred_df, score_dims=None):
    """Evaluate all score dimensions in a predictions DataFrame.

    Args:
        pred_df: DataFrame with {dim}_pred and {dim}_truth columns
        score_dims: list of dimension names to evaluate

    Returns:
        dict: {dimension: metrics_dict}
    """
    if score_dims is None:
        score_dims = SCORE_DIMS

    results = {}
    for dim in score_dims:
        pred_col = f"{dim}_pred"
        truth_col = f"{dim}_truth"
        if pred_col not in pred_df.columns or truth_col not in pred_df.columns:
            continue
        results[dim] = compute_metrics(pred_df[pred_col], pred_df[truth_col])

    return results


def print_metrics_table(results_dict, title="Metrics"):
    """Print a formatted metrics table.

    Args:
        results_dict: {dimension: metrics_dict} or {label: {dimension: metrics_dict}}
        title: table title
    """
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")

    # Check if nested (per-prompt) or flat (overall)
    first_val = next(iter(results_dict.values()), None)
    if first_val and isinstance(first_val, dict) and "acc" in first_val:
        # Flat: {dimension: metrics}
        _print_flat(results_dict)
    else:
        # Nested: {label: {dimension: metrics}}
        _print_nested(results_dict)


def _print_flat(results):
    """Print flat metrics table."""
    print(f"{'Dimension':<20} {'ACC':>8} {'QWK':>8} {'PCC':>8} {'MAE':>8} {'N':>8}")
    print(f"{'-'*60}")
    for dim, m in results.items():
        acc = f"{m['acc']:.4f}" if m['acc'] is not None else "N/A"
        qwk = f"{m['qwk']:.4f}" if m['qwk'] is not None else "N/A"
        pcc = f"{m['pcc']:.4f}" if m['pcc'] is not None else "N/A"
        mae = f"{m['mae']:.4f}" if m['mae'] is not None else "N/A"
        print(f"{dim:<20} {acc:>8} {qwk:>8} {pcc:>8} {mae:>8} {m['n_valid']:>8}")


def _print_nested(results):
    """Print nested metrics table (per-prompt or per-setting)."""
    all_dims = set()
    for label_results in results.values():
        all_dims.update(label_results.keys())
    all_dims = sorted(all_dims)

    header = f"{'Label':<30}"
    for dim in all_dims:
        header += f" {dim[:8]:>8}"
    print(header)
    print(f"{'-'*len(header)}")

    for label, dim_results in results.items():
        row = f"{label:<30}"
        for dim in all_dims:
            if dim in dim_results:
                qwk = f"{dim_results[dim]['qwk']:.4f}" if dim_results[dim]['qwk'] is not None else "N/A"
                row += f" {qwk:>8}"
            else:
                row += f" {'N/A':>8}"
        print(row)
