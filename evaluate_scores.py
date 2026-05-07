# -*- coding: utf-8 -*-
"""
Evaluate LLM-generated scores against ground truth.
Computes ACC (Accuracy), QWK (Quadratic Weighted Kappa), and PCC (Pearson Correlation Coefficient)
for each model across total_score, relevance_score, content_score, and expression_score.

Alignment: rows are matched by essay content text, NOT by row index,
to handle models that may have skipped or reordered essays.

Usage:
 python evaluate_scores.py
 python evaluate_scores.py --model gpt-4o
 python evaluate_scores.py --models gpt-4o claude deepseek
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

from config import MODELS


SCORE_COLS = ["total_score", "relevance_score", "content_score", "expression_score"]

# Mapping from result file name prefix to config model key
# Only needed when file name differs from MODELS keys
FILE_TO_MODEL = {
    "deepseek_v3_2": "deepseek",
    "roberta_ordinal_within_prompt": "roberta_ordinal_within_prompt",
    "roberta_ordinal_cross_prompt": "roberta_ordinal_cross_prompt",
}


def compute_metrics(pred, truth):
    """Compute ACC, QWK, PCC for a pair of arrays."""
    # Filter out rows where either pred or truth is invalid (-1 / NaN)
    valid_mask = ~(pd.isna(pred) | pd.isna(truth) | (pred == -1) | (truth == -1))
    pred_valid = pred[valid_mask].astype(float)
    truth_valid = truth[valid_mask].astype(float)

    if len(pred_valid) < 2:
        return {"acc": None, "qwk": None, "pcc": None, "n_valid": len(pred_valid)}

    # ACC: exact match rate
    acc = float((pred_valid == truth_valid).mean())

    # QWK: Quadratic Weighted Kappa
    qwk = float(cohen_kappa_score(pred_valid, truth_valid, weights="quadratic"))

    # PCC: Pearson Correlation Coefficient
    pcc, _ = pearsonr(pred_valid, truth_valid)
    pcc = float(pcc)

    return {"acc": acc, "qwk": qwk, "pcc": pcc, "n_valid": len(pred_valid)}


def evaluate_model(gt_df, model_df, model_name):
    """Evaluate a single model's scores against ground truth."""
    results = []
    for score_col in SCORE_COLS:
        if score_col not in gt_df.columns or score_col not in model_df.columns:
            continue
        metrics = compute_metrics(model_df[score_col], gt_df[score_col])
        results.append({
            "model": model_name,
            "dimension": score_col,
            "ACC": metrics["acc"],
            "QWK": metrics["qwk"],
            "PCC": metrics["pcc"],
            "n_valid": metrics["n_valid"],
        })
    return results


def align_by_content(gt_df, model_df):
    """Align model rows to GT rows by matching essay content text.

    Returns (aligned_gt, aligned_model) where both DataFrames have
    the same rows in the same order. GT rows without a matching model
    essay are dropped.
    """
    if "content" not in gt_df.columns or "content" not in model_df.columns:
        # Fallback: index-based alignment
        min_len = min(len(gt_df), len(model_df))
        return gt_df.iloc[:min_len].reset_index(drop=True), model_df.iloc[:min_len].reset_index(drop=True)

    # Build a lookup: content -> list of GT indices (content may not be unique)
    content_to_gt = {}
    for idx, row in gt_df.iterrows():
        key = str(row["content"])
        content_to_gt.setdefault(key, []).append(idx)

    matched_gt_indices = []
    matched_model_indices = []
    used_gt_counts = {}  # track how many times each GT index has been used

    for m_idx, m_row in model_df.iterrows():
        key = str(m_row["content"])
        if key not in content_to_gt:
            continue
        # Find the next unused GT row with this content
        gt_indices = content_to_gt[key]
        used = used_gt_counts.get(key, 0)
        if used < len(gt_indices):
            matched_gt_indices.append(gt_indices[used])
            matched_model_indices.append(m_idx)
            used_gt_counts[key] = used + 1

    aligned_gt = gt_df.iloc[matched_gt_indices].reset_index(drop=True)
    aligned_model = model_df.iloc[matched_model_indices].reset_index(drop=True)
    return aligned_gt, aligned_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM scores vs ground truth")
    parser.add_argument("--gt", type=str, default="essay_total_data_4500_260417.xlsx",
                        help="Ground truth Excel file")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Specific models to evaluate. If not set, all in config.")
    parser.add_argument("--output", type=str, default="results/score_evaluation.xlsx",
                        help="Output Excel file for evaluation results")
    args = parser.parse_args()

    # Load ground truth
    print(f"Loading ground truth from {args.gt}...")
    gt_df = pd.read_excel(args.gt)
    print(f"Ground truth: {len(gt_df)} rows")

    # Determine which models to evaluate
    # Include PLM baselines alongside LLMs
    default_models = list(MODELS.keys()) + ["roberta_ordinal_within_prompt", "roberta_ordinal_cross_prompt"]
    models_to_eval = args.models if args.models else default_models
    print(f"Models to evaluate: {models_to_eval}")

    all_results = []

    for model_key in models_to_eval:
        # Try config key first, then check FILE_TO_MODEL mapping
        file_prefix = model_key
        # Check if any file maps to this model key
        actual_file_prefix = None
        for fp, mk in FILE_TO_MODEL.items():
            if mk == model_key:
                actual_file_prefix = fp
                break
        if actual_file_prefix is None:
            actual_file_prefix = model_key

        score_path = f"results/{actual_file_prefix}_scores.xlsx"
        if not Path(score_path).exists():
            print(f" [SKIP] {score_path} not found")
            continue

        print(f"\nEvaluating {model_key} (file: {actual_file_prefix}_scores.xlsx)...")
        model_df = pd.read_excel(score_path)

        # Align by essay content to ensure data correspondence
        gt_aligned, model_aligned = align_by_content(gt_df, model_df)
        n_aligned = len(gt_aligned)
        print(f" Aligned {n_aligned} / {len(gt_df)} essays")

        eval_results = evaluate_model(gt_aligned, model_aligned, model_key)
        all_results.extend(eval_results)

        # Print summary for this model
        print(f" {model_key} results:")
        print(f" {'Dimension':<20} {'ACC':>8} {'QWK':>8} {'PCC':>8} {'N':>8}")
        print(f" {'-'*60}")
        for r in eval_results:
            acc = f"{r['ACC']:.4f}" if r['ACC'] is not None else "N/A"
            qwk = f"{r['QWK']:.4f}" if r['QWK'] is not None else "N/A"
            pcc = f"{r['PCC']:.4f}" if r['PCC'] is not None else "N/A"
            print(f" {r['dimension']:<20} {acc:>8} {qwk:>8} {pcc:>8} {r['n_valid']:>8}")

    if not all_results:
        print("No results to save.")
        return

    # Save results
    result_df = pd.DataFrame(all_results)
    # Reorder columns
    result_df = result_df[["model", "dimension", "ACC", "QWK", "PCC", "n_valid"]]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result_df.to_excel(args.output, index=False)
    print(f"\nEvaluation results saved to {args.output}")

    # Also print overall summary
    print(f"\nSummary table:")
    pivot_acc = result_df.pivot_table(values="ACC", index="dimension", columns="model")
    pivot_qwk = result_df.pivot_table(values="QWK", index="dimension", columns="model")
    pivot_pcc = result_df.pivot_table(values="PCC", index="dimension", columns="model")
    print("\nACC matrix:")
    print(pivot_acc.round(4).to_string())
    print("\nQWK matrix:")
    print(pivot_qwk.round(4).to_string())
    print("\nPCC matrix:")
    print(pivot_pcc.round(4).to_string())


if __name__ == "__main__":
    main()
