# -*- coding: utf-8 -*-
"""
Cross-prompt generalization experiment for LLM essay scoring.

Leave-one-prompt-out evaluation:
- Each round: 1 prompt as test set, remaining 13 prompts as few-shot pool
- Few-shot examples selected from training pool, one per score level (0-8)
- Two settings: 1 sample/score (9 total) and 3 samples/score (27 total)

Checkpoint & Resume:
- After each prompt, predictions are saved to parquet checkpoint
- Also supports legacy _checkpoint.json (metrics-only, skips completed prompts)
- Also supports _predictions.xlsx (per-essay data from completed runs)
- Use --resume to continue from last checkpoint (default: True)

Usage:
    python cross_prompt_exp/run_experiment.py --model qwen3_6_plus
    python cross_prompt_exp/run_experiment.py --model qwen3_6_plus --no-resume
    python cross_prompt_exp/run_experiment.py --model qwen3_6_plus --debug
    python cross_prompt_exp/run_experiment.py --models qwen3_6_plus claude_sonnet_4_6 deepseek
    python cross_prompt_exp/run_experiment.py --prompt "故事新编"

结果输出到 results/cross_prompt/，包含每篇作文的预测、每个prompt的指标、以及汇总对比表。
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add parent dir to path so config can be found
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_PATH, OUTPUT_DIR, PROMPT_COLUMN, FEW_SHOT_PER_BUCKET,
    RANDOM_SEED, MODELS, DEFAULT_MODEL, MAX_CONCURRENT,
    DEBUG_MODE, DEBUG_TEST_SIZE,
)
from data_utils import load_data, leave_one_prompt_out_splits, select_few_shot_examples
from prompts import build_messages
from scorer import score_essay
from metrics import evaluate_predictions, SCORE_DIMS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def score_single_essay(row, few_shot_examples, model_config):
    """Score a single essay row.

    Args:
        row: DataFrame row with essay data
        few_shot_examples: list of few-shot example dicts
        model_config: LLM config dict

    Returns:
        dict: {dimension}_pred and {dimension}_truth pairs
    """
    messages = build_messages(
        hw_title=row["hw_title"],
        hw_requirement=row["hw_requirement"],
        stu_title=row["stu_title"],
        content=row["content"],
        few_shot_examples=few_shot_examples,
    )

    scores = score_essay(messages, model_config)

    result = {}
    for dim in SCORE_DIMS:
        result[f"{dim}_pred"] = scores.get(dim, -1)
        result[f"{dim}_truth"] = int(row[dim])

    return result


def run_one_prompt(test_prompt, test_df, few_shot_examples, model_config, existing_pred_df=None):
    """Run scoring for all essays in one test prompt, resuming from checkpoint if available.

    Args:
        test_prompt: test prompt name
        test_df: test set DataFrame
        few_shot_examples: list of few-shot example dicts
        model_config: LLM config dict
        existing_pred_df: previously scored results for this prompt (may be None)

    Returns:
        pd.DataFrame: combined results with pred/truth columns for each dimension
    """
    # Determine which essays still need scoring
    scored_indices = set()
    if existing_pred_df is not None and len(existing_pred_df) > 0:
        scored_indices = set(existing_pred_df["_idx"].tolist())

    pending_indices = [i for i in range(len(test_df)) if i not in scored_indices]
    n_pending = len(pending_indices)
    n_scored = len(scored_indices)

    if n_scored > 0:
        logger.info(f"  Resuming {test_prompt}: {n_scored} already scored, {n_pending} remaining")

    if n_pending == 0:
        logger.info(f"  [{test_prompt}] All {len(test_df)} essays already scored, skipping")
        return existing_pred_df

    # Score pending essays
    new_results = []
    fail_count = 0

    pbar = tqdm(pending_indices, desc=f"  {test_prompt[:15]}", unit="essay")

    if MAX_CONCURRENT > 1:
        # Parallel scoring with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            futures = {
                executor.submit(score_single_essay, test_df.iloc[idx], few_shot_examples, model_config): idx
                for idx in pending_indices
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    result["_idx"] = idx
                    new_results.append(result)
                    if all(result.get(f"{dim}_pred", -1) == -1 for dim in SCORE_DIMS):
                        fail_count += 1
                except Exception as e:
                    logger.error(f"  Error scoring essay {idx}: {e}")
                    r = {"_idx": idx}
                    for dim in SCORE_DIMS:
                        r[f"{dim}_pred"] = -1
                        r[f"{dim}_truth"] = int(test_df.iloc[idx][dim])
                    new_results.append(r)
                    fail_count += 1
                pbar.update(1)
    else:
        # Sequential scoring
        for idx in pending_indices:
            row = test_df.iloc[idx]
            try:
                result = score_single_essay(row, few_shot_examples, model_config)
                result["_idx"] = idx
                new_results.append(result)
                if all(result.get(f"{dim}_pred", -1) == -1 for dim in SCORE_DIMS):
                    fail_count += 1
            except Exception as e:
                logger.error(f"  Error scoring essay {idx}: {e}")
                r = {"_idx": idx}
                for dim in SCORE_DIMS:
                    r[f"{dim}_pred"] = -1
                    r[f"{dim}_truth"] = int(test_df.iloc[idx][dim])
                new_results.append(r)
                fail_count += 1
            pbar.update(1)

    pbar.close()

    # Combine with existing
    new_df = pd.DataFrame(new_results)
    if existing_pred_df is not None and len(existing_pred_df) > 0:
        result_df = pd.concat([existing_pred_df, new_df], ignore_index=True)
    else:
        result_df = new_df

    fail_rate = fail_count / n_pending * 100
    logger.info(f"  [{test_prompt}] Scored {n_pending} new, {n_scored} resumed. "
                f"Failures: {fail_count}/{n_pending} ({fail_rate:.1f}%)")

    return result_df


# ============================================================
# Checkpoint paths — supports parquet, xlsx, and legacy json
# ============================================================

def get_parquet_path(output_dir, model_key, setting_name):
    return output_dir / f"{model_key}_{setting_name}_predictions.parquet"


def get_xlsx_path(output_dir, model_key, setting_name):
    return output_dir / f"{model_key}_{setting_name}_predictions.xlsx"


def get_metrics_json_path(output_dir, model_key, setting_name):
    return output_dir / f"{model_key}_{setting_name}_metrics.json"


def get_legacy_ckpt_path(output_dir, model_key, setting_name):
    return output_dir / f"{model_key}_{setting_name}_checkpoint.json"


def load_predictions_parquet(output_dir, model_key, setting_name):
    """Load parquet checkpoint. Returns {prompt: DataFrame} or None."""
    path = get_parquet_path(output_dir, model_key, setting_name)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        groups = {}
        for p in df["_prompt"].unique():
            groups[p] = df[df["_prompt"] == p].reset_index(drop=True)
        logger.info(f"  Loaded parquet checkpoint: {len(groups)} prompts, "
                     f"{len(df)} essays ({path.name})")
        return groups
    except Exception as e:
        logger.warning(f"  Failed to load parquet {path}: {e}")
        return None


def load_predictions_xlsx(output_dir, model_key, setting_name):
    """Load xlsx predictions. Returns {prompt: DataFrame} or None."""
    path = get_xlsx_path(output_dir, model_key, setting_name)
    if not path.exists():
        return None
    try:
        df = pd.read_excel(path)
        if "_prompt" not in df.columns:
            logger.warning(f"  xlsx {path.name} missing _prompt column, skipping")
            return None
        groups = {}
        for p in df["_prompt"].unique():
            groups[p] = df[df["_prompt"] == p].reset_index(drop=True)
        logger.info(f"  Loaded xlsx predictions: {len(groups)} prompts, "
                     f"{len(df)} essays ({path.name})")
        return groups
    except Exception as e:
        logger.warning(f"  Failed to load xlsx {path}: {e}")
        return None


def load_legacy_metrics_json(output_dir, model_key, setting_name):
    """Load legacy _checkpoint.json. Returns set of completed prompt names."""
    path = get_legacy_ckpt_path(output_dir, model_key, setting_name)
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        prompts_done = set(data.keys())
        logger.info(f"  Found legacy checkpoint: {len(prompts_done)} prompts done "
                     f"({path.name})")
        return prompts_done
    except Exception as e:
        logger.warning(f"  Failed to load legacy checkpoint {path}: {e}")
        return set()


def load_checkpoint(output_dir, model_key, setting_name):
    """Load checkpoint from any available source.

    Priority: parquet > xlsx > legacy json (metrics only, no per-essay data)

    Returns:
        tuple: (prompt_groups_dict or None, skipped_prompts_set)
        - prompt_groups_dict: {prompt: DataFrame} with per-essay predictions
        - skipped_prompts_set: prompts known to be done (but no per-essay data)
    """
    # Try parquet first
    parquet_groups = load_predictions_parquet(output_dir, model_key, setting_name)
    if parquet_groups:
        return parquet_groups, set()

    # Try xlsx
    xlsx_groups = load_predictions_xlsx(output_dir, model_key, setting_name)
    if xlsx_groups:
        return xlsx_groups, set()

    # Try legacy metrics json
    legacy_done = load_legacy_metrics_json(output_dir, model_key, setting_name)
    if legacy_done:
        logger.info(f"  Legacy checkpoint has metrics but no per-essay predictions. "
                     f"These prompts will be skipped (no essay-level resume): "
                     f"{', '.join(sorted(legacy_done))}")
        return None, legacy_done

    return None, set()


def save_predictions_checkpoint(output_dir, model_key, setting_name, all_pred_rows):
    """Save predictions checkpoint to parquet."""
    path = get_parquet_path(output_dir, model_key, setting_name)
    if all_pred_rows:
        combined = pd.concat(all_pred_rows, ignore_index=True)
        combined.to_parquet(path, index=False)
        logger.info(f"  Saved checkpoint: {len(all_pred_rows)} prompts, "
                     f"{len(combined)} essays -> {path.name}")


# ============================================================
# Main experiment
# ============================================================

def run_experiment(args):
    """Run the full cross-prompt experiment."""
    # Load data
    df = load_data()
    all_prompts = sorted(df[PROMPT_COLUMN].unique())

    # Determine models to run
    if args.models:
        models_to_run = args.models
    else:
        models_to_run = [args.model]

    # Determine which prompts to test
    test_prompts = args.prompt if args.prompt else None
    if isinstance(test_prompts, str):
        test_prompts = [test_prompts]

    # Output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}  # {model: {setting: per-prompt results}}

    for model_key in models_to_run:
        if model_key not in MODELS:
            logger.warning(f"Model {model_key} not found in config, skipping")
            continue

        model_config = MODELS[model_key]
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_key} ({model_config['model_name']})")
        logger.info(f"{'='*60}")

        model_results = {}  # {setting: per-prompt results}

        for n_per_bucket in FEW_SHOT_PER_BUCKET:
            setting_name = f"{n_per_bucket}_per_bucket"
            logger.info(f"\n{'='*50}")
            logger.info(f"Setting: {setting_name} (few-shot: {n_per_bucket} per bucket)")
            logger.info(f"{'='*50}")

            setting_results = {}  # {prompt_name: metrics}
            all_pred_rows = []  # list of DataFrames, one per prompt

            # Load checkpoint
            ckpt_groups = None
            skipped_prompts = set()
            if args.resume:
                ckpt_groups, skipped_prompts = load_checkpoint(
                    output_dir, model_key, setting_name
                )
                if ckpt_groups:
                    for prompt_name, prompt_df in ckpt_groups.items():
                        all_pred_rows.append(prompt_df)
                        metrics = evaluate_predictions(prompt_df)
                        setting_results[prompt_name] = metrics
                    logger.info(f"  Resuming: {len(setting_results)} prompts with "
                                f"per-essay data already done")
                if skipped_prompts:
                    for prompt_name in skipped_prompts:
                        # Load metrics from legacy checkpoint
                        legacy_path = get_legacy_ckpt_path(
                            output_dir, model_key, setting_name
                        )
                        if legacy_path.exists():
                            with open(legacy_path, "r", encoding="utf-8") as f:
                                legacy_data = json.load(f)
                            if prompt_name in legacy_data:
                                setting_results[prompt_name] = legacy_data[prompt_name]
                    logger.info(f"  Resuming: {len(skipped_prompts)} additional prompts "
                                f"done (metrics only, no per-essay resume)")
            else:
                # Clean old checkpoints if --no-resume
                for suffix in ["predictions.parquet", "predictions.xlsx",
                               "checkpoint.json", "metrics.json"]:
                    old = output_dir / f"{model_key}_{setting_name}_{suffix}"
                    if old.exists():
                        old.unlink()
                        logger.info(f"  Removed old file: {old.name}")

            # Determine prompts to run
            prompts_to_run = [p for p in all_prompts
                              if p not in setting_results
                              and (test_prompts is None or p in test_prompts)]
            total_prompts = len(prompts_to_run) + len(setting_results)
            n_done = len(setting_results)

            logger.info(f"\nProgress: {n_done}/{total_prompts} prompts already done, "
                         f"{len(prompts_to_run)} remaining")

            if not prompts_to_run:
                logger.info(f"  All {total_prompts} prompts already completed, skipping")
                # Still need to compute overall averages
                pass

            # Overall progress bar across prompts
            pbar = tqdm(prompts_to_run, desc=f"  {model_key}/{setting_name}",
                         unit="prompt", total=len(prompts_to_run))

            for test_prompt in pbar:
                pbar.set_postfix({"current": test_prompt[:12]})

                # Skip prompts not in requested list
                if test_prompts and test_prompt not in test_prompts:
                    continue

                test_df = df[df[PROMPT_COLUMN] == test_prompt].reset_index(drop=True)
                train_pool_df = df[df[PROMPT_COLUMN] != test_prompt].reset_index(drop=True)

                # Debug mode: subsample test set
                if DEBUG_MODE:
                    test_df = test_df.head(DEBUG_TEST_SIZE)
                    logger.info(f"  [DEBUG] Using only {len(test_df)} test essays")

                # Get existing predictions for this prompt from checkpoint
                existing_pred_df = None
                if ckpt_groups and test_prompt in ckpt_groups:
                    existing_pred_df = ckpt_groups[test_prompt]

                # Select few-shot examples
                few_shot_examples = select_few_shot_examples(
                    train_pool_df, n_per_bucket, seed=RANDOM_SEED
                )

                # Score all test essays (resumes from checkpoint)
                start_time = time.time()
                pred_df = run_one_prompt(
                    test_prompt, test_df, few_shot_examples, model_config,
                    existing_pred_df=existing_pred_df,
                )
                elapsed = time.time() - start_time
                logger.info(f"  Completed in {elapsed:.1f}s")

                # Evaluate
                metrics = evaluate_predictions(pred_df)
                setting_results[test_prompt] = metrics

                # Collect for overall average
                pred_df["_prompt"] = test_prompt
                all_pred_rows.append(pred_df)

                # Print per-prompt metrics
                print(f"\n  [{test_prompt}]")
                for dim, m in metrics.items():
                    acc = f"{m['acc']:.4f}" if m['acc'] is not None else "N/A"
                    qwk = f"{m['qwk']:.4f}" if m['qwk'] is not None else "N/A"
                    pcc = f"{m['pcc']:.4f}" if m['pcc'] is not None else "N/A"
                    mae = f"{m['mae']:.4f}" if m['mae'] is not None else "N/A"
                    print(f"    {dim}: ACC={acc} QWK={qwk} PCC={pcc} MAE={mae} (n={m['n_valid']})")

                # Save checkpoint after each prompt
                save_predictions_checkpoint(output_dir, model_key, setting_name, all_pred_rows)

                # Save per-prompt metrics checkpoint
                metrics_ckpt_path = get_metrics_json_path(
                    output_dir, model_key, setting_name
                )
                with open(metrics_ckpt_path, "w", encoding="utf-8") as f:
                    json.dump(setting_results, f, ensure_ascii=False, indent=2, default=str)

            pbar.close()

            if not setting_results:
                logger.warning(f"No prompts evaluated for {setting_name}")
                continue

            model_results[setting_name] = setting_results

            # Compute overall averages
            if all_pred_rows:
                combined_df = pd.concat(all_pred_rows, ignore_index=True)
                overall_metrics = evaluate_predictions(combined_df)

                logger.info(f"\n{'='*50}")
                logger.info(f"OVERALL AVERAGES — {model_key} / {setting_name}")
                logger.info(f"{'='*50}")
                for dim, m in overall_metrics.items():
                    acc = f"{m['acc']:.4f}" if m['acc'] is not None else "N/A"
                    qwk = f"{m['qwk']:.4f}" if m['qwk'] is not None else "N/A"
                    pcc = f"{m['pcc']:.4f}" if m['pcc'] is not None else "N/A"
                    mae = f"{m['mae']:.4f}" if m['mae'] is not None else "N/A"
                    logger.info(f"  {dim}: ACC={acc} QWK={qwk} PCC={pcc} MAE={mae} (n={m['n_valid']})")

                # Report failure rate
                n_total = len(combined_df)
                n_failed = sum(
                    1 for _, row in combined_df.iterrows()
                    if all(row.get(f"{dim}_pred", -1) == -1 for dim in SCORE_DIMS)
                )
                logger.info(f"  API failure rate: {n_failed}/{n_total} ({n_failed/n_total*100:.1f}%)")

                # Save per-essay predictions as Excel
                pred_path = get_xlsx_path(output_dir, model_key, setting_name)
                combined_df.to_excel(pred_path, index=False)
                logger.info(f"Saved predictions to {pred_path}")

            # Save per-prompt metrics
            metrics_path = output_dir / f"{model_key}_{setting_name}_metrics.json"
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(setting_results, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Saved per-prompt metrics to {metrics_path}")

        all_results[model_key] = model_results

        # Save per-model summary
        summary_path = output_dir / f"summary_{model_key}.xlsx"
        save_summary({model_key: model_results}, summary_path)
        logger.info(f"Saved summary for {model_key} to {summary_path}")

    # Print final summary across all models and settings
    print_final_summary(all_results)


def print_final_summary(all_results):
    """Print final summary table comparing all models and settings."""
    print(f"\n{'='*80}")
    print(" FINAL SUMMARY — Cross-Prompt Generalization")
    print(f"{'='*80}")

    for model_key, model_results in all_results.items():
        print(f"\nModel: {model_key}")
        print(f"{'-'*60}")
        print(f"{'Setting':<20} {'QWK (total)':>12} {'QWK (relevance)':>14} {'QWK (content)':>13} {'QWK (expression)':>15}")
        print(f"{'-'*60}")

        for setting_name, prompt_results in model_results.items():
            qwk_sums = {}
            for prompt_metrics in prompt_results.values():
                for dim, m in prompt_metrics.items():
                    if m['qwk'] is not None:
                        qwk_sums.setdefault(dim, []).append(m['qwk'])

            row = f"{setting_name:<20}"
            for dim in SCORE_DIMS:
                qwks = qwk_sums.get(dim, [])
                avg_qwk = sum(qwks) / len(qwks) if qwks else float('nan')
                row += f" {avg_qwk:>12.4f}"
            print(row)


def save_summary(all_results, output_path):
    """Save combined summary to Excel."""
    rows = []
    for model_key, model_results in all_results.items():
        for setting_name, prompt_results in model_results.items():
            for prompt, metrics in prompt_results.items():
                for dim, m in metrics.items():
                    rows.append({
                        "model": model_key,
                        "setting": setting_name,
                        "prompt": prompt,
                        "dimension": dim,
                        "ACC": m.get("acc"),
                        "QWK": m.get("qwk"),
                        "PCC": m.get("pcc"),
                        "MAE": m.get("mae"),
                        "n_valid": m.get("n_valid"),
                    })

    summary_df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_excel(output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-prompt LLM essay scoring experiment")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Multiple models to evaluate")
    parser.add_argument("--prompt", type=str, nargs="+", default=None,
                        help="Specific prompts to test (default: all)")
    parser.add_argument("--debug", action="store_true",
                        help="Run in debug mode with fewer test essays")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable resume, start fresh (removes existing checkpoints)")
    args = parser.parse_args()

    args.resume = not args.no_resume

    if args.debug:
        import config
        config.DEBUG_MODE = True

    run_experiment(args)
