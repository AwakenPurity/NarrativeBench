# -*- coding: utf-8 -*-
"""
Main script to call LLM APIs to score all essays in the dataset.
Each LLM's results are saved to a separate Excel file.

Usage:
 python score_essays.py
 python score_essays.py --model gpt-4o
 python score_essays.py --model gpt-4o --limit 5
 python score_essays.py --resume
"""

import os
import json
import time
import argparse
import re
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from config import MODELS, MAX_RETRIES, RETRY_DELAY, CHECKPOINT_INTERVAL, REQUEST_TIMEOUT
from prompts import SCORING_PROMPT_TEMPLATE, COMMENT_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def setup_logging(model_key=None):
    """Configure logging to both console and file."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{model_key}" if model_key else ""
    log_file = log_dir / f"score_essays{suffix}_{timestamp}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    logger.info("Log file: %s", log_file)
    return log_file


def build_score_messages(grade, hw_title, hw_requirement, stu_title, content):
    """Build messages for scoring prompt."""
    user_content = SCORING_PROMPT_TEMPLATE.format(
 grade=grade,
        hw_title=hw_title,
        hw_requirement=hw_requirement,
        stu_title=stu_title,
        content=content
    )
    return [{"role": "user", "content": user_content}]


def build_comment_messages(grade, hw_title, hw_requirement, stu_title, content):
    """Build messages for comment generation prompt."""
    user_content = COMMENT_PROMPT_TEMPLATE.format(
 grade=grade,
        hw_title=hw_title,
        hw_requirement=hw_requirement,
        stu_title=stu_title,
        content=content
    )
    return [{"role": "user", "content": user_content}]


def parse_score_response(text):
    """Parse score JSON from LLM response text."""
    text = text.strip()
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            data = json.loads(match.group())
            return {
                "total_score": int(data.get("total_score", -1)),
                "relevance_score": int(data.get("relevance_score", -1)),
                "content_score": int(data.get("content_score", -1)),
                "expression_score": int(data.get("expression_score", -1)),
            }
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    def extract_int(key):
        pattern = rf'"{key}"\s*:\s*(\d+)'
        m = re.search(pattern, text)
        return int(m.group(1)) if m else -1

    return {
        "total_score": extract_int("total_score"),
        "relevance_score": extract_int("relevance_score"),
        "content_score": extract_int("content_score"),
        "expression_score": extract_int("expression_score"),
    }


def parse_comment_response(text):
    """Parse comment JSON from LLM response text."""
    text = text.strip()
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            data = json.loads(match.group())
            return {
                "relevance_comment": str(data.get("relevance_comment", "")),
                "content_comment": str(data.get("content_comment", "")),
                "expression_comment": str(data.get("expression_comment", "")),
            }
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    def extract_str(key):
        pattern = rf'"{key}"\s*:\s*"([^"]*)"'
        m = re.search(pattern, text)
        return m.group(1) if m else ""

    return {
        "relevance_comment": extract_str("relevance_comment"),
        "content_comment": extract_str("content_comment"),
        "expression_comment": extract_str("expression_comment"),
    }


def call_llm(client, model_name, messages, timeout=REQUEST_TIMEOUT):
    """Call LLM API with retries on rate limit."""
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.2,
                timeout=timeout,
                extra_body={
                    "disable_reasoning": True # Seed 2.0系列模型（包括你用的doubao-seed-2-0-pro-260215）用这个参数名
                }
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                wait_time = RETRY_DELAY * (2 ** attempt)
                logger.warning("Rate limit hit, waiting %ds...", wait_time)
                time.sleep(wait_time)
            else:
                logger.error("API error (non-retryable): %s", e)
                return None
    logger.warning("Max retries exceeded.")
    return None


def score_single_essay(client, model_name, row):
    """Score a single essay: get scores + comments."""
    hw_title = str(row["hw_title"]) if pd.notna(row["hw_title"]) else ""
    hw_requirement = str(row["hw_requirement"]) if pd.notna(row["hw_requirement"]) else ""
    stu_title = str(row["stu_title"]) if pd.notna(row["stu_title"]) else ""
    content = str(row["content"]) if pd.notna(row["content"]) else ""
    class_id = str(int(row["class_id"])) if pd.notna(row["class_id"]) else ""

    score_messages = build_score_messages(class_id, hw_title, hw_requirement, stu_title, content)
    score_text = call_llm(client, model_name, score_messages)
    scores = parse_score_response(score_text) if score_text else {
        "total_score": -1, "relevance_score": -1,
        "content_score": -1, "expression_score": -1,
    }

    comment_messages = build_comment_messages(class_id, hw_title, hw_requirement, stu_title, content)
    comment_text = call_llm(client, model_name, comment_messages)
    comments = parse_comment_response(comment_text) if comment_text else {
        "relevance_comment": "", "content_comment": "", "expression_comment": "",
    }

    return {**scores, **comments}


def get_checkpoint_path(model_name):
    """Return path to checkpoint file for a model."""
    return Path("results") / f"{model_name}_checkpoint.json"


def load_checkpoint(model_name):
    """Load checkpoint data if exists."""
    path = get_checkpoint_path(model_name)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "results": []}


def save_checkpoint(model_name, checkpoint_data):
    """Save checkpoint to disk."""
    path = get_checkpoint_path(model_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)


def ensure_output_dir():
    """Create results directory if not exists."""
    os.makedirs("results", exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Score essays with LLM APIs")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to run (e.g., gpt-4o). If not set, runs all.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of essays to score (for testing).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last checkpoint.")
    parser.add_argument("--data", type=str,
                        default="essay_total_data_4500_260417.xlsx",
                        help="Path to input data file.")
    args = parser.parse_args()

    ensure_output_dir()

    print(f"Loading data from {args.data}...")
    df = pd.read_excel(args.data)
    if args.limit:
        df = df.head(args.limit)
    print(f"Total essays to score: {len(df)}")

    models_to_run = [args.model] if args.model else list(MODELS.keys())
    print(f"Models to run: {models_to_run}")

    for model_key in models_to_run:
        if model_key not in MODELS:
            print(f"Model '{model_key}' not found in config.py. Skipping.")
            continue

        setup_logging(model_key)

        cfg = MODELS[model_key]
        logger.info("=" * 60)
        logger.info("Scoring with %s (%s)", model_key, cfg['model_name'])
        logger.info("=" * 60)

        client = OpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            timeout=REQUEST_TIMEOUT,
        )

        checkpoint = load_checkpoint(model_key) if args.resume else {"completed": [], "results": []}
        completed_set = set(checkpoint["completed"])
        results = checkpoint["results"]

        logger.info("Checkpoint: %d already completed, %d remaining.",
                     len(completed_set), len(df) - len(completed_set))

        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc=model_key)):
            excel_row = idx + 2

            if excel_row in completed_set:
                continue

            result = score_single_essay(client, cfg["model_name"], row)
            result["_idx"] = int(idx)
            result["hw_title"] = row["hw_title"]
            result["hw_requirement"] = row["hw_requirement"]
            result["stu_title"] = row["stu_title"]
            result["class_id"] = row["class_id"]
            result["content"] = row["content"]

            results.append(result)
            completed_set.add(excel_row)

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint["completed"] = list(completed_set)
                checkpoint["results"] = results
                save_checkpoint(model_key, checkpoint)
                logger.info("Checkpoint saved at row %d.", i + 1)

        checkpoint["completed"] = list(completed_set)
        checkpoint["results"] = results
        save_checkpoint(model_key, checkpoint)

        col_order = [
            "hw_title", "hw_requirement", "stu_title", "class_id", "content",
            "total_score", "relevance_score", "relevance_comment",
            "content_score", "content_comment",
            "expression_score", "expression_comment",
        ]
        result_df = pd.DataFrame(results)
        for col in col_order:
            if col not in result_df.columns:
                result_df[col] = None
        result_df = result_df[col_order]

        output_path = f"results/{model_key}_scores.xlsx"
        result_df.to_excel(output_path, index=False)
        logger.info("Results saved to %s (%d rows)", output_path, len(result_df))

        ckpt_path = get_checkpoint_path(model_key)
        if ckpt_path.exists():
            ckpt_path.unlink()
        logger.info("Checkpoint cleaned up.")


if __name__ == "__main__":
    main()
