# -*- coding: utf-8 -*-
"""
Re-score essays where any score column is -1, blank, or non-numeric.
If the rescore result is still invalid, retry until it succeeds.

Usage:
 python rescore_invalid.py
 python rescore_invalid.py --model doubao_seed_2_0_pro
 python rescore_invalid.py --model doubao_seed_2_0_pro --limit 10
 python rescore_invalid.py --resume
"""

import json
import time
import argparse
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

from config import MODELS, MAX_RETRIES, RETRY_DELAY, CHECKPOINT_INTERVAL, REQUEST_TIMEOUT
from prompts import SCORING_PROMPT_TEMPLATE, COMMENT_PROMPT_TEMPLATE

SCORE_COLS = ["total_score", "relevance_score", "content_score", "expression_score"]
MAX_RESCORE_ATTEMPTS = 5


def build_score_messages(grade, hw_title, hw_requirement, stu_title, content):
    user_content = SCORING_PROMPT_TEMPLATE.format(
        grade=grade,
        hw_title=hw_title,
        hw_requirement=hw_requirement,
        stu_title=stu_title,
        content=content,
    )
    return [{"role": "user", "content": user_content}]


def build_comment_messages(grade, hw_title, hw_requirement, stu_title, content):
    user_content = COMMENT_PROMPT_TEMPLATE.format(
        grade=grade,
        hw_title=hw_title,
        hw_requirement=hw_requirement,
        stu_title=stu_title,
        content=content,
    )
    return [{"role": "user", "content": user_content}]


def parse_score_response(text):
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
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=1,
                timeout=timeout,
                extra_body={"disable_reasoning": True},
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                wait_time = RETRY_DELAY * (2 ** attempt)
                print(f"  Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  API error (non-retryable): {e}")
                return None
    print("  Max retries exceeded.")
    return None


def is_invalid_result(result):
    """Check if a result dict has any invalid score."""
    for col in SCORE_COLS:
        val = result.get(col)
        if val is None or val == -1:
            return True
        try:
            int(val)
        except (ValueError, TypeError):
            return True
    return False


def rescore_single_essay(client, model_name, row):
    """Score a single essay. Retry up to MAX_RESCORE_ATTEMPTS if result is invalid."""
    hw_title = str(row["hw_title"]) if pd.notna(row["hw_title"]) else ""
    hw_requirement = str(row["hw_requirement"]) if pd.notna(row["hw_requirement"]) else ""
    stu_title = str(row["stu_title"]) if pd.notna(row["stu_title"]) else ""
    content = str(row["content"]) if pd.notna(row["content"]) else ""
    class_id = str(int(row["class_id"])) if pd.notna(row["class_id"]) else ""

    for attempt in range(1, MAX_RESCORE_ATTEMPTS + 1):
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

        result = {**scores, **comments}

        if not is_invalid_result(result):
            return result

        print(f"  Attempt {attempt}/{MAX_RESCORE_ATTEMPTS} still invalid: {scores}")

    print(f"  Failed after {MAX_RESCORE_ATTEMPTS} attempts, keeping last result")
    return result


def find_invalid_rows(df):
    """Return boolean mask of rows where any score column is -1, NaN, or non-numeric."""
    mask = pd.Series(False, index=df.index)
    for col in SCORE_COLS:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            mask = mask | (numeric == -1) | numeric.isna()
    return mask


def get_checkpoint_path(model_key):
    return Path("results") / f"{model_key}_rescore_checkpoint.json"


def load_checkpoint(model_key):
    path = get_checkpoint_path(model_key)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(model_key, data):
    path = get_checkpoint_path(model_key)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Re-score essays with invalid scores (-1, blank, or non-numeric)")
    parser.add_argument("--model", type=str, default=None,
                        help="Specific model to rescore (e.g., doubao_seed_2_0_pro). If not set, checks all.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of invalid essays to rescore (for testing).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint.")
    parser.add_argument("--max-attempts", type=int, default=5,
                        help=f"Max rescore attempts per essay (default: {5})")
    args = parser.parse_args()

    # global MAX_RESCORE_ATTEMPTS = 5
    # MAX_RESCORE_ATTEMPTS = args.max_attempts

    models_to_check = [args.model] if args.model else list(MODELS.keys())

    for model_key in models_to_check:
        if model_key not in MODELS:
            print(f"Model '{model_key}' not found in config.py. Skipping.")
            continue

        score_path = f"results/{model_key}_scores.xlsx"
        if not Path(score_path).exists():
            print(f"[SKIP] {score_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Checking {model_key}")
        print(f"{'='*60}")

        df = pd.read_excel(score_path)
        invalid_mask = find_invalid_rows(df)
        invalid_indices = df.index[invalid_mask].tolist()

        if not invalid_indices:
            print(f"All {len(df)} rows are valid. Nothing to rescore.")
            continue

        print(f"Found {len(invalid_indices)} invalid rows out of {len(df)}")

        if args.limit:
            invalid_indices = invalid_indices[:args.limit]
            print(f"Limited to {len(invalid_indices)} rows for this run")

        cfg = MODELS[model_key]
        client = OpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            timeout=REQUEST_TIMEOUT,
        )

        checkpoint = load_checkpoint(model_key) if args.resume else {}

        for count, idx in enumerate(tqdm(invalid_indices, desc=f"rescore {model_key}")):
            if str(idx) in checkpoint:
                continue

            row = df.loc[idx]
            result = rescore_single_essay(client, cfg["model_name"], row)

            for col in SCORE_COLS:
                df.at[idx, col] = result[col]
            for col in ["relevance_comment", "content_comment", "expression_comment"]:
                df.at[idx, col] = result[col]

            checkpoint[str(idx)] = result

            if (count + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(model_key, checkpoint)
                df.to_excel(score_path, index=False)
                print(f"\n  Checkpoint & xlsx saved at {count+1}/{len(invalid_indices)}")

        # Final save
        save_checkpoint(model_key, checkpoint)
        df.to_excel(score_path, index=False)
        print(f"Results saved to {score_path}")

        # Verify
        remaining_invalid = find_invalid_rows(df).sum()
        print(f"Remaining invalid rows: {remaining_invalid}/{len(df)}")

        # Cleanup checkpoint
        ckpt_path = get_checkpoint_path(model_key)
        if ckpt_path.exists():
            ckpt_path.unlink()
            print("Checkpoint cleaned up.")


if __name__ == "__main__":
    main()
