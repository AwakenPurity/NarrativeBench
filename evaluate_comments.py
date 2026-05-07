# -*- coding: utf-8 -*-
"""
Evaluate LLM-generated comments using DeepSeek pairwise comparison.
For each essay, compare LLM comment vs ground truth comment across each dimension.
DeepSeek acts as judge: +1 = LLM wins, 0 = tie, -1 = LLM loses.

Usage:
 python evaluate_comments.py
 python evaluate_comments.py --model gpt-4o
 python evaluate_comments.py --model gpt-4o --limit 5
 python evaluate_comments.py --resume
"""

import os
import json
import time
import argparse
import re
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI

from config import MODELS, DEEPSEEK_EVAL_CONFIG, MAX_RETRIES, RETRY_DELAY, REQUEST_TIMEOUT
from prompts import COMMENT_COMPARE_PROMPT

COMMENT_COLS = ["relevance_comment", "content_comment", "expression_comment"]


def build_compare_messages(comment_a, comment_b):
 """Build messages for pairwise comparison."""
 user_content = COMMENT_COMPARE_PROMPT.format(
 comment_a=comment_a,
 comment_b=comment_b
 )
 return [{"role": "user", "content": user_content}]


def parse_compare_response(text):
 """Parse winner value from DeepSeek comparison response."""
 text = text.strip()
 try:
 match = re.search(r'\{[\s\S]*\}', text)
 if match:
 data = json.loads(match.group())
 winner = int(data.get("winner", 0))
 return winner
 except (json.JSONDecodeError, ValueError, KeyError):
 pass
 m = re.search(r'"winner"\s*:\s*(-?\d+)', text)
 if m:
 return int(m.group(1))
 return 0


def call_llm(client, model_name, messages, timeout=REQUEST_TIMEOUT):
 """Call LLM API with retries on rate limit."""
 for attempt in range(MAX_RETRIES):
 try:
 response = client.chat.completions.create(
 model=model_name,
 messages=messages,
 temperature=0,
 timeout=timeout,
 )
 return response.choices[0].message.content
 except Exception as e:
 err_str = str(e).lower()
 if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
 wait_time = RETRY_DELAY * (2 ** attempt)
 print(f" Rate limit hit, waiting {wait_time}s...")
 time.sleep(wait_time)
 else:
 print(f" API error (non-retryable): {e}")
 return None
 print(" Max retries exceeded.")
 return None


def compare_comments(client, model_name, comment_a, comment_b):
 """Compare two comments, return +1/0/-1."""
 messages = build_compare_messages(comment_a, comment_b)
 response = call_llm(client, model_name, messages)
 if response is None:
 return 0
 return parse_compare_response(response)


def ensure_output_dir():
 os.makedirs("results", exist_ok=True)


def load_checkpoint(model_key, comment_dim):
 path = Path("results") / f"{model_key}_{comment_dim}_comment_checkpoint.json"
 if path.exists():
 with open(path, "r", encoding="utf-8") as f:
 return json.load(f)
 return {}


def save_checkpoint(model_key, comment_dim, data):
 path = Path("results") / f"{model_key}_{comment_dim}_comment_checkpoint.json"
 with open(path, "w", encoding="utf-8") as f:
 json.dump(data, f, ensure_ascii=False, indent=2)


def main():
 parser = argparse.ArgumentParser(
 description="Evaluate LLM comments via DeepSeek pairwise comparison"
 )
 parser.add_argument(
 "--gt", type=str, default="essay_total_data_4500_260417.xlsx",
 help="Ground truth Excel file"
 )
 parser.add_argument(
 "--model", type=str, default=None,
 help="Specific model to evaluate (e.g., gpt-4o). If not set, evaluates all."
 )
 parser.add_argument(
 "--limit", type=int, default=None,
 help="Limit number of essays to compare (for testing)."
 )
 parser.add_argument(
 "--resume", action="store_true",
 help="Resume from checkpoint."
 )
 args = parser.parse_args()

 ensure_output_dir()

 print(f"Loading data from {args.gt}...")
 gt_df = pd.read_excel(args.gt)

 models_to_eval = [args.model] if args.model else list(MODELS.keys())

 # Init DeepSeek client
 deepseek_client = OpenAI(
 base_url=DEEPSEEK_EVAL_CONFIG["base_url"],
 api_key=DEEPSEEK_EVAL_CONFIG["api_key"],
 timeout=REQUEST_TIMEOUT,
 )

 all_results = []

 for model_key in models_to_eval:
 score_path = f"results/{model_key}_scores.xlsx"
 if not Path(score_path).exists():
 print(f"[SKIP] {score_path} not found")
 continue

 print(f"\n{'='*60}")
 print(f"Evaluating comments for {model_key}")
 print(f"{'='*60}")

 model_df = pd.read_excel(score_path)

 min_len = min(len(gt_df), len(model_df))
 gt_df_aligned = gt_df.iloc[:min_len].reset_index(drop=True)
 model_df_aligned = model_df.iloc[:min_len].reset_index(drop=True)

 if args.limit:
 gt_df_aligned = gt_df_aligned.head(args.limit)
 model_df_aligned = model_df_aligned.head(args.limit)

 for comment_col in COMMENT_COLS:
 print(f"\n Dimension: {comment_col}")
 checkpoint = (
 load_checkpoint(model_key, comment_col) if args.resume else {}
 )

 for i in tqdm(range(len(gt_df_aligned)), desc=f"{model_key}/{comment_col}"):
 essay_id = i

 if str(essay_id) in checkpoint:
 continue

 gt_comment = (
 str(gt_df_aligned.iloc[i][comment_col])
 if pd.notna(gt_df_aligned.iloc[i][comment_col]) else ""
 )
 llm_comment = (
 str(model_df_aligned.iloc[i][comment_col])
 if pd.notna(model_df_aligned.iloc[i][comment_col]) else ""
 )

 winner = compare_comments(
 deepseek_client,
 DEEPSEEK_EVAL_CONFIG["model_name"],
 llm_comment,
 gt_comment
 )

 checkpoint[str(essay_id)] = {
 "winner": winner,
 "llm_comment": llm_comment[:200],
 "gt_comment": gt_comment[:200],
 }

 if (i + 1) % 20 == 0:
 save_checkpoint(model_key, comment_col, checkpoint)

 save_checkpoint(model_key, comment_col, checkpoint)

 # Aggregate results for this model x dimension
 winners = [v["winner"] for v in checkpoint.values()]
 winners_arr = np.array(winners)
 n_total = len(winners_arr)
 n_win = int((winners_arr == 1).sum())
 n_tie = int((winners_arr == 0).sum())
 n_loss = int((winners_arr == -1).sum())
 win_rate = n_win / n_total if n_total > 0 else 0

 print(f" Results: win={n_win}, tie={n_tie}, loss={n_loss}, total={n_total}")
 print(f" Win rate: {win_rate:.2%}")

 all_results.append({
 "model": model_key,
 "dimension": comment_col,
 "n_total": n_total,
 "n_win": n_win,
 "n_tie": n_tie,
 "n_loss": n_loss,
 "win_rate": win_rate,
 })

 # Clean up checkpoints
 for col in COMMENT_COLS:
 ckpt_path = Path("results") / f"{model_key}_{col}_comment_checkpoint.json"
 if ckpt_path.exists():
 ckpt_path.unlink()

 if not all_results:
 print("No results.")
 return

 result_df = pd.DataFrame(all_results)
 result_df = result_df[[
 "model", "dimension", "n_total", "n_win", "n_tie", "n_loss", "win_rate"
 ]]

 Path("results/comment_evaluation.xlsx").parent.mkdir(parents=True, exist_ok=True)
 result_df.to_excel("results/comment_evaluation.xlsx", index=False)
 print(f"\nComment evaluation saved to results/comment_evaluation.xlsx")

 print(f"\nWin rate summary:")
 pivot = result_df.pivot_table(values="win_rate", index="dimension", columns="model")
 print(pivot.round(4).to_string())


if __name__ == "__main__":
 main()
