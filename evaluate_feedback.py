# -*- coding: utf-8 -*-
"""
Evaluate LLM-generated feedback quality using dual-LLM pairwise comparison
and auxiliary similarity metrics (BERTScore, sentence embedding cosine similarity).

Two independent LLM judges (deepseek + gemini) compare model feedback vs expert
reference feedback for each essay across three dimensions (relevance, content,
expression). A ternary preference label (+1/0/-1) is assigned per judge.

Usage:
 python evaluate_feedback.py
 python evaluate_feedback.py --model gpt-5.4
 python evaluate_feedback.py --limit 5
 python evaluate_feedback.py --resume
 python evaluate_feedback.py --no_similarity
"""

import os
import json
import time
import argparse
import re
from pathlib import Path

# Force transformers offline mode to avoid repeated HuggingFace network checks
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI

from feedback_config import MODELS, MAX_RETRIES, RETRY_DELAY, REQUEST_TIMEOUT
from prompts import FEEDBACK_COMPARE_PROMPT

JUDGE_KEYS = list(MODELS.keys())  # deepseek, gemini_3_1_pro
COMMENT_COLS = ["relevance_comment", "content_comment", "expression_comment"]

# Model paths (downloaded from HuggingFace, cached locally)
BERTSCORE_MODEL = "bert-base-chinese"
SENTENCE_ENCODER_MODEL = "model/chinese-roberta-wwm-ext"


def build_compare_messages(comment_a, comment_b):
    """Build messages for pairwise comparison."""
    user_content = FEEDBACK_COMPARE_PROMPT.format(
        comment_a=comment_a,
        comment_b=comment_b
    )
    return [{"role": "user", "content": user_content}]


def parse_compare_response(text):
    """Parse winner value from LLM comparison response."""
    text = text.strip()
    try:
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            data = json.loads(match.group())
            winner = int(data.get("winner", 0))
            if winner not in (-1, 0, 1):
                return 0
            return winner
    except (json.JSONDecodeError, ValueError, KeyError):
        pass
    m = re.search(r'"winner"\s*:\s*(-?\d+)', text)
    if m:
        v = int(m.group(1))
        return v if v in (-1, 0, 1) else 0
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
                print(f"    Rate limit hit, waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    API error (non-retryable): {e}")
                return None
    print("    Max retries exceeded.")
    return None


def compare_feedback(client, model_name, comment_a, comment_b):
    """Compare two feedback comments, return +1/0/-1."""
    messages = build_compare_messages(comment_a, comment_b)
    response = call_llm(client, model_name, messages)
    if response is None:
        return 0
    return parse_compare_response(response)


# ============================================================
# Auxiliary similarity metrics
# ============================================================
_bertscore_model = None
_sentence_encoder = None


def get_bertscore_f1(text_a, text_b):
    """Compute BERTScore F1 between two Chinese texts."""
    P, R, F1 = _bertscore_model(
        [text_a], [text_b],
        model_type=SENTENCE_ENCODER_MODEL,
        num_layers=9,
        verbose=False,
        rescale_with_baseline=False,
    )
    return F1.item()


def get_cosine_similarity(text_a, text_b):
    """Compute sentence embedding cosine similarity between two Chinese texts."""
    emb_a = _sentence_encoder.encode([text_a], normalize_embeddings=True)
    emb_b = _sentence_encoder.encode([text_b], normalize_embeddings=True)
    return float(np.dot(emb_a[0], emb_b[0]))


def ensure_output_dir():
    os.makedirs("results", exist_ok=True)


def load_checkpoint(model_key, comment_dim):
    path = Path("results") / f"{model_key}_{comment_dim}_feedback_checkpoint.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(model_key, comment_dim, data):
    path = Path("results") / f"{model_key}_{comment_dim}_feedback_checkpoint.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM feedback quality via dual-LLM pairwise comparison"
    )
    parser.add_argument(
        "--gt", type=str, default="essay_total_data_4500_260417.xlsx",
        help="Ground truth Excel file"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Specific model to evaluate (e.g., gpt-5.4). If not set, evaluates all."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of essays to compare (for testing)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint."
    )
    parser.add_argument(
        "--no_similarity", action="store_true",
        help="Skip BERTScore and cosine similarity computation."
    )
    parser.add_argument(
        "--bert_low", type=float, default=0.55,
        help="BERTScore F1 lower threshold. Below this -> loss (-1). Default 0.55."
    )
    parser.add_argument(
        "--bert_high", type=float, default=0.70,
        help="BERTScore F1 upper threshold. Above this -> win (+1). Default 0.70."
    )
    parser.add_argument(
        "--cos_low", type=float, default=0.60,
        help="Cosine sim lower threshold. Below this -> loss (-1). Default 0.60."
    )
    parser.add_argument(
        "--cos_high", type=float, default=0.75,
        help="Cosine sim upper threshold. Above this -> win (+1). Default 0.75."
    )
    args = parser.parse_args()

    ensure_output_dir()

    print(f"Loading expert data from {args.gt}...")
    gt_df = pd.read_excel(args.gt)

    models_to_eval = [args.model] if args.model else list(MODELS.keys())

    # Init judge clients
    judges = {}
    for jk in JUDGE_KEYS:
        cfg = MODELS[jk]
        judges[jk] = OpenAI(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            timeout=REQUEST_TIMEOUT,
        )
        print(f"Judge initialized: {jk} ({cfg['model_name']})")

    # Preload auxiliary models if needed
    if not args.no_similarity:
        print("Loading sentence embedding model...")
        from sentence_transformers import SentenceTransformer
        global _sentence_encoder
        _sentence_encoder = SentenceTransformer(
            SENTENCE_ENCODER_MODEL,
            device="cpu"
        )
        print("Sentence embedding model loaded.")

        print("Loading BERTScore model (offline mode)...")
        from bert_score import score
        global _bertscore_model
        _bertscore_model = score
        # Pre-warm: run one dummy call to trigger model loading into cache
        _ = get_bertscore_f1("测试", "测试")
        print("BERTScore model loaded.")

    all_results = []

    for model_key in models_to_eval:
        score_path = f"results/{model_key}_scores.xlsx"
        if not Path(score_path).exists():
            print(f"[SKIP] {score_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating feedback for {model_key}")
        print(f"{'='*60}")

        model_df = pd.read_excel(score_path)

        min_len = min(len(gt_df), len(model_df))
        gt_aligned = gt_df.iloc[:min_len].reset_index(drop=True)
        model_aligned = model_df.iloc[:min_len].reset_index(drop=True)

        if args.limit:
            gt_aligned = gt_aligned.head(args.limit)
            model_aligned = model_aligned.head(args.limit)

        for comment_col in COMMENT_COLS:
            print(f"\n  Dimension: {comment_col}")
            checkpoint = (
                load_checkpoint(model_key, comment_col) if args.resume else {}
            )

            for i in tqdm(range(len(gt_aligned)), desc=f"{model_key}/{comment_col}"):
                essay_id = i

                if str(essay_id) in checkpoint:
                    continue

                gt_feedback = (
                    str(gt_aligned.iloc[i][comment_col])
                    if pd.notna(gt_aligned.iloc[i][comment_col]) else ""
                )
                model_feedback = (
                    str(model_aligned.iloc[i][comment_col])
                    if pd.notna(model_aligned.iloc[i][comment_col]) else ""
                )

                # Dual-LLM pairwise comparison
                judge_votes = {}
                for jk in JUDGE_KEYS:
                    cfg = MODELS[jk]
                    vote = compare_feedback(
                        judges[jk],
                        cfg["model_name"],
                        model_feedback,
                        gt_feedback
                    )
                    judge_votes[jk] = vote

                # Average S_total per sample = mean of two judges
                avg_vote = np.mean(list(judge_votes.values()))

                # Auxiliary similarity metrics
                bert_f1 = 0.0
                cos_sim = 0.0
                if not args.no_similarity and gt_feedback and model_feedback:
                    try:
                        bert_f1 = get_bertscore_f1(model_feedback, gt_feedback)
                        cos_sim = get_cosine_similarity(model_feedback, gt_feedback)
                    except Exception as e:
                        print(f"    Similarity error: {e}")

                checkpoint[str(essay_id)] = {
                    "deepseek_vote": judge_votes.get("deepseek", 0),
                    "gemini_vote": judge_votes.get("gemini_3_1_pro", 0),
                    "avg_vote": round(avg_vote, 4),
                    "bert_f1": round(bert_f1, 4),
                    "cos_sim": round(cos_sim, 4),
                    "model_fb": model_feedback[:200],
                    "expert_fb": gt_feedback[:200],
                }

                if (i + 1) % 20 == 0:
                    save_checkpoint(model_key, comment_col, checkpoint)

            save_checkpoint(model_key, comment_col, checkpoint)

            # Aggregate results
            deepseek_votes = [v["deepseek_vote"] for v in checkpoint.values()]
            gemini_votes = [v["gemini_vote"] for v in checkpoint.values()]
            bert_f1s = [v["bert_f1"] for v in checkpoint.values()]
            cos_sims = [v["cos_sim"] for v in checkpoint.values()]

            ds_arr = np.array(deepseek_votes)
            gm_arr = np.array(gemini_votes)
            n_total = len(ds_arr)

            # Per-judge win/tie/loss
            ds_win = int((ds_arr == 1).sum())
            ds_tie = int((ds_arr == 0).sum())
            ds_loss = int((ds_arr == -1).sum())
            gm_win = int((gm_arr == 1).sum())
            gm_tie = int((gm_arr == 0).sum())
            gm_loss = int((gm_arr == -1).sum())

            # S_total averaged across two judges
            s_total_ds = ds_win - ds_loss
            s_total_gm = gm_win - gm_loss
            s_total_avg = (s_total_ds + s_total_gm) / 2.0

            # Win rate (fraction where avg_vote > 0)
            avg_votes = [v["avg_vote"] for v in checkpoint.values()]
            win_count = sum(1 for v in avg_votes if v > 0.5)
            tie_count = sum(1 for v in avg_votes if -0.5 <= v <= 0.5)
            loss_count = sum(1 for v in avg_votes if v < -0.5)

            # ============================================================
            # Convert similarity scores to ternary labels (+1/0/-1)
            # Fixed thresholds (not percentile-based):
            #   score > high  -> +1 (close enough to expert = win)
            #   low < score < high -> 0 (moderate similarity = tie)
            #   score < low -> -1 (too different from expert = loss)
            # ============================================================
            bert_low, bert_high = args.bert_low, args.bert_high
            cos_low, cos_high = args.cos_low, args.cos_high

            bert_labels = []
            cos_labels = []
            for bf, cs in zip(bert_f1s, cos_sims):
                bert_labels.append(1 if bf > bert_high else (-1 if bf < bert_low else 0))
                cos_labels.append(1 if cs > cos_high else (-1 if cs < cos_low else 0))

            bl_arr = np.array(bert_labels)
            cl_arr = np.array(cos_labels)
            bert_win = int((bl_arr == 1).sum())
            bert_tie = int((bl_arr == 0).sum())
            bert_loss = int((bl_arr == -1).sum())
            cos_win = int((cl_arr == 1).sum())
            cos_tie = int((cl_arr == 0).sum())
            cos_loss = int((cl_arr == -1).sum())
            bert_S_total = bert_win - bert_loss
            cos_S_total = cos_win - cos_loss

            print(f"    DeepSeek: win={ds_win}, tie={ds_tie}, loss={ds_loss}, S_total={s_total_ds}")
            print(f"    Gemini:   win={gm_win}, tie={gm_tie}, loss={gm_loss}, S_total={s_total_gm}")
            print(f"    Averaged: win={win_count}, tie={tie_count}, loss={loss_count}, S_total={s_total_avg:.1f}")
            print(f"    BERTScore: win={bert_win}, tie={bert_tie}, loss={bert_loss}, S_total={bert_S_total}")
            print(f"      (thresholds: low={bert_low:.4f}, high={bert_high:.4f})")
            print(f"    Cosine sim:  win={cos_win}, tie={cos_tie}, loss={cos_loss}, S_total={cos_S_total}")
            print(f"      (thresholds: low={cos_low:.4f}, high={cos_high:.4f})")
            print(f"    BERTScore F1 (raw mean): {np.mean(bert_f1s):.4f}")
            print(f"    Cosine sim (raw mean):   {np.mean(cos_sims):.4f}")

            all_results.append({
                "model": model_key,
                "dimension": comment_col,
                "n_total": n_total,
                "deepseek_win": ds_win,
                "deepseek_tie": ds_tie,
                "deepseek_loss": ds_loss,
                "deepseek_S_total": s_total_ds,
                "gemini_win": gm_win,
                "gemini_tie": gm_tie,
                "gemini_loss": gm_loss,
                "gemini_S_total": s_total_gm,
                "avg_win": win_count,
                "avg_tie": tie_count,
                "avg_loss": loss_count,
                "avg_S_total": round(s_total_avg, 2),
                "bert_win": bert_win,
                "bert_tie": bert_tie,
                "bert_loss": bert_loss,
                "bert_S_total": bert_S_total,
                "bert_low": round(bert_low, 4),
                "bert_high": round(bert_high, 4),
                "cos_win": cos_win,
                "cos_tie": cos_tie,
                "cos_loss": cos_loss,
                "cos_S_total": cos_S_total,
                "cos_low": round(cos_low, 4),
                "cos_high": round(cos_high, 4),
                "bertscore_f1": round(np.mean(bert_f1s), 4),
                "cosine_similarity": round(np.mean(cos_sims), 4),
            })

            # Clean up checkpoint
            ckpt_path = Path("results") / f"{model_key}_{comment_col}_feedback_checkpoint.json"
            if ckpt_path.exists():
                ckpt_path.unlink()

    if not all_results:
        print("No results.")
        return

    result_df = pd.DataFrame(all_results)
    result_df.to_excel("results/grok_4_feedback_evaluation.xlsx", index=False)
    print(f"\n{'='*60}")
    print(f"Feedback evaluation saved to results/feedback_evaluation.xlsx")
    print(f"{'='*60}")

    # Print summary
    print(f"\nS_total summary (averaged across 2 judges):")
    pivot = result_df.pivot_table(values="avg_S_total", index="dimension", columns="model")
    print(pivot.round(2).to_string())

    print(f"\nBERTScore ternary S_total summary:")
    pivot_bert_t = result_df.pivot_table(values="bert_S_total", index="dimension", columns="model")
    print(pivot_bert_t.round(2).to_string())

    print(f"\nCosine sim ternary S_total summary:")
    pivot_cos_t = result_df.pivot_table(values="cos_S_total", index="dimension", columns="model")
    print(pivot_cos_t.round(2).to_string())

    print(f"\nBERTScore F1 (raw mean) summary:")
    pivot_bert = result_df.pivot_table(values="bertscore_f1", index="dimension", columns="model")
    print(pivot_bert.round(4).to_string())

    print(f"\nCosine similarity (raw mean) summary:")
    pivot_cos = result_df.pivot_table(values="cosine_similarity", index="dimension", columns="model")
    print(pivot_cos.round(4).to_string())


if __name__ == "__main__":
    main()
