# -*- coding: utf-8 -*-
"""
RoBERTa cross-prompt leave-one-prompt-out experiment.

For each of the 14 prompts:
  - Train RoBERTa ordinal regression on the other 13 prompts
  - Evaluate on the held-out prompt
  - Report ACC, QWK, PCC per dimension

Finally: compute averages across all 14 prompts.

Usage:
    python roberta_cross_prompt.py
    python roberta_cross_prompt.py --epochs 5 --batch_size 32
    python roberta_cross_prompt.py --prompt "故事新编"  # single prompt test
"""

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW

from evaluate_scores import compute_metrics, SCORE_COLS

NUM_THRESHOLDS = 8
SCORE_DIMS = ["total_score", "relevance_score", "content_score", "expression_score"]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EssayDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512, labels=None):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_length,
            padding="max_length", return_tensors="pt",
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class OrdinalRegressionRoBERTa(nn.Module):
    def __init__(self, model_name, num_dimensions=4, num_thresholds=NUM_THRESHOLDS, dropout=0.1):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        hidden = self.roberta.config.hidden_size
        self.heads = nn.ModuleList([
            nn.Linear(hidden, num_thresholds) for _ in range(num_dimensions)
        ])

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        all_logits = [head(pooled) for head in self.heads]
        if labels is not None:
            loss = self.ordinal_loss(all_logits, labels)
            return loss, all_logits
        return all_logits

    @staticmethod
    def ordinal_loss(logits_list, labels):
        total_loss = 0
        thresholds = torch.arange(1, NUM_THRESHOLDS + 1, device=labels.device)
        for dim_logits, dim_labels in zip(logits_list, labels.unbind(dim=1)):
            targets = (dim_labels.unsqueeze(1) >= thresholds.unsqueeze(0)).float()
            total_loss += F.binary_cross_entropy_with_logits(dim_logits, targets)
        return total_loss / len(logits_list)

    @staticmethod
    def logits_to_scores(all_logits):
        predictions = []
        for dim_logits in all_logits:
            probs = torch.sigmoid(dim_logits)
            pred = probs.sum(dim=1)
            pred = torch.round(pred).clamp(0, NUM_THRESHOLDS).long()
            predictions.append(pred)
        return predictions


def build_texts(df):
    texts = []
    for _, row in df.iterrows():
        hw_req = str(row["hw_requirement"]) if pd.notna(row["hw_requirement"]) else ""
        stu_title = str(row["stu_title"]) if pd.notna(row["stu_title"]) else ""
        content = str(row["content"]) if pd.notna(row["content"]) else ""
        texts.append(f"{hw_req} [SEP] {stu_title} [SEP] {content}")
    return texts


def build_labels(df):
    labels = df[SCORE_DIMS].values.astype(np.int64)
    return torch.tensor(labels, dtype=torch.long)


def get_optimizer_param_groups(model, lr, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return AdamW(grouped, lr=lr)


def train_one_epoch(model, loader, optimizer, scheduler, device, desc="  Training"):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    pbar.close()
    return total_loss / len(loader)


@torch.no_grad()
def predict_on_data(model, texts, tokenizer, batch_size, max_length, device):
    model.eval()
    dataset = EssayDataset(texts, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    for batch in tqdm(loader, desc="  Predicting", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = OrdinalRegressionRoBERTa.logits_to_scores(logits)
        stacked = torch.stack(preds, dim=1)
        all_preds.append(stacked.cpu())
    return torch.cat(all_preds, dim=0)


def main():
    parser = argparse.ArgumentParser(description="RoBERTa cross-prompt experiment")
    parser.add_argument("--data", type=str, default="essay_total_data_4500_260417.xlsx")
    parser.add_argument("--model", type=str, default="model/chinese-roberta-wwm-ext")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--prompt", type=str, nargs="+", default=None,
                        help="Specific prompts to test (default: all 14)")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_excel(args.data)
    print(f"Data: {len(df)} rows, {df['hw_title'].nunique()} prompts")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Determine prompts to test
    all_prompts = sorted(df["hw_title"].unique())
    test_prompts = args.prompt if args.prompt else all_prompts
    if isinstance(test_prompts, str):
        test_prompts = [test_prompts]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Texts and labels for entire dataset
    texts = build_texts(df)
    labels = build_labels(df)

    # Per-prompt results
    per_prompt_results = {}  # {prompt: {dim: {acc, qwk, pcc}}}
    all_preds_list = []  # (prompt, indices, pred_tensor, label_tensor)

    for fold_idx, test_prompt in enumerate(tqdm(test_prompts, desc="Overall progress", unit="prompt")):
        print(f"\n{'='*60}")
        print(f"[{fold_idx+1}/{len(test_prompts)}] Testing on: '{test_prompt}'")
        n_test = (df["hw_title"] == test_prompt).sum()
        n_train = len(df) - n_test
        print(f"  Train: {n_train}, Test: {n_test}")
        print(f"{'='*60}")

        train_mask = df["hw_title"] != test_prompt
        test_mask = df["hw_title"] == test_prompt

        train_texts = [texts[i] for i in range(len(texts)) if train_mask[i]]
        train_labels = labels[train_mask]
        test_texts = [texts[i] for i in range(len(texts)) if test_mask[i]]
        test_labels = labels[test_mask]

        train_dataset = EssayDataset(train_texts, tokenizer, args.max_length, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # Build fresh model
        model = OrdinalRegressionRoBERTa(args.model).to(device)
        optimizer = get_optimizer_param_groups(model, args.lr, args.weight_decay)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps,
        )

        # Train
        epoch_pbar = tqdm(range(args.epochs), desc=f"  Epochs ({test_prompt[:15]})", unit="epoch")
        for epoch in epoch_pbar:
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device,
                                          desc=f"  Epoch {epoch+1}/{args.epochs}")
            if (epoch + 1) % 3 == 0 or epoch == 0:
                epoch_pbar.set_postfix({"loss": f"{train_loss:.4f}"})
        epoch_pbar.close()

        # Predict
        test_preds = predict_on_data(
            model, test_texts, tokenizer, args.batch_size, args.max_length, device
        )

        # Compute per-dimension metrics
        prompt_metrics = {}
        print(f"\n  [{test_prompt}] Metrics:")
        for d_idx, dim in enumerate(SCORE_DIMS):
            pred_d = test_preds[:, d_idx].numpy()
            truth_d = test_labels[:, d_idx].numpy()
            m = compute_metrics(pred_d, truth_d)
            prompt_metrics[dim] = {
                "acc": m["acc"],
                "qwk": m["qwk"],
                "pcc": m["pcc"],
                "n_valid": m["n_valid"],
            }
            acc_s = f"{m['acc']:.4f}" if m["acc"] is not None else "N/A"
            qwk_s = f"{m['qwk']:.4f}" if m["qwk"] is not None else "N/A"
            pcc_s = f"{m['pcc']:.4f}" if m["pcc"] is not None else "N/A"
            print(f"    {dim}: ACC={acc_s} QWK={qwk_s} PCC={pcc_s}")

        per_prompt_results[test_prompt] = prompt_metrics

        # Store for final output
        test_indices = np.where(test_mask)[0]
        all_preds_list.append((test_prompt, test_indices, test_preds, test_labels))

    # ============================================================
    # Summary: per-prompt + average
    # ============================================================
    print(f"\n{'='*70}")
    print(" SUMMARY — RoBERTa Cross-Prompt Results")
    print(f"{'='*70}")

    # Per-prompt table
    print(f"\n{'Prompt':<20}", end="")
    for dim in SCORE_DIMS:
        print(f" {dim[:8]:>10} QWK", end="")
    print()
    print(f"{'-'*62}")

    for prompt in test_prompts:
        m = per_prompt_results[prompt]
        print(f"{prompt:<20}", end="")
        for dim in SCORE_DIMS:
            qwk = m[dim]["qwk"]
            qwk_s = f"{qwk:.4f}" if qwk is not None else "N/A"
            print(f" {qwk_s:>10}", end="")
        print()

    # Average across prompts
    print(f"\n{'Average':<20}", end="")
    avg_qwk = {}
    avg_acc = {}
    avg_pcc = {}
    for dim in SCORE_DIMS:
        qwks = [per_prompt_results[p][dim]["qwk"] for p in test_prompts
                if per_prompt_results[p][dim]["qwk"] is not None]
        accs = [per_prompt_results[p][dim]["acc"] for p in test_prompts
                if per_prompt_results[p][dim]["acc"] is not None]
        pccs = [per_prompt_results[p][dim]["pcc"] for p in test_prompts
                if per_prompt_results[p][dim]["pcc"] is not None]
        avg_qwk[dim] = sum(qwks) / len(qwks) if qwks else float("nan")
        avg_acc[dim] = sum(accs) / len(accs) if accs else float("nan")
        avg_pcc[dim] = sum(pccs) / len(pccs) if pccs else float("nan")

        print(f" {avg_qwk[dim]:>10.4f}", end="")
    print()

    # Detailed average table
    print(f"\n{'Dimension':<20} {'ACC_avg':>10} {'QWK_avg':>10} {'PCC_avg':>10}")
    print(f"{'-'*52}")
    for dim in SCORE_DIMS:
        print(f"{dim:<20} {avg_acc[dim]:>10.4f} {avg_qwk[dim]:>10.4f} {avg_pcc[dim]:>10.4f}")

    # ============================================================
    # Save results
    # ============================================================
    # 1. Per-prompt metrics JSON
    import json
    metrics_path = output_dir / "roberta_cross_prompt_per_prompt.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(per_prompt_results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved per-prompt metrics to {metrics_path}")

    # 2. Summary Excel (per-prompt + average)
    summary_rows = []
    for prompt in test_prompts:
        for dim in SCORE_DIMS:
            m = per_prompt_results[prompt][dim]
            summary_rows.append({
                "prompt": prompt,
                "dimension": dim,
                "ACC": m["acc"],
                "QWK": m["qwk"],
                "PCC": m["pcc"],
                "n_valid": m["n_valid"],
            })
    # Add average rows
    for dim in SCORE_DIMS:
        summary_rows.append({
            "prompt": "AVERAGE",
            "dimension": dim,
            "ACC": avg_acc[dim],
            "QWK": avg_qwk[dim],
            "PCC": avg_pcc[dim],
            "n_valid": sum(per_prompt_results[p][dim]["n_valid"] for p in test_prompts),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "roberta_cross_prompt_summary.xlsx"
    summary_df.to_excel(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # 3. Predictions Excel (compatible with evaluate_scores.py)
    result_df = df.copy()
    result_df["total_score"] = -1
    result_df["relevance_score"] = -1
    result_df["content_score"] = -1
    result_df["expression_score"] = -1

    for prompt, indices, preds, _ in all_preds_list:
        for d_idx, dim in enumerate(SCORE_DIMS):
            for i, idx in enumerate(indices):
                result_df.at[idx, dim] = preds[i, d_idx].item()

    col_order = [
        "hw_title", "hw_requirement", "stu_title", "class_id", "content",
        "total_score", "relevance_score", "relevance_comment",
        "content_score", "content_comment",
        "expression_score", "expression_comment",
    ]
    for col in col_order:
        if col not in result_df.columns:
            result_df[col] = ""
    result_df = result_df[col_order]

    pred_path = output_dir / "roberta_ordinal_cross_prompt_scores.xlsx"
    result_df.to_excel(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")


if __name__ == "__main__":
    main()
