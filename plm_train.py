# -*- coding: utf-8 -*-
"""
Train Chinese RoBERTa with ordinal regression on NarrativeBench essays.

Two evaluation modes:
  - within_prompt: 5-fold stratified CV (out-of-fold predictions, fair vs LLM zero-shot)
  - cross_prompt: leave-one-assignment-type-out (tests generalization across writing tasks)

Usage:
    python plm_train.py --mode within_prompt
    python plm_train.py --mode cross_prompt
    python plm_train.py --mode both
    python plm_train.py --mode within_prompt --epochs 5 --batch_size 32
"""

import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup

from evaluate_scores import compute_metrics, SCORE_COLS

NUM_THRESHOLDS = 8  # score >= k for k=1..8
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
        self.labels = labels  # tensor [N, 4]

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: self.encodings[key][idx] for key in self.encodings}
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


class OrdinalRegressionRoBERTa(nn.Module):
    """RoBERTa backbone + 4 ordinal regression heads (one per score dimension)."""

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
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS]
        pooled = self.dropout(pooled)

        all_logits = [head(pooled) for head in self.heads]

        if labels is not None:
            loss = self.ordinal_loss(all_logits, labels)
            return loss, all_logits
        return all_logits

    @staticmethod
    def ordinal_loss(logits_list, labels):
        """BCE on cumulative thresholds. labels: [batch, 4], logits: list of 4 [batch, 8]."""
        total_loss = 0
        thresholds = torch.arange(1, NUM_THRESHOLDS + 1, device=labels.device)
        for dim_logits, dim_labels in zip(logits_list, labels.unbind(dim=1)):
            # target[k] = 1 if score >= k+1, else 0
            targets = (dim_labels.unsqueeze(1) >= thresholds.unsqueeze(0)).float()
            total_loss += F.binary_cross_entropy_with_logits(dim_logits, targets)
        return total_loss / len(logits_list)

    @staticmethod
    def logits_to_scores(all_logits):
        """Convert logits to integer scores 0-8."""
        predictions = []
        for dim_logits in all_logits:
            probs = torch.sigmoid(dim_logits)  # [batch, 8]
            pred = probs.sum(dim=1)  # expected score
            pred = torch.round(pred).clamp(0, NUM_THRESHOLDS).long()
            predictions.append(pred)
        return predictions


def build_texts(df):
    """Combine hw_requirement + stu_title + content into single strings."""
    texts = []
    for _, row in df.iterrows():
        hw_req = str(row["hw_requirement"]) if pd.notna(row["hw_requirement"]) else ""
        stu_title = str(row["stu_title"]) if pd.notna(row["stu_title"]) else ""
        content = str(row["content"]) if pd.notna(row["content"]) else ""
        texts.append(f"{hw_req} [SEP] {stu_title} [SEP] {content}")
    return texts


def build_labels(df):
    """Extract score labels as tensor [N, 4]."""
    labels = df[SCORE_DIMS].values.astype(np.int64)
    return torch.tensor(labels, dtype=torch.long)


def get_optimizer_param_groups(model, lr, weight_decay):
    """Separate weight decay for bias/layernorm."""
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return AdamW(grouped, lr=lr)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        loss, _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = OrdinalRegressionRoBERTa.logits_to_scores(logits)
        all_labels.append(labels.cpu())
        stacked = torch.stack(preds, dim=1)  # [batch, 4]
        all_preds.append(stacked)
    labels = torch.cat(all_labels, dim=0)  # [N, 4]
    preds = torch.cat(all_preds, dim=0)    # [N, 4]
    return labels, preds


def predict_on_data(model, texts, tokenizer, batch_size=32, max_length=512, device=None):
    """Generate predictions for arbitrary texts."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = EssayDataset(texts, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    for batch in tqdm(loader, desc="Predicting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = OrdinalRegressionRoBERTa.logits_to_scores(logits)
        stacked = torch.stack(preds, dim=1)
        all_preds.append(stacked.cpu())
    return torch.cat(all_preds, dim=0)  # [N, 4]


def run_within_prompt_cv(df, tokenizer, model_name, args):
    """5-fold stratified CV. Returns out-of-fold predictions DataFrame."""
    print("=" * 60)
    print("Within-Prompt 5-Fold Cross-Validation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Stratified folds by hw_title
    df_copy = df.copy()
    df_copy["_label"] = df_copy["hw_title"]
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = np.zeros(len(df_copy), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(skf.split(df_copy, df_copy["_label"])):
        folds[val_idx] = fold_idx
    df_copy["_fold"] = folds

    texts = build_texts(df_copy)
    labels = build_labels(df_copy)

    # Placeholder for OOF predictions
    oof_preds = torch.zeros((len(df_copy), 4), dtype=torch.long)

    for fold in range(args.n_folds):
        print(f"\n--- Fold {fold + 1}/{args.n_folds} ---")
        train_mask = df_copy["_fold"] != fold
        val_mask = df_copy["_fold"] == fold

        train_texts = [texts[i] for i in range(len(texts)) if train_mask[i]]
        train_labels = labels[train_mask]
        val_texts = [texts[i] for i in range(len(texts)) if val_mask[i]]
        val_labels = labels[val_mask]

        train_dataset = EssayDataset(train_texts, tokenizer, args.max_length, train_labels)
        val_dataset = EssayDataset(val_texts, tokenizer, args.max_length, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2, shuffle=False)

        # Build fresh model for this fold
        model = OrdinalRegressionRoBERTa(model_name).to(device)
        optimizer = get_optimizer_param_groups(model, args.lr, args.weight_decay)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps,
        )

        # Training
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)

            # Validation
            model.eval()
            val_loss_accum = 0.0
            val_n = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids_b = batch["input_ids"].to(device)
                    attention_mask_b = batch["attention_mask"].to(device)
                    labels_b = batch["labels"].to(device)
                    loss_b, _ = model(input_ids=input_ids_b, attention_mask=attention_mask_b, labels=labels_b)
                    val_loss_accum += loss_b.item() * input_ids_b.size(0)
                    val_n += input_ids_b.size(0)
            val_loss = val_loss_accum / val_n
            model.train()  # switch back for next epoch
            print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        # Load best model and predict on validation fold
        model.load_state_dict(best_state)
        fold_preds = predict_on_data(model, val_texts, tokenizer, args.batch_size, args.max_length, device)

        oof_indices = np.where(val_mask)[0]
        oof_preds[oof_indices] = fold_preds

        # Print fold metrics
        for d_idx, dim in enumerate(SCORE_DIMS):
            pred_d = oof_preds[oof_indices, d_idx].numpy()
            truth_d = labels[oof_indices, d_idx].numpy()
            m = compute_metrics(pred_d, truth_d)
            print(f"  Fold {fold+1} {dim}: ACC={m['acc']:.4f} QWK={m['qwk']:.4f} PCC={m['pcc']:.4f}")

    # Aggregate OOF metrics
    print(f"\n--- Overall OOF Metrics ---")
    metrics_records = []
    for d_idx, dim in enumerate(SCORE_DIMS):
        m = compute_metrics(oof_preds[:, d_idx].numpy(), labels[:, d_idx].numpy())
        print(f"  {dim}: ACC={m['acc']:.4f} QWK={m['qwk']:.4f} PCC={m['pcc']:.4f} N={m['n_valid']}")
        metrics_records.append({
            "mode": "within_prompt",
            "dimension": dim,
            "ACC": round(m["acc"], 4) if m["acc"] is not None else None,
            "QWK": round(m["qwk"], 4) if m["qwk"] is not None else None,
            "PCC": round(m["pcc"], 4) if m["pcc"] is not None else None,
            "n_valid": m["n_valid"],
        })

    metrics_df = pd.DataFrame(metrics_records)

    # Build output DataFrame
    result_df = df_copy.copy()
    for d_idx, dim in enumerate(SCORE_DIMS):
        result_df[dim] = oof_preds[:, d_idx].numpy()

    return result_df, metrics_df


def run_cross_prompt(df, tokenizer, model_name, args):
    """Leave-one-assignment-type-out cross-validation."""
    print("=" * 60)
    print("Cross-Prompt Leave-One-Assignment-Type-Out")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    hw_titles = df["hw_title"].unique()
    print(f"Assignment types ({len(hw_titles)}): {sorted(hw_titles)}")

    texts = build_texts(df)
    labels = build_labels(df)
    all_preds = torch.zeros((len(df), 4), dtype=torch.long)

    for hw in tqdm(hw_titles, desc="Cross-prompt types"):
        print(f"\n--- Testing on: '{hw}' ({(df['hw_title'] == hw).sum()} essays) ---")
        train_mask = df["hw_title"] != hw
        test_mask = df["hw_title"] == hw

        train_texts = [texts[i] for i in range(len(texts)) if train_mask[i]]
        train_labels = labels[train_mask]
        test_texts = [texts[i] for i in range(len(texts)) if test_mask[i]]
        test_labels = labels[test_mask]

        train_dataset = EssayDataset(train_texts, tokenizer, args.max_length, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        model = OrdinalRegressionRoBERTa(model_name).to(device)
        optimizer = get_optimizer_param_groups(model, args.lr, args.weight_decay)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps,
        )

        # Train without validation split (use last epoch model)
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
            if (epoch + 1) % 3 == 0:
                print(f"  Epoch {epoch+1}: loss={train_loss:.4f}")

        # Predict on test
        fold_preds = predict_on_data(model, test_texts, tokenizer, args.batch_size, args.max_length, device)

        test_indices = np.where(test_mask)[0]
        all_preds[test_indices] = fold_preds

        # Print metrics for this type
        for d_idx, dim in enumerate(SCORE_DIMS):
            m = compute_metrics(all_preds[test_indices, d_idx].numpy(), test_labels[:, d_idx].numpy())
            print(f"  {dim}: ACC={m['acc']:.4f} QWK={m['qwk']:.4f} PCC={m['pcc']:.4f}")

    # Overall metrics
    print(f"\n--- Overall Cross-Prompt Metrics ---")
    metrics_records = []
    for d_idx, dim in enumerate(SCORE_DIMS):
        m = compute_metrics(all_preds[:, d_idx].numpy(), labels[:, d_idx].numpy())
        print(f"  {dim}: ACC={m['acc']:.4f} QWK={m['qwk']:.4f} PCC={m['pcc']:.4f} N={m['n_valid']}")
        metrics_records.append({
            "mode": "cross_prompt",
            "dimension": dim,
            "ACC": round(m["acc"], 4) if m["acc"] is not None else None,
            "QWK": round(m["qwk"], 4) if m["qwk"] is not None else None,
            "PCC": round(m["pcc"], 4) if m["pcc"] is not None else None,
            "n_valid": m["n_valid"],
        })

    metrics_df = pd.DataFrame(metrics_records)

    # Build output DataFrame
    result_df = df.copy()
    for d_idx, dim in enumerate(SCORE_DIMS):
        result_df[dim] = all_preds[:, d_idx].numpy()

    return result_df, metrics_df


def save_results(result_df, output_path):
    """Save predictions in format compatible with evaluate_scores.py."""
    col_order = [
        "hw_title", "hw_requirement", "stu_title", "class_id", "content",
        "total_score", "relevance_score", "relevance_comment",
        "content_score", "content_comment",
        "expression_score", "expression_comment",
    ]
    for col in col_order:
        if col not in result_df.columns:
            result_df[col] = ""

    output_df = result_df[col_order]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_excel(output_path, index=False)
    print(f"\nSaved {len(output_df)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train PLM baselines on NarrativeBench")
    parser.add_argument("--data", type=str, default="essay_total_data_4500_260417.xlsx")
    parser.add_argument("--model", type=str, default="model/chinese-roberta-wwm-ext")
    parser.add_argument("--mode", type=str, default="both",
                        choices=["within_prompt", "cross_prompt", "both"])
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    seed_everything(args.seed)

    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_excel(args.data)
    print(f"Data: {len(df)} rows, {df['hw_title'].nunique()} assignment types")

    # Load tokenizer
    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if args.mode in ["within_prompt", "both"]:
        oof_df, metrics_within = run_within_prompt_cv(df, tokenizer, args.model, args)
        save_results(oof_df, f"{args.output_dir}/roberta_ordinal_within_prompt_scores.xlsx")
        metrics_within.to_excel(f"{args.output_dir}/roberta_ordinal_within_prompt_metrics.xlsx", index=False)
        print(f"Saved metrics to {args.output_dir}/roberta_ordinal_within_prompt_metrics.xlsx")

    if args.mode in ["cross_prompt", "both"]:
        cp_df, metrics_cross = run_cross_prompt(df, tokenizer, args.model, args)
        save_results(cp_df, f"{args.output_dir}/roberta_ordinal_cross_prompt_scores.xlsx")
        metrics_cross.to_excel(f"{args.output_dir}/roberta_ordinal_cross_prompt_metrics.xlsx", index=False)
        print(f"Saved metrics to {args.output_dir}/roberta_ordinal_cross_prompt_metrics.xlsx")

    if args.mode == "both":
        # Combined metrics file
        all_metrics = pd.concat([metrics_within, metrics_cross], ignore_index=True)
        all_metrics.to_excel(f"{args.output_dir}/roberta_ordinal_all_metrics.xlsx", index=False)

    print("\nDone. Run evaluate_scores.py to compare with LLMs:")
    print("  python evaluate_scores.py --models roberta_ordinal_within_prompt roberta_ordinal_cross_prompt")


if __name__ == "__main__":
    main()
