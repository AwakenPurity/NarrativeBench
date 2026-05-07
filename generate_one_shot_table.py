# -*- coding: utf-8 -*-
"""
Generate LaTeX performance table from one-shot cross-prompt LLM results.

Reads per-model metrics JSONs from results/cross_prompt_llm/one-shot/
and outputs a LaTeX table with per-prompt + average rows.
"""

import json
import re
from pathlib import Path

SCORE_DIMS = ["relevance_score", "content_score", "expression_score", "total_score"]
DIM_LABELS = {
    "relevance_score": "Relevance",
    "content_score": "Content",
    "expression_score": "Expression",
    "total_score": "Total Score",
}

# Model display config: key -> (display_name, group)
# Groups: "en" = English, "cn" = Chinese
MODEL_CONFIG = {
    "gpt_5_4": ("GPT-5.4", "en"),
    "gemini_3_1_pro": ("Gemini-3.1-pro", "en"),
    "claude_sonnet_4_6": ("Claude-4.6-Sonnet", "en"),
    "llama_4_scout": ("Llama-4-Scout", "en"),
    "grok_4": ("Grok-4", "en"),
    "deepseek": ("DeepSeek-V4-pro", "cn"),
    "qwen3_6_plus": ("Qwen3.6-Plus", "cn"),
    "qwen3_5_27b": ("Qwen3.5-27B", "cn"),
    "qwen3_5_9b": ("Qwen3.5-9B", "cn"),
    "doubao_seed_2_0_pro": ("Doubao-seed-2.0-pro", "cn"),
}

RESULTS_DIR = Path("results/cross_prompt_llm/one-shot")


def load_model_data():
    """Load all model JSON results and return dict of {model_key: {prompt: {dim: metrics}}}."""
    data = {}

    # Primary: _per_bucket_metrics.json files
    for json_file in sorted(RESULTS_DIR.glob("*_per_bucket_metrics.json")):
        stem = json_file.stem
        match = re.match(r"(.+?)_\d+_per_bucket_metrics", stem)
        if not match:
            continue
        model_key = match.group(1)
        if model_key not in MODEL_CONFIG:
            print(f"[SKIP] Unknown model: {model_key}")
            continue
        with open(json_file, "r", encoding="utf-8") as f:
            data[model_key] = json.load(f)

    # Fallback: _per_bucket_checkpoint.json files (same structure, used when experiment didn't finish cleanly)
    for json_file in sorted(RESULTS_DIR.glob("*_per_bucket_checkpoint.json")):
        stem = json_file.stem
        match = re.match(r"(.+?)_\d+_per_bucket_checkpoint", stem)
        if not match:
            continue
        model_key = match.group(1)
        if model_key not in MODEL_CONFIG:
            continue
        if model_key not in data:
            print(f"[FALLBACK] Using checkpoint for {model_key}")
            with open(json_file, "r", encoding="utf-8") as f:
                data[model_key] = json.load(f)

    return data


def collect_all_prompts(model_data):
    """Get sorted list of all prompts across all models."""
    prompts = set()
    for m in model_data.values():
        prompts.update(m.keys())
    return sorted(prompts)


def fmt(v):
    """Format a float to 3 decimal places."""
    if v is None:
        return "N/A"
    return f"{v:.3f}"


def generate_latex_table(model_data):
    """Generate LaTeX tabular rows."""
    prompts = collect_all_prompts(model_data)

    # Build a flat list of rows: (model_key, prompt, dim, acc, pcc, qwk)
    # For the table format shown, each model gets ONE row with averaged metrics.
    # But the user wants per-prompt results too.

    # First: compute per-model per-prompt metrics
    # Then: average across prompts for the "Average" row

    # The example table has one row per model (averaged).
    # The user says "需要每个prompt下的结果", so we need per-prompt rows.

    # Let's generate: for each model, one row per prompt, plus an Average row.

    # But that would be 10 models x 14 prompts = 140 rows, too wide.
    # Re-read: the user provided a table with one row per model.
    # Maybe they want the same table format (one row per model) but with one-shot data.
    # Plus a separate per-prompt table.

    # Let's generate BOTH:
    # 1. Overall table (one row per model, averaged across prompts)
    # 2. Per-prompt breakdown

    # ---- Table 1: Overall (averaged across prompts) ----
    print("=" * 70)
    print("TABLE 1: Overall performance (averaged across prompts)")
    print("=" * 70)

    # Compute per-model averages
    model_avg = {}
    for mk, pdata in model_data.items():
        dim_metrics = {}
        for dim in SCORE_DIMS:
            vals = []
            for p, dims in pdata.items():
                if dim in dims:
                    m = dims[dim]
                    vals.append({
                        "acc": m.get("acc"),
                        "pcc": m.get("pcc"),
                        "qwk": m.get("qwk"),
                    })
            if vals:
                dim_metrics[dim] = {
                    "acc": sum(v["acc"] for v in vals if v["acc"] is not None) /
                           max(sum(1 for v in vals if v["acc"] is not None), 1),
                    "pcc": sum(v["pcc"] for v in vals if v["pcc"] is not None) /
                           max(sum(1 for v in vals if v["pcc"] is not None), 1),
                    "qwk": sum(v["qwk"] for v in vals if v["qwk"] is not None) /
                           max(sum(1 for v in vals if v["qwk"] is not None), 1),
                }
        model_avg[mk] = dim_metrics

    # Collect LLM values for bold/underline (by QWK per dimension)
    # We only bold/underline among LLMs, not pretrained models
    llm_keys_en = [k for k in MODEL_CONFIG if MODEL_CONFIG[k][1] == "en"]
    llm_keys_cn = [k for k in MODEL_CONFIG if MODEL_CONFIG[k][1] == "cn"]
    all_llm_keys = llm_keys_en + llm_keys_cn

    def rank_values(keys, dim, metric):
        """Return (best_key, second_best_key) by metric value among keys."""
        vals = [(k, model_avg.get(k, {}).get(dim, {}).get(metric)) for k in keys]
        vals = [(k, v) for k, v in vals if v is not None]
        vals.sort(key=lambda x: x[1], reverse=True)
        best = vals[0][0] if len(vals) > 0 else None
        second = vals[1][0] if len(vals) > 1 else None
        return best, second

    # For each dimension, find best and second best by QWK among ALL LLMs
    best_qwk = {}
    second_qwk = {}
    for dim in SCORE_DIMS:
        vals = [(k, model_avg.get(k, {}).get(dim, {}).get("qwk")) for k in all_llm_keys]
        vals = [(k, v) for k, v in vals if v is not None]
        vals.sort(key=lambda x: x[1], reverse=True)
        best_qwk[dim] = vals[0][0] if len(vals) > 0 else None
        second_qwk[dim] = vals[1][0] if len(vals) > 1 else None

    # Generate LaTeX
    print("\n\\begin{table*}[htbp]")
    print("\\centering")
    print("\\caption{Performance comparison of English and Chinese LLMs across four dimensions on \\textbf{one-shot scenario}. The final column reports the average QWK across four dimensions. In the table, \\textbf{bold} indicates the best result and \\underline{underlined} values denote the second-best, applied only to LLMs.}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{lccccccccccccc}")
    print("\\toprule")
    print("\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{3}{c}{\\textbf{Relevance}} & \\multicolumn{3}{c}{\\textbf{Content}} & \\multicolumn{3}{c}{\\textbf{Expression}} & \\multicolumn{3}{c}{\\textbf{Total Score}} & \\multirow{2}{*}{\\textbf{Avg.}} \\\\")
    print("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13}")
    print(" & \\textbf{ACC $\\uparrow$} & \\textbf{PCC $\\uparrow$} & \\textbf{QWK $\\uparrow$} & \\textbf{ACC $\\uparrow$} & \\textbf{PCC $\\uparrow$} & \\textbf{QWK $\\uparrow$} & \\textbf{ACC $\\uparrow$} & \\textbf{PCC $\\uparrow$} & \\textbf{QWK $\\uparrow$} & \\textbf{ACC $\\uparrow$} & \\textbf{PCC $\\uparrow$} & \\textbf{QWK $\\uparrow$} \\\\")
    print("\\midrule")

    current_group = None
    for mk in all_llm_keys:
        if mk not in model_avg:
            continue
        display_name, group = MODEL_CONFIG[mk]
        if group != current_group:
            current_group = group
            group_label = "English LLMs" if group == "en" else "Chinese LLMs"
            print(f"\\multicolumn{{14}}{{c}}{{\\textit{{{group_label}}}}} \\\\")

        m = model_avg[mk]
        parts = [display_name]
        for dim in SCORE_DIMS:
            dm = m.get(dim, {})
            acc_s = fmt(dm.get("acc"))
            pcc_s = fmt(dm.get("pcc"))
            qwk_s = fmt(dm.get("qwk"))
            qwk_val = dm.get("qwk")

            # Bold/underline for QWK among LLMs
            acc_prefix = ""
            pcc_prefix = ""
            qwk_prefix = ""
            if mk == best_qwk.get(dim):
                qwk_prefix = "\\textbf{"
            elif mk == second_qwk.get(dim):
                qwk_prefix = "\\underline{"

            if qwk_prefix:
                parts.append(f"{acc_prefix}{acc_s} & {pcc_prefix}{pcc_s} & {qwk_prefix}{qwk_s}}}")
            else:
                parts.append(f"{acc_s} & {pcc_s} & {qwk_s}")

        # Average QWK
        qwks = [m[d].get("qwk") for d in SCORE_DIMS if m.get(d, {}).get("qwk") is not None]
        avg_qwk = sum(qwks) / len(qwks) if qwks else float("nan")
        parts.append(f"{avg_qwk:.3f}")
        print(" & ".join(parts) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\label{tab:performance_comparison_one_shot}")
    print("\\end{table*}")

    # ---- Table 2: Per-prompt breakdown ----
    print("\n\n" + "=" * 70)
    print("TABLE 2: Per-prompt QWK breakdown (one-shot)")
    print("=" * 70)

    for mk in all_llm_keys:
        if mk not in model_data:
            continue
        display_name, group = MODEL_CONFIG[mk]
        print(f"\n--- {display_name} ({group}) ---")
        print(f"{'Prompt':<25}", end="")
        for dim in SCORE_DIMS:
            print(f" {DIM_LABELS[dim][:8]:>10} QWK", end="")
        print(f" {'Avg QWK':>10}")
        print("-" * 75)

        qwk_vals = {dim: [] for dim in SCORE_DIMS}
        for p in prompts:
            pdata = model_data[mk].get(p, {})
            print(f"{p:<25}", end="")
            for dim in SCORE_DIMS:
                dm = pdata.get(dim, {})
                qwk = dm.get("qwk")
                if qwk is not None:
                    qwk_vals[dim].append(qwk)
                    print(f" {qwk:>10.4f}", end="")
                else:
                    print(f" {'N/A':>10}", end="")
            # Per-prompt average
            p_qwks = [pdata.get(d, {}).get("qwk") for d in SCORE_DIMS
                      if pdata.get(d, {}).get("qwk") is not None]
            p_avg = sum(p_qwks) / len(p_qwks) if p_qwks else float("nan")
            print(f" {p_avg:>10.4f}")

        # Average
        print(f"{'AVERAGE':<25}", end="")
        for dim in SCORE_DIMS:
            avg = sum(qwk_vals[dim]) / len(qwk_vals[dim]) if qwk_vals[dim] else float("nan")
            print(f" {avg:>10.4f}", end="")
        all_avg_qwks = [sum(qwk_vals[d]) / len(qwk_vals[d]) for d in SCORE_DIMS if qwk_vals[d]]
        overall_avg = sum(all_avg_qwks) / len(all_avg_qwks) if all_avg_qwks else float("nan")
        print(f" {overall_avg:>10.4f}")


def generate_per_prompt_latex(model_data):
    """Generate a LaTeX table with per-prompt rows for each model."""
    prompts = collect_all_prompts(model_data)
    all_llm_keys = [k for k in MODEL_CONFIG if k in model_data]

    # Group by: for each model, for each prompt, output ACC/PCC/QWK per dimension
    # This is a VERY wide table. Let's do one table per dimension showing all models x all prompts.

    # Alternative: for each model, output per-prompt QWK values (compact)
    print("\n\n" + "=" * 70)
    print("PER-PROMPT QWK TABLE (LaTeX)")
    print("=" * 70)

    for dim in SCORE_DIMS:
        print(f"\n\\subsection{{{DIM_LABELS[dim]} QWK}}")
        print("\\resizebox{\\textwidth}{!}{%")
        # Header: Model | Prompt1 | Prompt2 | ... | Avg
        header = "Model" + " & ".join([p.replace("_", "\\_")[:12] for p in prompts]) + " & Avg"
        print(f"\\begin{{tabular}}{{l{'c' * (len(prompts) + 1)}}}")
        print("\\toprule")
        print(f"{header} \\\\")
        print("\\midrule")

        current_group = None
        for mk in all_llm_keys:
            display_name, group = MODEL_CONFIG[mk]
            if group != current_group:
                current_group = group
                group_label = "English LLMs" if group == "en" else "Chinese LLMs"
                n_cols = len(prompts) + 1
                print(f"\\multicolumn{{{n_cols}}}{{c}}{{\\textit{{{group_label}}}}} \\\\")

            pdata = model_data.get(mk, {})
            vals = []
            for p in prompts:
                qwk = pdata.get(p, {}).get(dim, {}).get("qwk")
                if qwk is not None:
                    vals.append(f"{qwk:.3f}")
                else:
                    vals.append("N/A")
            avg_qwk = sum(float(v) for v in vals if v != "N/A") / max(sum(1 for v in vals if v != "N/A"), 1)
            row = " & ".join([display_name] + vals + [f"{avg_qwk:.3f}"])
            print(row + " \\\\")

        print("\\bottomrule")
        print("\\end{tabular}%")
        print("}")


import sys

if __name__ == "__main__":
    # Redirect stdout to file for clean output
    out_path = Path("results/cross_prompt_llm/one_shot_table.tex")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(out_path, "w", encoding="utf-8")

    model_data = load_model_data()
    print(f"Loaded {len(model_data)} models: {list(model_data.keys())}")

    generate_latex_table(model_data)
    generate_per_prompt_latex(model_data)
