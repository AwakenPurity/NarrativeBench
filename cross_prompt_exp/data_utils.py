# -*- coding: utf-8 -*-
"""
Data utilities for cross-prompt experiment:
- Loading and preprocessing data
- Leave-one-prompt-out splitting
- Few-shot example selection
"""

import pandas as pd
import numpy as np
from config import DATA_PATH, PROMPT_COLUMN, SCORE_RANGE, RANDOM_SEED


def load_data():
    """Load and validate the essay dataset.

    Returns:
        pd.DataFrame: loaded essay data
    """
    df = pd.read_excel(DATA_PATH)
    print(f"Loaded {len(df)} essays from {DATA_PATH}")
    print(f"Prompts: {df[PROMPT_COLUMN].nunique()} unique topics")
    print(f"Prompt distribution:")
    for prompt, count in df[PROMPT_COLUMN].value_counts().items():
        print(f"  {prompt}: {count}")
    return df


def leave_one_prompt_out_splits(df):
    """Generate leave-one-prompt-out train/test splits.

    Yields:
        tuple: (test_df, train_pool_df, test_prompt_name)
    """
    prompts = sorted(df[PROMPT_COLUMN].unique())
    for i, test_prompt in enumerate(prompts):
        test_df = df[df[PROMPT_COLUMN] == test_prompt].reset_index(drop=True)
        train_pool_df = df[df[PROMPT_COLUMN] != test_prompt].reset_index(drop=True)
        print(f"\n[{i+1}/{len(prompts)}] Test prompt: {test_prompt} "
              f"(n={len(test_df)}, train_pool={len(train_pool_df)})")
        yield test_df, train_pool_df, test_prompt


def select_few_shot_examples(train_pool_df, n_per_bucket, seed=RANDOM_SEED):
    """Select few-shot examples from the training pool, stratified by individual score levels.

    For each score level (0-8), randomly select n_per_bucket examples.
    Selection is deterministic given the same seed.

    Args:
        train_pool_df: DataFrame of training essays (all prompts except test prompt)
        n_per_bucket: number of examples to select per score level
        seed: random seed for reproducibility

    Returns:
        list[dict]: selected few-shot examples, each as a dict with scoring fields
    """
    rng = np.random.RandomState(seed)
    examples = []

    for score in SCORE_RANGE:
        score_df = train_pool_df[train_pool_df["total_score"] == score]

        if len(score_df) == 0:
            print(f"  WARNING: No training essays with total_score={score}")
            continue

        n_select = min(n_per_bucket, len(score_df))
        sampled = score_df.sample(n=n_select, random_state=rng.randint(0, 2**31))

        for _, row in sampled.iterrows():
            examples.append({
                "stu_title": row["stu_title"],
                "content": row["content"],
                "total_score": int(row["total_score"]),
                "relevance_score": int(row["relevance_score"]),
                "content_score": int(row["content_score"]),
                "expression_score": int(row["expression_score"]),
            })

        print(f"  Few-shot score={score}: selected {n_select} examples")

    print(f"  Total few-shot examples: {len(examples)}")
    return examples
