# -*- coding: utf-8 -*-
"""
Configuration for cross-prompt LLM essay scoring experiment.
"""

# ============================================================
# Data paths
# ============================================================
DATA_PATH = "essay_total_data_4500_260417.xlsx"
OUTPUT_DIR = "results/cross_prompt"

# ============================================================
# Prompt column — which column identifies the essay topic
# ============================================================
PROMPT_COLUMN = "hw_title"

# ============================================================
# Few-shot configuration
# ============================================================
# Each individual score (0-8) is its own bucket.
# Few-shot selection will pick samples from each score level.
SCORE_RANGE = range(0, 9)  # 0, 1, 2, ..., 8

# Number of few-shot samples per score level.
# Experiment runs both 1 and 3.
FEW_SHOT_PER_BUCKET = [1, 3]

# Random seed for reproducible few-shot sampling
RANDOM_SEED = 42

# ============================================================
# LLM API configurations (OpenAI-compatible)
# ============================================================
MODELS = {
    "qwen3_6_plus": {
        "base_url": "xxx",
        "api_key": "xxx",
        "model_name": "qwen3.6-plus",
    },
    "claude_sonnet_4_6": {
        "base_url": "xxx",
        "api_key": "xxx",
        "model_name": "xxx",
    },
    "deepseek": {
        "base_url": "xxx",
        "api_key": "xxx",
        "model_name": "xxx",
    },
    "gpt_5_4": {
        "base_url": "xxx",
        "api_key": "xxx",
        "model_name": "xxx",
    },
    "gemini_3_1_pro": {
        "base_url": "xxx",
        "api_key": "xxx",
        "model_name": "xxx",
    },
    "qwen3_5_27b": {
        "base_url": "xxx",
        "api_key": "xxx",
        "model_name": "xxx",
    },
    "qwen3_5_9b": { 
        "base_url": "xxx",
        "model_name": "xxx",
        "api_key": "xxx",
    },
    "grok_4":{
        "base_url": "xxx",
        "api_key": "xxx",
        "model_name": "xxx",
    },
    "doubao_seed_2_0_pro": {
        "base_url": "xxx",
        "api_key": "xxx",
        "model_name": "xxx",
    },
    "llama_4_scout":{
        "base_url": "xxx",
        "api_key": "xxx",
        "model_name": "xxx",
    }
}

# Default model to use for the experiment
DEFAULT_MODEL = "qwen3_6_plus"

# ============================================================
# API call parameters
# ============================================================
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds
REQUEST_TIMEOUT = 120  # seconds
MAX_CONCURRENT = 3  # max parallel API calls

# ============================================================
# Experiment control
# ============================================================
# If True, only run on a small subset for debugging
DEBUG_MODE = False
DEBUG_TEST_SIZE = 3  # essays per test prompt in debug mode
