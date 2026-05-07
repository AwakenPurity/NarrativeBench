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
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-4439e80c66384d32beb81f0573a7f32e",
        "model_name": "qwen3.6-plus",
    },
    "claude_sonnet_4_6": {
        # "base_url": "https://api.innospark.cn/v1",
        # "api_key": "sk-nKVMps5VWWN9ciqbH6YpE00HEG9VuFe6TRm6cMfSFoiNiFlK",
        # "model_name": "claude-opus-4-6",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-daf3202ea2d94c3d8ae0602666c96743",
        "model_name": "deepseek-v4-pro",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-daf3202ea2d94c3d8ae0602666c96743",
        "model_name": "deepseek-chat",
    },
    "gpt_5_4": {
        "base_url": "https://api.nuwaapi.com/v1",
        "api_key": "sk-UK60Dux3qXNLriuzZXIK3gKVZGwTSRTx7mBOI2BXfeFm5fRp",
        "model_name": "gpt-5.4",
    },
    "gemini_3_1_pro": {
        # "base_url": "https://api.nuwaapi.com/v1",
        # "api_key": "sk-UK60Dux3qXNLriuzZXIK3gKVZGwTSRTx7mBOI2BXfeFm5fRp",
        # "model_name": "gemini-3-flash-preview-nothinking",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-daf3202ea2d94c3d8ae0602666c96743",
        "model_name": "deepseek-v4-pro",
    },
    "qwen3_5_27b": {
        # "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        # "api_key": "sk-4439e80c66384d32beb81f0573a7f32e",
        # "model_name": "qwen3.5-27b",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-daf3202ea2d94c3d8ae0602666c96743",
        "model_name": "deepseek-chat",
    },
    "qwen3_5_9b": { 
        "base_url": "https://api.siliconflow.cn/v1",
        "model_name": "Qwen/Qwen3.6-35B-A3B",
        "api_key": "sk-beeghpzaufeveryqxewtixbjjvptrbmqivhahxrxezmcmltp",
    },
    "grok_4":{
        "base_url": "https://api.chatanywhere.tech/v1",
        "api_key": "sk-R1HQDYWNH8qfr9nSxD4Zk9RZ5f8bvSkFBiRUk2MYGWxJ4AiC",
        "model_name": "deepseek-v4-flash",
    },
    "doubao_seed_2_0_pro": {
        # "base_url": "https://api.chatanywhere.tech/v1",
        # "api_key": "sk-R1HQDYWNH8qfr9nSxD4Zk9RZ5f8bvSkFBiRUk2MYGWxJ4AiC",
        # "model_name": "minimax-m2.5",
        "base_url": "https://api.deepseek.com/v1",
        "api_key": "sk-daf3202ea2d94c3d8ae0602666c96743",
        "model_name": "deepseek-chat",
    },
    "llama_4_scout":{
        "base_url": "https://api.chatanywhere.tech/v1",
        "api_key": "sk-R1HQDYWNH8qfr9nSxD4Zk9RZ5f8bvSkFBiRUk2MYGWxJ4AiC",
        "model_name": "gpt-5-nano-ca",
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
