# -*- coding: utf-8 -*-
"""
API configurations for LLM essay scoring.
Fill in the base_url, api_key, and model_name for each model.
All APIs use OpenAI-compatible format (POST /v1/chat/completions).

√ doubao-seed-2.0-pro
√ gpt-5.4
√ gemini-3.1-pro
√ claude-sonnet-4.6
√ qwen3.6-plus
O grok-4

llama-4-scout
√ deepseek-v3.2
√ qwen3.5-27b
√ qwen3.5-9b
"""

# Model configurations
MODELS = {

 "gemini_3_1_pro": {
 "base_url": "https://api.deepseek.com/v1",
 "api_key": "sk-a9b70c494a98465b99019e7ce1c13729",
 "model_name": "deepseek-v4-pro",
 },
 "deepseek": {
 "base_url": "https://api.deepseek.com/v1",
 "api_key": "sk-a9b70c494a98465b99019e7ce1c13729",
 "model_name": "deepseek-chat",
 }
}

# Scoring parameters
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds
CHECKPOINT_INTERVAL = 50 # save checkpoint every N essays
REQUEST_TIMEOUT = 600  # seconds
