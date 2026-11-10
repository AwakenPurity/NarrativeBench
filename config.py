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
 "gpt_5_4": {
 "base_url": "xxx",
 "api_key": "sk-xxx",
 "model_name": "gpt-5.4",
 },
 "gemini_3_1_pro": {
"base_url": "xxx",
 "api_key": "xxx",
 "model_name": "xxx",
 },
 "claude_sonnet_4_6": {
 "base_url": "xxx",
 "api_key": "xxx",
 "model_name": "claude-sonnet-4-6",
 },
 "deepseek": {
 "base_url": "https://api.deepseek.com/v1",
 "api_key": "xxx",
 "model_name": "deepseek-chat",
 },
 "qwen3_6_plus": {
 "base_url": "xxx",
 "api_key": "xxx",
 "model_name": "qwen3.6-plus",
 },
 "qwen3_5_27b":{
 "base_url": "xxx",
 "api_key": "xxx",
 "model_name": "qwen3.5-27b",
 },
 "qwen3_5_9b": {
 "base_url": "xxx",
 "model_name": "qwen3_5_9b",
 "api_key": "xxx",
 "is_local": True  
 },
 "doubao_seed_2_0_pro": {
 "base_url": "xxx",
 "api_key": "xxx",
 "model_name": "xxx",
 },
 "grok_4":{
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

DEEPSEEK_EVAL_CONFIG = {
 "base_url": "https://api.deepseek.com/v1",
 "api_key": "xxx",
 "model_name": "xxx",
}

# Scoring parameters
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds
CHECKPOINT_INTERVAL = 50 # save checkpoint every N essays
REQUEST_TIMEOUT = 600  # seconds
