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
 "base_url": "https://api.nuwaapi.com/v1",
 "api_key": "sk-g9d1ft6U9iUUWdt7FTVfIqlmxYkUtTaEFGdZTsgsTFgLCz4j",
 "model_name": "gpt-5.4",
 },
 "gemini_3_1_pro": {
#  "base_url": "https://api.nuwaapi.com/v1",
#  "api_key": "sk-G72A8tTkmLAN9QOUa8YtdeFdrMmlQyIn9YmqV3IKYGVNFGOP",
#  "model_name": "gemini-3.1-pro-preview",
"base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
 "api_key": "sk-91276432e8364e3d94a6dd72930655ce",
 "model_name": "qwen3.6-plus",
 },
 "claude_sonnet_4_6": {
 "base_url": "https://runapi.co/v1",
 "api_key": "sk-BCizVdlDGTJECA4KOg0q1fWEAX7x0fchIwsk3zDSrB5fr0xS",
 "model_name": "claude-sonnet-4-6",
 },
 "deepseek": {
 "base_url": "https://api.deepseek.com/v1",
 "api_key": "sk-daf3202ea2d94c3d8ae0602666c96743",
 "model_name": "deepseek-chat",
 },
 "qwen3_6_plus": {
 "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
 "api_key": "sk-91276432e8364e3d94a6dd72930655ce",
 "model_name": "qwen3.6-plus",
 },
 "qwen3_5_27b":{
 "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
 "api_key": "sk-91276432e8364e3d94a6dd72930655ce",
 "model_name": "qwen3.5-27b",
 },
 "qwen3_5_9b": {
 "base_url": "https://notebook-inspire.sii.edu.cn/ws-9dcc0e1f-80a4-4af2-bc2f-0e352e7b17e6/project-b795c114-135a-40db-b3d0-19b60f25237b/user-543feed4-0be2-4972-8987-a324af06c93f/vscode/bc41494c-305d-41a4-a724-85a8f7bb7518/ef0a4b00-c56f-461b-836e-c43a32ab2bfa/proxy/8000/v1",
 "model_name": "qwen3_5_9b",
 "api_key": "EMPTY",
 "is_local": True  
#  "base_url": "https://api.deepseek.com/v1",
#  "api_key": "sk-daf3202ea2d94c3d8ae0602666c96743",
#  "model_name": "deepseek-chat",
 },
 "doubao_seed_2_0_pro": {
#  "base_url": "https://ark.cn-beijing.volces.com/api/v3",
#  "api_key": "ark-94b0f30a-5f12-4f4f-b096-8d5a14b69319-fb499",
#  "model_name": "doubao-seed-2-0-pro-260215",
 "base_url": "https://api.nuwaapi.com/v1",
 "api_key": "sk-UK60Dux3qXNLriuzZXIK3gKVZGwTSRTx7mBOI2BXfeFm5fRp",
 "model_name": "gpt-4o-mini",
 },
 "grok_4":{
#  "base_url": "https://runapi.co/v1",
#  "api_key": "sk-rJMAGdSVKXV5jgWVCIpehsGi4SjqgGNlUcZdljEvXVTuA83j",
#  "model_name": "grok-4.2",
 "base_url": "https://api.nuwaapi.com/v1",
 "api_key": "sk-UK60Dux3qXNLriuzZXIK3gKVZGwTSRTx7mBOI2BXfeFm5fRp",
 "model_name": "gpt-4o-mini",
 },
 "llama_4_scout":{
 "base_url": "https://api.nuwaapi.com/v1",
 "api_key": "sk-UK60Dux3qXNLriuzZXIK3gKVZGwTSRTx7mBOI2BXfeFm5fRp",
 "model_name": "gpt-4o-mini",
 }
}

# modelscope download --model Qwen/Qwen3.5-9B --local_dir /inspire/hdd/project/ai4education/qianhong-p-qianhong/LLM/Qwen3.5-9B
# DeepSeek config for comment evaluation (pairwise comparison)
DEEPSEEK_EVAL_CONFIG = {
 "base_url": "https://api.deepseek.com/v1",
 "api_key": "sk-daf3202ea2d94c3d8ae0602666c96743",
 "model_name": "deepseek-chat",
}

# Scoring parameters
MAX_RETRIES = 3
RETRY_DELAY = 5 # seconds
CHECKPOINT_INTERVAL = 50 # save checkpoint every N essays
REQUEST_TIMEOUT = 600  # seconds
