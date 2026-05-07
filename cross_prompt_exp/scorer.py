# -*- coding: utf-8 -*-
"""
LLM scorer: calls LLM APIs to score essays.
Uses OpenAI-compatible API format.
"""

import json
import re
import time
import logging

import requests
from config import MAX_RETRIES, RETRY_DELAY, REQUEST_TIMEOUT

logger = logging.getLogger(__name__)


def call_llm_api(messages, model_config):
    """Call an OpenAI-compatible API to score an essay.

    Args:
        messages: list of message dicts for the API
        model_config: dict with base_url, api_key, model_name

    Returns:
        dict or None: parsed JSON response with scores, or None on failure
    """
    url = f"{model_config['base_url']}/chat/completions"
    headers = {
        "Authorization": f"Bearer {model_config['api_key']}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_config["model_name"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 8192,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Handle empty response explicitly
            if not content:
                finish_reason = data["choices"][0].get("finish_reason", "unknown")
                logger.warning(
                    f"Empty response from API (attempt {attempt+1}), "
                    f"finish_reason={finish_reason}, model={model_config['model_name']}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return None

            # Parse JSON from response
            scores = _parse_score_json(content)
            if scores is None:
                logger.warning(
                    f"Failed to parse JSON from response (attempt {attempt+1}): "
                    f"{content[:300]}"
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                return None

            return scores

        except requests.exceptions.Timeout as e:
            logger.warning(
                f"API timeout after {REQUEST_TIMEOUT}s (attempt {attempt+1}), "
                f"model={model_config['model_name']}: {e}"
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                return None
        except (KeyError, IndexError) as e:
            logger.warning(f"Unexpected API response format (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                return None

    return None


def _parse_score_json(text):
    """Extract and parse score JSON from LLM response text.

    Handles cases where the response may contain markdown code blocks
    or extra text around the JSON.

    Args:
        text: raw response text

    Returns:
        dict or None: parsed scores dict, or None if parsing fails
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code blocks
    # Pattern: ```json ... ``` or ``` ... ```
    pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    match = re.search(pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find JSON-like content between braces
    # This handles cases like: 好的，评分如下：{"total_score": 5, ...}
    pattern = r"\{[^{}]*\}"
    matches = re.findall(pattern, text)
    for m in matches:
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue

    return None


def score_essay(messages, model_config):
    """Score a single essay.

    Args:
        messages: list of message dicts
        model_config: dict with base_url, api_key, model_name

    Returns:
        dict: scores with keys total_score, relevance_score, content_score,
              expression_score. Invalid scores default to -1.
    """
    result = call_llm_api(messages, model_config)

    if result is None:
        return {
            "total_score": -1,
            "relevance_score": -1,
            "content_score": -1,
            "expression_score": -1,
        }

    return {
        "total_score": int(result.get("total_score", -1)),
        "relevance_score": int(result.get("relevance_score", -1)),
        "content_score": int(result.get("content_score", -1)),
        "expression_score": int(result.get("expression_score", -1)),
    }
