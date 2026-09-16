"""Shared LLM initialization — single source of truth for all model instances."""

from __future__ import annotations

import logging
import os
import time

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

from scotus_v3 import config, keys

log = logging.getLogger(__name__)

# Track which models are temporarily quota-exhausted
_quota_blocked: dict[str, float] = {}  # model_label -> unblock_timestamp


def _ensure_env() -> dict[str, str]:
    """Load API keys from Keychain and inject into environment (required by LangChain)."""
    api_keys = keys.get_all_keys()
    os.environ["OPENAI_API_KEY"] = api_keys["openai"]
    os.environ["GOOGLE_API_KEY"] = api_keys["google"]
    os.environ["ANTHROPIC_API_KEY"] = api_keys["anthropic"]
    return api_keys


def _make_openai(api_key: str, model: str, temperature: float, max_tokens: int, timeout: int) -> ChatOpenAI:
    """Create ChatOpenAI with correct token parameter for the model generation."""
    is_new_model = any(model.startswith(p) for p in ["gpt-5", "o3", "o4"])
    kwargs = {
        "api_key": api_key,
        "model": model,
        "temperature": temperature,
        "timeout": timeout,
        "max_retries": 2,
    }
    if is_new_model:
        kwargs["max_completion_tokens"] = max_tokens
    else:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


def _model_label(model_id: str) -> str:
    """Derive a short display label from the model ID."""
    if "gpt-5" in model_id:
        return "GPT-5.2"
    if "gpt-4" in model_id:
        return "GPT-4o"
    if "claude" in model_id:
        if "opus" in model_id:
            return "Claude-Opus"
        if "4-6" in model_id or "4.6" in model_id:
            return "Claude-4.6"
        return "Claude"
    if "gemini-3" in model_id:
        return "Gemini-3.1"
    if "gemini-2" in model_id:
        return "Gemini-2.5"
    return model_id


def mark_quota_exhausted(model_label: str, retry_after_seconds: int = 3600) -> None:
    """Mark a model as quota-exhausted. It will be skipped until the cooldown expires."""
    unblock_at = time.time() + retry_after_seconds
    _quota_blocked[model_label] = unblock_at
    log.warning("  %s quota exhausted — skipping for %d min", model_label, retry_after_seconds // 60)


def is_quota_exhausted(model_label: str) -> bool:
    """Check if a model is currently blocked due to quota."""
    if model_label not in _quota_blocked:
        return False
    if time.time() >= _quota_blocked[model_label]:
        del _quota_blocked[model_label]
        log.info("  %s quota cooldown expired — re-enabling", model_label)
        return False
    return True


def get_blocked_models() -> list[str]:
    """Return list of currently quota-blocked model labels."""
    now = time.time()
    return [label for label, until in _quota_blocked.items() if now < until]


def is_rate_limit_error(error: Exception) -> bool:
    """Detect if an exception is a rate-limit / quota-exhausted error."""
    err_str = str(error).lower()
    return any(keyword in err_str for keyword in [
        "429", "rate", "quota", "resource_exhausted", "too many requests",
    ])


def get_prediction_models() -> dict[str, ChatOpenAI | ChatGoogleGenerativeAI | ChatAnthropic]:
    """LLMs configured for prediction. Skips quota-blocked models."""
    api_keys = _ensure_env()
    cfg = config.load()
    model_cfg = cfg["models"]
    pred = cfg["prediction"]

    all_models = {
        _model_label(model_cfg["openai"]): lambda: _make_openai(
            api_keys["openai"], model_cfg["openai"],
            pred["temperature"], pred["max_tokens"], 120,
        ),
        _model_label(model_cfg["google"]): lambda: ChatGoogleGenerativeAI(
            google_api_key=api_keys["google"],
            model=model_cfg["google"],
            temperature=pred["temperature"],
            max_tokens=pred["max_tokens"],
            timeout=120,
            max_retries=2,
        ),
        _model_label(model_cfg["anthropic"]): lambda: ChatAnthropic(
            anthropic_api_key=api_keys["anthropic"],
            model=model_cfg["anthropic"],
            temperature=pred["temperature"],
            max_tokens=pred["max_tokens"],
            timeout=120,
            max_retries=2,
        ),
    }

    result = {}
    for label, factory in all_models.items():
        if is_quota_exhausted(label):
            log.info("  Skipping %s (quota exhausted)", label)
            continue
        result[label] = factory()
    return result


def get_embedding_model() -> OpenAIEmbeddings:
    """OpenAI embedding model for FAISS indexing."""
    api_keys = _ensure_env()
    cfg = config.load()["retrieval"]
    return OpenAIEmbeddings(
        openai_api_key=api_keys["openai"],
        model=cfg["embedding_model"],
        dimensions=cfg["embedding_dimensions"],
    )


def get_classifier_model() -> ChatOpenAI:
    """Lightweight model for document classification."""
    api_keys = _ensure_env()
    cfg = config.load()
    return _make_openai(api_keys["openai"], cfg["models"]["openai"], 0.0, 256, 60)
