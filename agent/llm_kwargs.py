"""LLM kwargs builder with thinking mode injection support."""

import json
import logging
import os

from thinking import get_thinking_mode

logger = logging.getLogger(__name__)

# Model parameter mapping table (add mapping when switching models)
MODEL_THINKING_PARAM_MAP = {
    # DashScope Qwen series
    "qwen": {"param": "chat_template_kwargs", "key": "enable_thinking"},
    # OpenAI GPT series (uses reasoning_effort)
    "gpt": {"param": "extra_kwargs", "key": "reasoning_effort"},
    # Gemini series (uses thinking_budget)
    "gemini": {"param": "extra_kwargs", "key": "thinking_budget"},
}


def build_llm_kwargs(room_id: str, user_id: str) -> dict:
    """Build LLM call parameters including current thinking mode state.

    Args:
        room_id: Room ID
        user_id: User ID

    Returns:
        kwargs dict with thinking mode parameters, can be passed to LLM constructor or chat() call
    """
    is_thinking = get_thinking_mode(room_id, user_id)
    model = os.environ.get("LLM_MODEL", "qwen3.5-122b-a10b").lower()

    # Select parameter mapping based on model type
    kwargs = {}
    for model_prefix, param_map in MODEL_THINKING_PARAM_MAP.items():
        if model_prefix in model:
            if param_map["param"] == "chat_template_kwargs":
                # DashScope format
                template_kwargs = json.loads(
                    os.environ.get("LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
                )
                template_kwargs["enable_thinking"] = is_thinking
                kwargs["extra_kwargs"] = {"chat_template_kwargs": template_kwargs}
            else:
                # Other formats (extra_kwargs directly)
                extra_kwargs = kwargs.get("extra_kwargs", {})
                extra_kwargs[param_map["key"]] = is_thinking
                kwargs["extra_kwargs"] = extra_kwargs
            break
    else:
        # Default: try DashScope format
        template_kwargs = json.loads(
            os.environ.get("LLM_CHAT_TEMPLATE_KWARGS", '{"enable_thinking": false}')
        )
        template_kwargs["enable_thinking"] = is_thinking
        kwargs["extra_kwargs"] = {"chat_template_kwargs": template_kwargs}

    logger.debug(f"[llm_kwargs] room={room_id}, thinking={is_thinking}, kwargs={kwargs}")
    return kwargs