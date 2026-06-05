"""Thinking mode store for per-session/per-user thinking state."""

import logging

logger = logging.getLogger(__name__)

# key format: "{room_id}:{user_id}"
# only valid within session lifecycle, auto-cleared after session ends
_thinking_mode_store: dict[str, bool] = {}


def get_thinking_mode(room_id: str, user_id: str) -> bool:
    """Get current session's thinking mode state."""
    return _thinking_mode_store.get(f"{room_id}:{user_id}", False)


def set_thinking_mode(room_id: str, user_id: str, enabled: bool) -> None:
    """Set current session's thinking mode state."""
    _thinking_mode_store[f"{room_id}:{user_id}"] = enabled
    logger.info(f"[thinking_mode] room={room_id}, user={user_id}, enabled={enabled}")


def clear_thinking_mode(room_id: str, user_id: str) -> None:
    """Clear thinking mode state (called on session end)."""
    key = f"{room_id}:{user_id}"
    if key in _thinking_mode_store:
        del _thinking_mode_store[key]