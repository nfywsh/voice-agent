"""Chat history management utilities."""

import logging
from typing import Optional

from livekit.agents import ChatContext, ChatMessage

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Chat history manager with external injection and truncation support."""

    def __init__(self, max_turns: int = 10):
        self.max_turns = max_turns

    def inject_messages(self, chat_ctx: ChatContext, messages: list[dict]) -> None:
        """Inject external messages into chat history.

        Args:
            chat_ctx: Agent's ChatContext instance
            messages: Message list in format [{"role": "user"|"assistant", "content": "..."}]
        """
        for msg in messages:
            chat_ctx.add_message(role=msg["role"], content=msg["content"])

    def truncate(self, chat_ctx: ChatContext) -> None:
        """Truncate chat history to max_turns (to prevent infinite accumulation)."""
        items = chat_ctx.messages()
        if len(items) <= self.max_turns:
            return

        # Keep system message + last max_turns messages
        kept_messages = items[-self.max_turns:]
        chat_ctx.messages.clear()
        for msg in kept_messages:
            chat_ctx.add_message(role=msg.role, content=msg.content)
        logger.debug(f"[ChatHistoryManager] Truncated to {self.max_turns} turns")

    def get_history_text(self, chat_ctx: ChatContext) -> str:
        """Get formatted chat history as text for logging/debugging."""
        lines = []
        for msg in chat_ctx.messages():
            if isinstance(msg, ChatMessage):
                content = msg.content
                if isinstance(content, list):
                    content = str(content)
                lines.append(f"{msg.role}: {content[:100]}...")
        return "\n".join(lines)