import logging
from typing import TypedDict

from config import MAX_CONTEXT_MESSAGES, SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class Message(TypedDict):
    role: str
    content: str


# In-memory storage: {user_id: [{"role": ..., "content": ...}, ...]}
_context: dict[int, list[Message]] = {}


def get_context(user_id: int) -> list[Message]:
    """Return the full message list for a user, including the system prompt."""
    if user_id not in _context:
        _context[user_id] = []
    return [{"role": "system", "content": SYSTEM_PROMPT}] + _context[user_id]


def add_message(user_id: int, role: str, content: str) -> None:
    """Append a message to the user's context and trim if it exceeds the limit."""
    if user_id not in _context:
        _context[user_id] = []

    _context[user_id].append({"role": role, "content": content})

    if len(_context[user_id]) > MAX_CONTEXT_MESSAGES:
        # Remove the oldest messages but keep the conversation coherent
        _context[user_id] = _context[user_id][-MAX_CONTEXT_MESSAGES:]
        logger.debug("Context for user %d trimmed to %d messages", user_id, MAX_CONTEXT_MESSAGES)


def clear_context(user_id: int) -> None:
    """Delete all stored messages for a user."""
    _context.pop(user_id, None)
    logger.info("Context cleared for user %d", user_id)


def context_size(user_id: int) -> int:
    """Return the number of stored messages (excluding the system prompt)."""
    return len(_context.get(user_id, []))
