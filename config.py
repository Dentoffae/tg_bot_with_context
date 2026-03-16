import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

# Optional proxy for regions where Telegram is blocked (e.g. socks5://host:port)
PROXY_URL: str = os.getenv("PROXY_URL", "")

OPENAI_MODEL: str = "gpt-4o-mini"

# System prompt sent at the start of every conversation
SYSTEM_PROMPT: str = (
    "Ты полезный ИИ-ассистент. Отвечай на русском языке, если пользователь пишет по-русски."
)

# Maximum number of messages to keep in context per user (excluding system prompt)
MAX_CONTEXT_MESSAGES: int = 20

# OpenAI generation parameters
MAX_COMPLETION_TOKENS: int = 1024
TEMPERATURE: float = 0.7

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не задан в .env файле")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY не задан в .env файле")
