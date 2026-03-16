import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

import context_manager as ctx
from config import BOT_TOKEN, PROXY_URL
from openai_client import get_ai_response

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bot & Dispatcher
# ---------------------------------------------------------------------------
_session = AiohttpSession(proxy=PROXY_URL) if PROXY_URL else None
bot = Bot(token=BOT_TOKEN, session=_session)
dp = Dispatcher()

# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@dp.message(CommandStart())
async def handle_start(message: Message) -> None:
    user_id = message.from_user.id
    ctx.clear_context(user_id)
    await message.answer(
        "Привет! Я ИИ-ассистент на базе GPT.\n\n"
        "Просто напиши мне что-нибудь, и я отвечу.\n"
        "Чтобы начать диалог заново, отправь /clear или напиши «очистить контекст»."
    )
    logger.info("User %d started the bot", user_id)


@dp.message(Command("clear"))
async def handle_clear_command(message: Message) -> None:
    user_id = message.from_user.id
    ctx.clear_context(user_id)
    await message.answer("Контекст диалога очищен. Можем начать сначала!")
    logger.info("User %d cleared context via /clear", user_id)


@dp.message(Command("info"))
async def handle_info(message: Message) -> None:
    user_id = message.from_user.id
    size = ctx.context_size(user_id)
    await message.answer(f"Сообщений в контексте: {size}")


@dp.message(F.text)
async def handle_text(message: Message) -> None:
    user_id = message.from_user.id
    text = (message.text or "").strip()

    if not text:
        return

    # Allow clearing context via plain text
    if text.lower() in ("очистить контекст", "clear context", "/clear"):
        ctx.clear_context(user_id)
        await message.answer("Контекст диалога очищен. Можем начать сначала!")
        logger.info("User %d cleared context via text command", user_id)
        return

    # Build messages list and add the new user message
    ctx.add_message(user_id, "user", text)
    messages = ctx.get_context(user_id)

    # Show typing indicator while waiting for GPT
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")

    try:
        reply = await get_ai_response(messages)
    except RuntimeError as exc:
        await message.answer(str(exc))
        # Remove the user message we already added so the failed turn is not
        # included in the next context
        ctx.clear_context(user_id)
        logger.warning("OpenAI error for user %d: %s", user_id, exc)
        return

    # Persist the assistant reply in the context
    ctx.add_message(user_id, "assistant", reply)

    await message.answer(reply)
    logger.info(
        "User %d | context size: %d | reply length: %d chars",
        user_id,
        ctx.context_size(user_id),
        len(reply),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    logger.info("Starting bot...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
