import logging

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

from config import OPENAI_API_KEY, OPENAI_MODEL, MAX_COMPLETION_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def get_ai_response(messages: list[dict]) -> str:
    """
    Send a list of messages to the OpenAI Chat Completions API and return
    the assistant's reply as a plain string.

    Raises a user-friendly RuntimeError on API failures so the caller can
    forward the message to the user without exposing internal details.
    """
    try:
        logger.debug("Sending %d messages to OpenAI (model=%s)", len(messages), OPENAI_MODEL)

        response = await _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            temperature=TEMPERATURE,
        )

        choice = response.choices[0]
        message = choice.message

        # Log the full raw response for debugging
        logger.info(
            "OpenAI raw response | finish_reason=%s | content=%r | refusal=%r",
            choice.finish_reason,
            message.content,
            getattr(message, "refusal", None),
        )

        reply = (message.content or "").strip()

        if not reply:
            finish_reason = choice.finish_reason
            refusal = getattr(message, "refusal", None)

            if refusal:
                logger.warning("OpenAI refused the request: %s", refusal)
                raise RuntimeError(f"Модель отказала в ответе: {refusal}")

            logger.warning(
                "OpenAI returned empty content (finish_reason=%s)", finish_reason
            )
            raise RuntimeError(
                f"Модель вернула пустой ответ (finish_reason={finish_reason}). "
                "Попробуйте переформулировать запрос."
            )

        return reply

    except RateLimitError:
        logger.warning("OpenAI rate limit exceeded")
        raise RuntimeError(
            "Превышен лимит запросов к OpenAI. Пожалуйста, подождите немного и попробуйте снова."
        )
    except APIConnectionError as exc:
        logger.error("OpenAI connection error: %s", exc)
        raise RuntimeError(
            "Не удалось подключиться к OpenAI. Проверьте интернет-соединение и повторите попытку."
        )
    except APIError as exc:
        logger.error("OpenAI API error: %s", exc)
        raise RuntimeError(
            f"Ошибка OpenAI API: {exc.message if hasattr(exc, 'message') else str(exc)}"
        )
