"""Answer generation with Groq (OpenAI-compatible chat completions, streamed)."""

from __future__ import annotations

from collections.abc import Iterator

# Preferred order when these are available. Groq retires models over time, so the app
# asks Groq for the live list (see list_chat_models) and only uses this as a fallback.
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]

_NOT_CHAT = ("whisper", "tts", "guard", "playai", "orpheus", "distil", "embed", "compound")


def list_chat_models(api_key: str) -> list[str]:
    """Return the chat models this Groq key can use right now, preferred ones first."""
    from groq import Groq

    models = Groq(api_key=api_key).models.list().data
    ids = [m.id for m in models if getattr(m, "active", True) and not any(x in m.id.lower() for x in _NOT_CHAT)]
    preferred = [m for m in GROQ_MODELS if m in ids]
    return preferred + sorted(set(ids) - set(preferred))


class LLMError(Exception):
    """A friendly, user-facing error."""


def _friendly(err: Exception) -> LLMError:
    name = type(err).__name__
    msg = {
        "AuthenticationError": "Groq rejected the API key. Check it and try again.",
        "RateLimitError": "Groq rate limit reached (free tier). Wait a minute, then ask again.",
        "APIConnectionError": "Couldn't reach Groq. Check your internet connection.",
        "APITimeoutError": "Groq took too long to respond. Try again.",
        "NotFoundError": "That model isn't available on Groq. Pick another in the sidebar.",
    }.get(name)
    if msg is None and "context" in str(err).lower():
        msg = "The question plus passages was too long for this model. Lower 'Passages per answer'."
    return LLMError(msg or f"Groq returned an error: {err}")


class GroqLLM:
    def __init__(self, api_key: str, model: str = GROQ_MODELS[0]):
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.model = model

    def stream(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1500) -> Iterator[str]:
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:  # noqa: BLE001 - convert SDK errors into friendly messages
            raise _friendly(e) from e
