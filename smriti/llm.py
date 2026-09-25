"""Answer generation with Groq (OpenAI-compatible chat completions, streamed)."""

from __future__ import annotations

from collections.abc import Iterator

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]


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
