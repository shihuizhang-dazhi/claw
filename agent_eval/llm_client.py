"""Shared LLM client abstraction for evaluation and defense components."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from email.utils import parsedate_to_datetime
from typing import Any, Callable, TypeVar
from typing import Optional

T = TypeVar("T")


class RetryableLLMResponseError(RuntimeError):
    """Raised when an OpenAI-compatible backend returns a transient bad shape."""


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _llm_error_status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status if isinstance(response_status, int) else None


def _is_rate_limit_llm_error(exc: Exception) -> bool:
    if _llm_error_status_code(exc) == 429:
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "ratelimit" in name or "rate_limit" in name or "rate limit" in text


def _is_retryable_llm_error(exc: Exception) -> bool:
    """Return whether an LLM backend error is likely transient."""
    if isinstance(exc, RetryableLLMResponseError):
        return True
    status_code = _llm_error_status_code(exc)
    if status_code in {403, 408, 409, 429, 500, 502, 503, 504}:
        return True
    name = type(exc).__name__.lower()
    return "connection" in name or "timeout" in name or "rate" in name


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    if headers is None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
    if headers is None:
        return None
    retry_after = None
    if hasattr(headers, "get"):
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        return max(float(retry_after), 0.0)
    except (TypeError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(str(retry_after))
        return max(retry_at.timestamp() - time.time(), 0.0)
    except (TypeError, ValueError, OverflowError):
        return None


def _retry_delay_seconds(
    exc: Exception,
    retry_index: int,
    *,
    rate_limit_initial_delay: float,
    rate_limit_max_delay: float,
) -> float:
    if _is_rate_limit_llm_error(exc):
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return min(retry_after, rate_limit_max_delay)
        return min(rate_limit_initial_delay * (2 ** retry_index), rate_limit_max_delay)
    return min(2 ** retry_index, 8)


def _with_retries(
    call: Callable[[], T],
    attempts: int = 4,
    *,
    rate_limit_attempts: Optional[int] = None,
    rate_limit_initial_delay: Optional[float] = None,
    rate_limit_max_delay: Optional[float] = None,
) -> T:
    rate_limit_attempts = rate_limit_attempts or max(
        attempts,
        _positive_int_env("LLM_RATE_LIMIT_RETRY_ATTEMPTS", 8),
    )
    rate_limit_initial_delay = rate_limit_initial_delay or _positive_float_env(
        "LLM_RATE_LIMIT_INITIAL_DELAY_SECONDS",
        5.0,
    )
    rate_limit_max_delay = rate_limit_max_delay or _positive_float_env(
        "LLM_RATE_LIMIT_MAX_DELAY_SECONDS",
        60.0,
    )
    rate_limit_retries = 0
    standard_retries = 0
    while True:
        try:
            return call()
        except Exception as exc:
            if not _is_retryable_llm_error(exc):
                raise
            is_rate_limit = _is_rate_limit_llm_error(exc)
            retry_index = rate_limit_retries if is_rate_limit else standard_retries
            allowed_attempts = rate_limit_attempts if is_rate_limit else attempts
            if retry_index >= allowed_attempts - 1:
                raise
            delay = _retry_delay_seconds(
                exc,
                retry_index,
                rate_limit_initial_delay=rate_limit_initial_delay,
                rate_limit_max_delay=rate_limit_max_delay,
            )
            if is_rate_limit:
                rate_limit_retries += 1
            else:
                standard_retries += 1
            time.sleep(delay)


def _chat_completion_with_retries(client: Any, **kwargs: Any) -> Any:
    def call() -> Any:
        resp = client.chat.completions.create(**kwargs)
        choices = getattr(resp, "choices", None)
        if not choices:
            raise RetryableLLMResponseError(
                f"LLM backend returned invalid chat completion shape: {type(resp).__name__}"
            )
        return resp

    return _with_retries(call)


class BaseLLMClient(ABC):
    """Abstract LLM completion interface."""

    @abstractmethod
    def complete(self, system: str, user: str) -> str:
        """Single-turn completion. Returns the assistant's text response."""


class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions backend."""

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            timeout=_positive_float_env("LLM_REQUEST_TIMEOUT_SECONDS", 180.0),
        )
        self.model = model

    def complete(self, system: str, user: str) -> str:
        text, _ = self.complete_with_entropy(system, user)
        return text

    def complete_with_entropy(self, system: str, user: str, top_logprobs: int = 20):
        """Return (text, per_token_entropies) using OpenAI logprobs."""
        import math

        resp = _chat_completion_with_retries(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=1024,
            logprobs=True,
            top_logprobs=top_logprobs,
        )
        choice = resp.choices[0]
        text = choice.message.content or ""

        entropies = []
        if hasattr(choice, "logprobs") and choice.logprobs:
            for token_logprob in choice.logprobs.content:
                if token_logprob.top_logprobs:
                    entropy = 0.0
                    for tl in token_logprob.top_logprobs:
                        p = math.exp(tl.logprob)
                        entropy -= p * math.log(max(p, 1e-12))
                    entropies.append(round(entropy, 6))
                else:
                    entropies.append(0.0)
        return text, entropies


class AnthropicClient(BaseLLMClient):
    """Anthropic Messages API backend."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        from anthropic import Anthropic

        self.client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = _with_retries(
            lambda: self.client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                temperature=0,
                max_tokens=1024,
            )
        )
        for block in resp.content:
            if block.type == "text":
                return block.text  # type: ignore[union-attr]
        return ""


class RightcodesClient(BaseLLMClient):
    """right.codes OpenAI-compatible backend."""

    def __init__(self, model: str = "gpt-5.4"):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ["RIGHTCODES_API_KEY"],
            base_url="https://right.codes/codex/v1",
            timeout=_positive_float_env("LLM_REQUEST_TIMEOUT_SECONDS", 180.0),
        )
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = _chat_completion_with_retries(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""


class CubanceClient(BaseLLMClient):
    """Cubance OpenAI-compatible backend."""

    def __init__(self, model: str = "gpt-5.4"):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ["CUBANCE_API_KEY"],
            base_url="https://api.cubence.com/v1",
            timeout=_positive_float_env("LLM_REQUEST_TIMEOUT_SECONDS", 180.0),
        )
        self.model = model

    def complete(self, system: str, user: str) -> str:
        resp = _chat_completion_with_retries(
            self.client,
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""


class SiliconFlowClient(BaseLLMClient):
    """SiliconFlow OpenAI-compatible backend with logprobs support."""

    def __init__(self, model: str = "deepseek-ai/DeepSeek-V4-Flash"):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=os.environ["SILICONFLOW_API_KEY"],
            base_url="https://api.siliconflow.cn/v1",
            timeout=_positive_float_env("LLM_REQUEST_TIMEOUT_SECONDS", 180.0),
        )
        self.model = model

    def complete(self, system: str, user: str) -> str:
        text, _ = self.complete_with_entropy(system, user)
        return text

    def complete_with_entropy(self, system: str, user: str, top_logprobs: int = 20):
        """Return (text, per_token_entropies) using API logprobs.

        Tries logprobs+top_logprobs first. Falls back to text-based entropy
        estimation (lexical diversity + n-gram entropy) if the model does not
        support logprobs (as is the case with many SiliconFlow models).
        """
        import math

        try:
            resp = _chat_completion_with_retries(
                self.client,
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0,
                max_tokens=1024,
                logprobs=True,
                top_logprobs=top_logprobs,
            )
            choice = resp.choices[0]
            text = choice.message.content or ""

            entropies = []
            if hasattr(choice, "logprobs") and choice.logprobs:
                for token_logprob in choice.logprobs.content:
                    if token_logprob.top_logprobs:
                        entropy = 0.0
                        for tl in token_logprob.top_logprobs:
                            p = math.exp(tl.logprob)
                            entropy -= p * math.log(p + 1e-12)
                        entropies.append(entropy)
                    else:
                        entropies.append(0.0)
            return text, entropies

        except Exception as e:
            if "top_logprobs" in str(e).lower() or "logprobs" in str(e).lower():
                # Fall back to text-based entropy
                resp = _chat_completion_with_retries(
                    self.client,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    max_tokens=1024,
                )
                choice = resp.choices[0]
                text = choice.message.content or ""
                entropies = self._text_based_entropy(text)
                return text, entropies
            raise

    def _text_based_entropy(self, text: str) -> list[float]:
        """Estimate per-token entropy from generated text using n-gram statistics.

        When API logprobs are unavailable, we estimate token-level entropy
        from the character-level and token-level dispersion of the generated text.
        The resulting entropy values are scaled to match the expected range of
        Shannon entropy from LLM output distributions (0.0--3.0 nats).
        """
        import math
        from collections import Counter

        if not text:
            return []

        # Tokenize roughly by whitespace + punctuation boundaries
        tokens = text.split()
        if len(tokens) < 2:
            return [1.5]  # default mid-entropy

        entropies = []
        window = min(5, len(tokens))

        for i, token in enumerate(tokens):
            start = max(0, i - window)
            end = min(len(tokens), i + window + 1)
            context = tokens[start:end]

            # Character-level entropy within the token
            char_counts = Counter(token)
            char_entropy = 0.0
            for count in char_counts.values():
                p = count / len(token)
                char_entropy -= p * math.log(p + 1e-12)
            char_entropy = min(char_entropy, 3.0)

            # N-gram diversity in local window (proxy for predictability)
            type_count = len(set(context))
            token_count = len(context)
            diversity = type_count / token_count if token_count > 0 else 0

            # Combine: low diversity + low char entropy → injection signal
            # Scale to match typical Shannon entropy range (0.0-3.0 nats)
            combined = char_entropy * (0.3 + 0.7 * diversity)
            entropies.append(round(combined, 6))

        return entropies


def create_llm_client(
    backend: str = "openai", model: Optional[str] = None
) -> BaseLLMClient:
    """Create an LLM client by backend name."""

    if backend == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini")
    if backend == "anthropic":
        return AnthropicClient(model=model or "claude-haiku-4-5-20251001")
    if backend == "rightcodes":
        return RightcodesClient(model=model or "gpt-5.4")
    if backend == "cubance":
        return CubanceClient(model=model or "gpt-5.4")
    if backend == "siliconflow":
        return SiliconFlowClient(model=model or "Pro/zai-org/GLM-5.1")
    raise ValueError(f"Unknown LLM backend: {backend}")
