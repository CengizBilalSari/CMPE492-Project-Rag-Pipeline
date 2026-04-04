from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

from core.config import LLMProvider, LMSTUDIO_BASE_URL

logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.total_requests: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_429_errors: int = 0
        self.request_timestamps: list[float] = []

    async def _execute_with_retry(self, provider_name: str, call_fn) -> any:
        retries = 0
        max_retries = 10
        base_delay = 4.0

        while True:
            try:
                return await call_fn()
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(
                    s in err_str
                    for s in ("429", "resource exhausted", "resourceexhausted", "too many requests")
                )
                if is_rate_limit:
                    self.total_429_errors += 1
                    if retries >= max_retries:
                        raise
                    retries += 1
                    delay = (base_delay * (1.5 ** (retries - 1))) + random.uniform(0.0, 2.0)
                    logger.warning(
                        "%s Retrying in %.2fs (Attempt %d/%d)... Error: %s",
                        provider_name, delay, retries, max_retries, e,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise

    def _update_and_log_usage(self, prompt_tokens: int, completion_tokens: int, duration_sec: float) -> None:
        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        total = self.total_prompt_tokens + self.total_completion_tokens

        now = time.time()
        self.request_timestamps.append(now)
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts <= 60.0]
        rpm = len(self.request_timestamps)

        logger.info(
            "[LLM Usage] Req #%d | Time: %.2fs | RPM: %d | "
            "Tokens: %d in, %d out | Total: %d (P: %d, C: %d)",
            self.total_requests, duration_sec, rpm,
            prompt_tokens, completion_tokens, total,
            self.total_prompt_tokens, self.total_completion_tokens,
        )

    @abstractmethod
    async def ainvoke(self, prompt: str, system_prompt: Optional[str] = None) -> str: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"


class OpenAILLM(LLMInterface):
    """Standard OpenAI API client."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self._client = AsyncOpenAI(api_key=api_key, max_retries=0)

    async def ainvoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()

        async def _call():
            return await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        response = await self._execute_with_retry("OpenAI", _call)
        duration = time.time() - start

        content = response.choices[0].message.content or ""
        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0
        self._update_and_log_usage(prompt_tokens, completion_tokens, duration)

        return content


class LMStudioLLM(LLMInterface):
    """LMStudio-hosted model via OpenAI-compatible API at localhost:1234."""

    def __init__(
        self,
        model: str = "deepseek/deepseek-r1-0528-qwen3-8b",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = LMSTUDIO_BASE_URL,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key="lm-studio",
            base_url=base_url,
            max_retries=0,
        )

    async def ainvoke(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.time()

        async def _call():
            return await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        response = await self._execute_with_retry("LMStudio", _call)
        duration = time.time() - start

        content = response.choices[0].message.content or ""
        prompt_tokens = getattr(response.usage, "prompt_tokens", 0) if response.usage else 0
        completion_tokens = getattr(response.usage, "completion_tokens", 0) if response.usage else 0
        self._update_and_log_usage(prompt_tokens, completion_tokens, duration)

        return content


def get_llm(provider: str, **kwargs) -> LLMInterface:
    p = LLMProvider(provider.lower())
    if p == LLMProvider.OPENAI:
        return OpenAILLM(**kwargs)
    elif p == LLMProvider.LMSTUDIO:
        return LMStudioLLM(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider '{provider}'. Choose from: openai, lmstudio")
