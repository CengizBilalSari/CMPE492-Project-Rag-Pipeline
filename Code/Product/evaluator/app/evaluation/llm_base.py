from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 4096) -> None:
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

        now = time.time()
        self.request_timestamps.append(now)
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts <= 60.0]

    @abstractmethod
    async def ainvoke(self, prompt: str, system_prompt: Optional[str] = None) -> str: ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
