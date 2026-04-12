from __future__ import annotations
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

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
        import asyncio
        import random
        retries = 0
        max_retries = 10
        base_delay = 4.0
        
        while True:
            try:
                return await call_fn()
            except Exception as e:
                err_str = str(e).lower()
                if "429" in err_str or "resource exhausted" in err_str or "resourceexhausted" in err_str or "too many requests" in err_str:
                    self.total_429_errors += 1
                    if retries >= max_retries:
                        raise e
                    retries += 1
                    
                    # Exponential backoff with random jitter (0 to 2 seconds)
                    delay = (base_delay * (1.5 ** (retries - 1))) + random.uniform(0.0, 2.0)
                    
                    logger.warning(f"{provider_name} Retrying in {delay:.2f}s (Attempt {retries}/{max_retries})... Error details: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise e

    def _update_and_log_usage(self, prompt_tokens: int, completion_tokens: int, duration_sec: float) -> None:
        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        total = self.total_prompt_tokens + self.total_completion_tokens
        
        now = time.time()
        self.request_timestamps.append(now)
        # Keep only timestamps within the last 60 seconds
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts <= 60.0]
        rpm = len(self.request_timestamps)
        
        logger.info(
            f"[LLM Usage] Req #{self.total_requests} | "
            f"Time: {duration_sec:.2f}s | RPM: {rpm} | "
            f"Tokens: {prompt_tokens} in, {completion_tokens} out | "
            f"Total: {total} (P: {self.total_prompt_tokens}, C: {self.total_completion_tokens})"
        )


    @abstractmethod
    async def ainvoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
     
        ...

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(prompt, system_prompt)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
