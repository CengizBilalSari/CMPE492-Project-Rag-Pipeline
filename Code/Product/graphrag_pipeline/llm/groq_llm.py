from __future__ import annotations

import logging
import os
from typing import Optional

from .base import LLMInterface

logger = logging.getLogger(__name__)


class GroqLLM(LLMInterface):
    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        try:
            from groq import AsyncGroq
        except ImportError as exc:
            raise ImportError("Install the 'groq' package: pip install groq") from exc

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        self._client = AsyncGroq(api_key=api_key, max_retries=0)

    async def ainvoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        import time
        start_time = time.time()
        
        async def _call():
            return await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            
        response = await self._execute_with_retry("Groq", _call)
        
        duration_sec = time.time() - start_time
        
        content = response.choices[0].message.content or ""
        
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0
            
        self._update_and_log_usage(prompt_tokens, completion_tokens, duration_sec)
        
        logger.debug("Groq response (%s tokens): %s...", len(content), content[:200])
        return content
