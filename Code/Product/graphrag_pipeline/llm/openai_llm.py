from __future__ import annotations

import logging
import os
from typing import Optional

from .base import LLMInterface

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("Install the 'openai' package: pip install openai") from exc

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self._client = AsyncOpenAI(api_key=api_key)

    async def ainvoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.debug("OpenAI response (%s tokens): %s...", len(content), content[:200])
        return content
