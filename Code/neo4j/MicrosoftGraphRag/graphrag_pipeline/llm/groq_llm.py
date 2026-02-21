"""
Groq LLM implementation.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .base import LLMInterface

logger = logging.getLogger(__name__)


class GroqLLM(LLMInterface):
    """LLM backed by the Groq inference API.

    Reads ``GROQ_API_KEY`` from the environment.
    """

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
        self._client = AsyncGroq(api_key=api_key)

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
        logger.debug("Groq response (%s tokens): %s...", len(content), content[:200])
        return content
