"""
Abstract base class for LLM providers.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    """Unified async interface for large language models.

    Args:
        model: Model identifier (e.g. 'gpt-4o', 'llama-3.3-70b-versatile').
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in the response.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ----- public API ----- #

    @abstractmethod
    async def ainvoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Send a prompt to the LLM asynchronously and return the text response.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system-level instruction.

        Returns:
            The model's text response.
        """
        ...

    def invoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Synchronous wrapper around :meth:`ainvoke`."""
        return asyncio.get_event_loop().run_until_complete(
            self.ainvoke(prompt, system_prompt)
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"
