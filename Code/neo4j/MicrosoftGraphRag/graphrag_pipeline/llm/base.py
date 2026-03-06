from __future__ import annotations
import asyncio
import logging
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
