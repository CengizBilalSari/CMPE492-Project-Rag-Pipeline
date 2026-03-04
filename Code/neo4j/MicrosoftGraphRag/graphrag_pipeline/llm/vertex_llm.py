"""
Vertex AI LLM implementation.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI

from .base import LLMInterface

logger = logging.getLogger(__name__)


class VertexLLM(LLMInterface):
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

        project_id = os.getenv("GCP_PROJECT_ID")
        location = os.getenv("GCP_LOCATION", "global")

        if not project_id:
            raise ValueError("GCP_PROJECT_ID is not set in your .env file.")

        try:
            self._client = ChatVertexAI(
                model=self.model,
                project=project_id,
                location=location,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Vertex AI client: {exc}") from exc

    async def ainvoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        response = await self._client.ainvoke(messages)
        content = str(response.content)
        logger.debug("Vertex AI response (%s chars): %s...", len(content), content[:200])
        return content
