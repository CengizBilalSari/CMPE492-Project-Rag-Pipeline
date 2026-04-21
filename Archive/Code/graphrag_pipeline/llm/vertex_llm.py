"""
Vertex AI LLM implementation.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .base import LLMInterface

logger = logging.getLogger(__name__)


class VertexLLM(LLMInterface):
    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        
        try:
            from langchain_google_vertexai import ChatVertexAI
        except ImportError as exc:
            raise ImportError("Install the 'langchain-google-vertexai' package: pip install langchain-google-vertexai") from exc

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
                max_retries=0,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Vertex AI client: {exc}") from exc

    async def ainvoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))

        import time
        start_time = time.time()
        
        async def _call():
            return await self._client.ainvoke(messages)
            
        response = await self._execute_with_retry("Vertex AI", _call)
        
        duration_sec = time.time() - start_time
        
        content = str(response.content)
        
        prompt_tokens = 0
        completion_tokens = 0
        
        if hasattr(response, "usage_metadata") and response.usage_metadata is not None:
            prompt_tokens = response.usage_metadata.get("input_tokens", 0)
            completion_tokens = response.usage_metadata.get("output_tokens", 0)
        elif hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            usage = response.response_metadata["token_usage"]
            prompt_tokens = getattr(usage, "prompt_token_count", 0)
            if not isinstance(prompt_tokens, int):
                prompt_tokens = usage.get("prompt_token_count", 0)
            completion_tokens = getattr(usage, "candidates_token_count", 0)
            if not isinstance(completion_tokens, int):
                completion_tokens = usage.get("candidates_token_count", 0)
                
        self._update_and_log_usage(prompt_tokens, completion_tokens, duration_sec)
        
        logger.debug("Vertex AI response (%s chars): %s...", len(content), content[:200])
        return content
