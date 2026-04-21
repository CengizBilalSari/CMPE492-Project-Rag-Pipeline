from __future__ import annotations

import logging
import os
from typing import Optional

from .base import LLMInterface

logger = logging.getLogger(__name__)


class OllamaLLM(LLMInterface):
    def __init__(
        self,
        model: str = "llama3",
        temperature: float = 0.0,
        max_tokens: int = 8192,
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        try:
            from ollama import AsyncClient
        except ImportError as exc:
            raise ImportError("Install the 'ollama' package: pip install ollama") from exc

        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self._client = AsyncClient(host=host)

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
            options = {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
            return await self._client.chat(
                model=self.model,
                messages=messages,
                options=options,
            )
            
        response = await self._execute_with_retry("Ollama", _call)
        
        duration_sec = time.time() - start_time
        
        content = response.get("message", {}).get("content", "")
        
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
            
        self._update_and_log_usage(prompt_tokens, completion_tokens, duration_sec)
        
        logger.debug("Ollama response (%s tokens): %s...", len(content), content[:200])
        return content
