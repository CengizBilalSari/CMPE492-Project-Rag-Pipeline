from __future__ import annotations

import logging
import os
from typing import Optional

from .base import LLMInterface

logger = logging.getLogger(__name__)


class VllmLLM(LLMInterface):
    """LLM provider for self-hosted vLLM servers.

    Uses the OpenAI-compatible API that vLLM exposes.
    Set VLLM_API_BASE in your environment or .env file, e.g.:
        VLLM_API_BASE=http://34.89.123.45:8000/v1
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        enable_thinking: bool = False, # <-- ADDED: Toggle for Qwen reasoning
    ) -> None:
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.enable_thinking = enable_thinking # Store the toggle state
        
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError("Install the 'openai' package: pip install openai") from exc

        api_base = os.getenv("VLLM_API_BASE")
        logger.info("VLLM API Base: %s", api_base)
        if not api_base:
            raise ValueError(
                "VLLM_API_BASE environment variable is not set.\n"
                "Set it to your vLLM server URL, e.g.: VLLM_API_BASE=http://<VM_IP>:8000/v1"
            )

        # vLLM doesn't require an API key, but the OpenAI client needs one
        self._client = AsyncOpenAI(
            api_key="dummy",
            base_url=api_base,
            max_retries=0,
        )

    async def ainvoke(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if not self.enable_thinking:
            # Qwen3 soft-switch: append /no_think so the model skips reasoning natively
            prompt = f"{prompt}\n/no_think"

        messages.append({"role": "user", "content": prompt})

        import time
        start_time = time.time()


        async def _call():
            # Build the base arguments
            call_kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            # vLLM's OpenAI-compatible API does not natively support disabling thought 
            # tags via the payload for Qwen models, so we will post-process it instead.
            return await self._client.chat.completions.create(**call_kwargs)

        try:
            response = await self._execute_with_retry("vLLM", _call)
        except Exception as e:
            print(f"❌ [DEBUG] Request failed for URL {full_url}. Error: {e}")
            raise e

        duration_sec = time.time() - start_time

        content = response.choices[0].message.content or ""
        
        # If thinking is disabled, manually strip out the <think> blocks
        if not self.enable_thinking:
            import re
            content = re.sub(r"<think>.*?</think>\n*", "", content, flags=re.DOTALL).strip()

        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, "usage") and response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0

        self._update_and_log_usage(prompt_tokens, completion_tokens, duration_sec)

        logger.info("vLLM response (%s tokens): %s...", len(content), content)
        return content
