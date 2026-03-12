"""
Unified LLM interface with factory function.
"""

from .base import LLMInterface
from .openai_llm import OpenAILLM
from .groq_llm import GroqLLM
from .vertex_llm import VertexLLM
from .ollama_llm import OllamaLLM
from .vllm_llm import VllmLLM


def get_llm(provider: str, **kwargs) -> LLMInterface:
    """Factory function to create an LLM instance.

    Args:
        provider: One of 'openai', 'groq', 'vertex', 'ollama', or 'vllm'.
        **kwargs: Forwarded to the LLM constructor (model, temperature, max_tokens).

    Returns:
        An LLMInterface implementation.

    Raises:
        ValueError: If the provider is not supported.
    """
    providers = {
        "openai": OpenAILLM,
        "groq": GroqLLM,
        "vertex": VertexLLM,
        "ollama": OllamaLLM,
        "vllm": VllmLLM,
    }
    cls = providers.get(provider.lower())
    if cls is None:
        raise ValueError(f"Unsupported LLM provider '{provider}'. Choose from: {list(providers.keys())}")
    return cls(**kwargs)


__all__ = ["LLMInterface", "OpenAILLM", "GroqLLM", "VertexLLM", "OllamaLLM", "VllmLLM", "get_llm"]

