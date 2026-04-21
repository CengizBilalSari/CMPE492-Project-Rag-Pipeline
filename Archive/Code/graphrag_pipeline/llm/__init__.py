"""
Unified LLM interface with factory function.
"""

from .base import LLMInterface

def get_llm(provider: str, **kwargs) -> LLMInterface:
    """Factory function to create an LLM instance."""
    p = provider.lower()
    if p == "openai":
        from .openai_llm import OpenAILLM
        return OpenAILLM(**kwargs)
    elif p == "groq":
        from .groq_llm import GroqLLM
        return GroqLLM(**kwargs)
    elif p == "vertex":
        from .vertex_llm import VertexLLM
        return VertexLLM(**kwargs)
    elif p == "ollama":
        from .ollama_llm import OllamaLLM
        return OllamaLLM(**kwargs)
    elif p == "vllm":
        from .vllm_llm import VllmLLM
        return VllmLLM(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM provider '{provider}'. Choose from: openai, groq, vertex, ollama, vllm")


__all__ = ["LLMInterface", "OpenAILLM", "GroqLLM", "VertexLLM", "OllamaLLM", "VllmLLM", "get_llm"]

