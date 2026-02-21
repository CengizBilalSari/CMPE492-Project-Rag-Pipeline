"""
Chunking module — all strategies in one place.

Provides :func:`get_chunker` to instantiate any supported chunking strategy.

Strategies:
  - sentence:      sentence boundary splitting
  - token:         tiktoken-based token splitting
  - character:     raw character count splitting
  - recursive:     recursive character splitting (default)
  - semantic:      embedding-based semantic grouping
  - propositional: LLM → atomic facts → semantic grouping
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..llm.base import LLMInterface

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseChunker(ABC):
    """Interface that every chunking strategy must implement."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split *text* into a list of chunks."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(chunk_size={self.chunk_size}, overlap={self.chunk_overlap})"


# ---------------------------------------------------------------------------
# Simple LangChain-backed strategies
# ---------------------------------------------------------------------------

class SentenceChunker(BaseChunker):
    """Splits text by sentence boundaries (NLTK)."""

    def chunk(self, text: str) -> list[str]:
        from langchain_text_splitters import NLTKTextSplitter
        return NLTKTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
        ).split_text(text)


class TokenChunker(BaseChunker):
    """Splits text by token count (tiktoken)."""

    def chunk(self, text: str) -> list[str]:
        from langchain_text_splitters import TokenTextSplitter
        return TokenTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
        ).split_text(text)


class CharacterChunker(BaseChunker):
    """Splits text by raw character count."""

    def chunk(self, text: str) -> list[str]:
        from langchain_text_splitters import CharacterTextSplitter
        return CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, separator="\n",
        ).split_text(text)


class RecursiveChunker(BaseChunker):
    """Splits text recursively by a hierarchy of separators (default strategy)."""

    def chunk(self, text: str) -> list[str]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap,
        ).split_text(text)


# ---------------------------------------------------------------------------
# Embedding-based strategies
# ---------------------------------------------------------------------------

class SemanticChunker(BaseChunker):
    """Splits text into semantically coherent chunks using HuggingFace embeddings."""

    def __init__(
        self, chunk_size: int = 1000, chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_model = embedding_model

    def chunk(self, text: str) -> list[str]:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker

        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        docs = LCSemanticChunker(embeddings=embeddings).create_documents([text])
        return [doc.page_content for doc in docs]


# ---------------------------------------------------------------------------
# LLM-based propositional strategy
# ---------------------------------------------------------------------------

_PROPOSITION_SYSTEM_PROMPT = """You are a text decomposition expert. Break the following text into a list of atomic, standalone propositions (facts).

Rules:
- Each proposition must be a single, self-contained fact.
- Decontextualize: replace pronouns with the entities they refer to.
- Each proposition should be understandable without reading the original text.
- Return ONLY a JSON array of strings, nothing else.

Example: ["The Eiffel Tower is located in Paris.", "The Eiffel Tower was built in 1889."]"""


class PropositionalChunker(BaseChunker):
    """Two-phase: LLM decomposes text into atomic propositions, then groups by similarity."""

    def __init__(
        self, llm: LLMInterface, chunk_size: int = 1000, chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.llm = llm
        self.embedding_model = embedding_model

    def chunk(self, text: str) -> list[str]:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, self._achunk(text)).result()
        return asyncio.run(self._achunk(text))

    async def _achunk(self, text: str) -> list[str]:
        propositions = await self._generate_propositions(text)
        if not propositions:
            return [text]
        return self._group_propositions(propositions)

    async def _generate_propositions(self, text: str) -> list[str]:
        response = await self.llm.ainvoke(prompt=text, system_prompt=_PROPOSITION_SYSTEM_PROMPT)
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(clean)
            if isinstance(data, list):
                return [str(p) for p in data]
        except (json.JSONDecodeError, IndexError):
            logger.warning("Failed to parse LLM propositions — returning raw chunks.")
        return []

    def _group_propositions(self, propositions: list[str]) -> list[str]:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker

        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        docs = LCSemanticChunker(embeddings=embeddings).create_documents(["\n".join(propositions)])
        return [doc.page_content for doc in docs]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[BaseChunker]] = {
    "sentence": SentenceChunker,
    "token": TokenChunker,
    "character": CharacterChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
    "propositional": PropositionalChunker,
}


def get_chunker(
    strategy: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    llm: LLMInterface | None = None,
    **kwargs: Any,
) -> BaseChunker:
    """Create a chunker by strategy name.

    Args:
        strategy: One of ``sentence``, ``token``, ``character``, ``recursive``,
                  ``semantic``, or ``propositional``.
        chunk_size: Target chunk size.
        chunk_overlap: Overlap between consecutive chunks.
        llm: Required when *strategy* is ``propositional``.

    Raises:
        ValueError: If the strategy is not recognized or propositional is missing llm.
    """
    cls = _REGISTRY.get(strategy.lower())
    if cls is None:
        raise ValueError(f"Unknown chunking strategy '{strategy}'. Choose from: {list(_REGISTRY.keys())}")

    if strategy.lower() == "propositional":
        if llm is None:
            raise ValueError("Propositional chunking requires an LLM instance.")
        return cls(llm=llm, chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    return cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)


__all__ = [
    "BaseChunker", "SentenceChunker", "TokenChunker", "CharacterChunker",
    "RecursiveChunker", "SemanticChunker", "PropositionalChunker", "get_chunker",
]
