"""
Parallel Entity-Relationship extraction using an LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMInterface

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """An extracted entity."""
    name: str
    type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    """An extracted (Subject, Predicate, Object) triplet."""
    subject: str
    predicate: str
    object: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Holds entities and relations extracted from a single chunk."""
    chunk_index: int
    chunk_text: str
    entities: list[Entity]
    relations: list[Relation]


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

ER_SYSTEM_PROMPT = """You are a knowledge-graph extraction expert.

Given a text chunk, extract all entities and their relationships.

Return ONLY valid JSON with this exact schema:
{
  "entities": [
    {"name": "...", "type": "...", "description": "..."}
  ],
  "relations": [
    {"subject": "...", "predicate": "...", "object": "...", "properties": {}}
  ]
}

Rules:
- Entity names should be canonical (proper casing, no duplicates).
- Entity types should be general categories (Person, Organization, Location, Concept, Event, Date, etc.).
- Predicates should be descriptive verb phrases in UPPER_SNAKE_CASE (e.g., WORKS_AT, LOCATED_IN).
- subject and object in relations must exactly match an entity name.
- If no entities or relations are found, return empty arrays.
- Return ONLY the JSON, no markdown fences, no explanation."""

# TODO(the entity types should be more specific, this prompt could create problem for medicine domain)

# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class EntityRelationshipExtractor:
    """Extracts entities and relations from text chunks using an LLM.

    Processes multiple chunks in parallel with configurable concurrency.

    Args:
        llm: The language model to use.
        max_concurrency: Maximum number of parallel LLM calls.
    """

    def __init__(self, llm: LLMInterface, max_concurrency: int = 5) -> None:
        self.llm = llm
        self.max_concurrency = max_concurrency

    async def extract_from_chunks(
        self,
        chunks: list[str],
    ) -> list[ExtractionResult]:
        """Extract entities and relations from all *chunks* in parallel.

        Args:
            chunks: List of text chunks.

        Returns:
            A list of ``ExtractionResult`` objects, one per chunk.
        """
        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = [
            self._extract_single(sem, idx, chunk)
            for idx, chunk in enumerate(chunks)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        extraction_results: list[ExtractionResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Extraction failed for chunk %d: %s", i, result)
                extraction_results.append(
                    ExtractionResult(chunk_index=i, chunk_text=chunks[i], entities=[], relations=[])
                )
            else:
                extraction_results.append(result)
        return extraction_results

    async def _extract_single(
        self,
        sem: asyncio.Semaphore,
        index: int,
        chunk: str,
    ) -> ExtractionResult:
        async with sem:
            logger.info("Extracting entities from chunk %d (%d chars)", index, len(chunk))
            response = await self.llm.ainvoke(prompt=chunk, system_prompt=ER_SYSTEM_PROMPT)
            await asyncio.sleep(1.5)  # Rate-limit guard for Groq free tier (unit: second)
            parsed = self._parse_response(response)
            return ExtractionResult(
                chunk_index=index,
                chunk_text=chunk,
                entities=parsed["entities"],
                relations=parsed["relations"],
            )

    @staticmethod
    def _parse_response(response: str) -> dict[str, list]:
        """Parse the JSON response from the LLM, with fallback on error."""
        try:
            clean = response.strip()
            # Strip markdown code fences if present
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(clean)

            entities = [
                Entity(
                    name=e.get("name", ""),
                    type=e.get("type", "Unknown"),
                    description=e.get("description", ""),
                    properties=e.get("properties", {}),
                )
                for e in data.get("entities", [])
            ]
            relations = [
                Relation(
                    subject=r.get("subject", ""),
                    predicate=r.get("predicate", ""),
                    object=r.get("object", ""),
                    properties=r.get("properties", {}),
                )
                for r in data.get("relations", [])
            ]
            return {"entities": entities, "relations": relations}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse extraction response: %s", exc)
            return {"entities": [], "relations": []}
