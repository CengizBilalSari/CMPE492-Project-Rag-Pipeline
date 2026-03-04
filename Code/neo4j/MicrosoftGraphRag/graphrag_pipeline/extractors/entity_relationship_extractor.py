from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMInterface

logger = logging.getLogger(__name__)


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
            response = await self._invoke_with_backoff(chunk)
            parsed = self._parse_response(response)
            return ExtractionResult(
                chunk_index=index,
                chunk_text=chunk,
                entities=parsed["entities"],
                relations=parsed["relations"],
            )

    async def _invoke_with_backoff(self, chunk: str, max_retries: int = 5) -> str:
        """Call the LLM with exponential backoff on 429 / rate-limit errors."""
        wait = 2.0
        for attempt in range(max_retries):
            try:
                return await self.llm.ainvoke(prompt=chunk, system_prompt=ER_SYSTEM_PROMPT)
            except Exception as exc:
                err = str(exc).lower()
                is_rate_limit = "429" in err or "quota" in err or "rate" in err or "resource exhausted" in err
                if is_rate_limit and attempt < max_retries - 1:
                    logger.warning(
                        "Rate limit hit on chunk (attempt %d/%d) — retrying in %.1fs...",
                        attempt + 1, max_retries, wait,
                    )
                    await asyncio.sleep(wait)
                    wait = min(wait * 2, 60.0)  # cap at 60s
                else:
                    raise
        raise RuntimeError("Max retries exceeded for chunk extraction")


    @staticmethod
    def _parse_response(response: str) -> dict[str, list]:
        """Parse the JSON response from the LLM, with fallback on error."""
        try:
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            data = json.loads(clean)

            entities = [
                Entity(
                    name=e.get("name") or "",
                    type=e.get("type") or "Unknown",
                    description=e.get("description") or "",
                    properties=e.get("properties") or {},
                )
                for e in data.get("entities", [])
                if e.get("name")   
            ]
            relations = [
                Relation(
                    subject=r.get("subject") or "",
                    predicate=r.get("predicate") or "",
                    object=r.get("object") or "",
                    properties=r.get("properties") or {},
                )
                for r in data.get("relations", [])
                if r.get("subject") and r.get("predicate") and r.get("object")  # all required
            ]
            return {"entities": entities, "relations": relations}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse extraction response: %s , response: %s", exc, response)
            return {"entities": [], "relations": []}

