from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from .llm_service import LLMInterface

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
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
- Return ONLY the JSON, no markdown fences, no explanation.
- DO NOT include any inline comments (e.g., //) inside the JSON."""


class EntityRelationshipExtractor:
    def __init__(self, llm: LLMInterface, max_concurrency: int = 5) -> None:
        self.llm = llm
        self.max_concurrency = max_concurrency

    async def extract_from_chunks(self, chunks: list[str]) -> list[ExtractionResult]:
        return await self.extract_from_chunks_with_progress(chunks)

    async def extract_from_chunks_with_progress(
        self, chunks: list[str], progress_callback=None,
    ) -> list[ExtractionResult]:
        sem = asyncio.Semaphore(self.max_concurrency)
        total = len(chunks)
        completed_count = 0

        async def _tracked(idx: int, chunk: str) -> ExtractionResult:
            nonlocal completed_count
            result = await self._extract_single(sem, idx, chunk)
            completed_count += 1
            if progress_callback:
                progress_callback(completed_count, total, result)
            return result

        tasks = [_tracked(idx, chunk) for idx, chunk in enumerate(chunks)]
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
        self, sem: asyncio.Semaphore, index: int, chunk: str,
    ) -> ExtractionResult:
        async with sem:
            logger.info("Extracting entities from chunk %d (%d chars)", index, len(chunk))
            response = await self.llm.ainvoke(prompt=chunk, system_prompt=ER_SYSTEM_PROMPT)
            parsed = self._parse_response(response)
            return ExtractionResult(
                chunk_index=index,
                chunk_text=chunk,
                entities=parsed["entities"],
                relations=parsed["relations"],
            )

    @staticmethod
    def _parse_response(response: str) -> dict[str, list]:
        import re
        try:
            clean = response.strip()
            clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL).strip()
            clean = re.sub(r"//.*$", "", clean, flags=re.MULTILINE)
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
            
            # Fallback to regex json extraction if it still fails
            try:
                data = json.loads(clean)
            except json.JSONDecodeError:
                json_match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(0))
                else:
                    raise

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
                if r.get("subject") and r.get("predicate") and r.get("object")
            ]
            return {"entities": entities, "relations": relations}
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse extraction response: %s, response: %s", exc, response)
            return {"entities": [], "relations": []}
