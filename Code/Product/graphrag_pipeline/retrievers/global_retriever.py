from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, List, Dict

from ..llm.base import LLMInterface

logger = logging.getLogger(__name__)

MAP_PROMPT = """--- ROLE ---
You are a helpful assistant responding to questions about data in the knowledge graph.

--- GOAL ---
Generate a relevant intermediate response to the user's query based ONLY on the provided list of community summaries. 

Assign an "importance_score" between 0 and 100 to your response, indicating how well it answers the user's query.

--- DATA ---
{context}

--- QUERY ---
{query}

--- RESPONSE FORMAT ---
Return your response in the following JSON format:
{{
  "answer": "Your detailed answer here...",
  "importance_score": 85
}}

If the summaries do not contain information relevant to the query, set the "importance_score" to 0 and provide a brief explanation in the "answer" field.
"""

REDUCE_PROMPT = """--- ROLE ---
You are a helpful assistant responding to questions about data in the knowledge graph.

--- GOAL ---
Combine the following intermediate responses into a single, comprehensive final answer to the user's query.
The intermediate responses have been filtered based on their relevance scores.

--- INTERMEDIATE RESPONSES ---
{responses}

--- QUERY ---
{query}

--- FINAL RESPONSE ---
"""

class GlobalRetriever:
    def __init__(
        self,
        driver: Any,
        llm: LLMInterface,
        token_limit: int = 1000,
        max_concurrency: int = 1,
        top_k: int = 10,
        database: str = "neo4j",
    ) -> None:
        self.driver = driver
        self.llm = llm
        self.token_limit = token_limit
        self.max_concurrency = max_concurrency
        self.top_k = top_k
        self.database = database
        import tiktoken
        self._encoder = tiktoken.get_encoding("cl100k_base")

    async def search(self, query: str) -> str:
        logger.info("[Retriever] Starting global search for query: %s", query)
        logger.info("[Retriever] Fetching community summaries from Neo4j...")
        summaries = self._fetch_all_summaries()
        if not summaries:
            logger.warning("[Retriever] No community summaries found in Neo4j!")
            return "No community summaries found in the knowledge graph. Please index a document first."
        logger.info("[Retriever] Fetched %d community summaries from Neo4j.", len(summaries))

        batches = self._batch_summaries(summaries)
        logger.info("[Retriever] Created %d batch(es) (token_limit=%d per batch).",
                    len(batches), self.token_limit)

        logger.info("[Retriever] MAP phase: sending %d batch(es) to LLM (max_concurrency=%d)...",
                    len(batches), self.max_concurrency)
        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._map_batch(sem, batch, query, idx) for idx, batch in enumerate(batches)]
        intermediate_results = await asyncio.gather(*tasks)
        logger.info("[Retriever] MAP phase complete. Got %d raw LLM responses.", len(intermediate_results))
        parsed_results = []
        for i, res in enumerate(intermediate_results):
            parsed = self._parse_json_result(res)
            if parsed:
                score = parsed.get("importance_score", 0)
                logger.info("[Retriever]   Batch %d/%d -> importance_score=%s  (kept=%s)",
                            i + 1, len(intermediate_results), score, score > 0)
                if score > 0:
                    parsed_results.append(parsed)
            else:
                logger.warning("[Retriever]   Batch %d/%d -> could not parse JSON from LLM response.",
                               i + 1, len(intermediate_results))

        if not parsed_results:
            logger.warning("[Retriever] All batches scored 0 or failed to parse — no relevant context found.")
            return "I couldn't find any relevant information in the community summaries to answer your query."

        parsed_results.sort(key=lambda x: x.get("importance_score", 0), reverse=True)
        top_results = parsed_results[:self.top_k]
        logger.info("[Retriever] Scoring phase: kept top %d/%d result(s). Scores: %s",
                    len(top_results), len(parsed_results),
                    [r.get("importance_score") for r in top_results])

        logger.info("[Retriever] REDUCE phase: combining %d intermediate answer(s) into final response...",
                    len(top_results))
        final_answer = await self._reduce_answers([r["answer"] for r in top_results], query)
        logger.info("[Retriever] REDUCE phase complete. Final answer length: %d chars.", len(final_answer))

        return final_answer

    def _fetch_all_summaries(self) -> List[str]:
        records, _, _ = self.driver.execute_query(
            "MATCH (cs:CommunitySummary) RETURN cs.text AS text",
            database_=self.database,
        )
        return [r["text"] for r in records if r["text"]]

    def _batch_summaries(self, summaries: List[str]) -> List[List[str]]:
        batches = []
        current_batch = []
        current_tokens = 0

        for s in summaries:
            tokens = len(self._encoder.encode(s))
            if current_tokens + tokens > self.token_limit and current_batch:
                batches.append(current_batch)
                current_batch = [s]
                current_tokens = tokens
            else:
                current_batch.append(s)
                current_tokens += tokens
        
        if current_batch:
            batches.append(current_batch)
        
        return batches

    async def _map_batch(self, sem: asyncio.Semaphore, batch: List[str], query: str, batch_idx: int = 0) -> str:
        async with sem:
            logger.info("[Retriever]   MAP batch %d: sending %d summary/summaries to LLM...",
                        batch_idx + 1, len(batch))
            context = "\n\n".join([f"--- Summary ---\n{s}" for s in batch])
            prompt = MAP_PROMPT.format(context=context, query=query)
            response = await self.llm.ainvoke(prompt)
            logger.info("[Retriever]   MAP batch %d: LLM responded (%d chars).",
                        batch_idx + 1, len(response))
            return response

    def _parse_json_result(self, text: str) -> Dict[str, Any] | None:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        logger.warning("Failed to parse JSON response from LLM: %s", text[:100])
        return None

    async def _reduce_answers(self, answers: List[str], query: str) -> str:
        if len(answers) == 1:
            logger.info("[Retriever] Only 1 intermediate answer — skipping reduce LLM call.")
            return answers[0]

        logger.info("[Retriever] Sending %d answers to LLM for final reduce...", len(answers))
        responses_text = "\n\n".join([f"--- Intermediate Response ---\n{a}" for a in answers])
        prompt = REDUCE_PROMPT.format(responses=responses_text, query=query)
        return await self.llm.ainvoke(prompt)
