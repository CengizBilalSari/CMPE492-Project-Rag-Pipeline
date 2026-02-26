"""
Global Summary Retriever for Microsoft GraphRAG.
Implements a map-reduce approach over community summaries with importance scoring.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, List, Dict

import tiktoken
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
    """Retrieves answers by summarizing community summaries with importance scoring (Map-Reduce).

    Args:
        driver: Neo4j driver.
        llm: Language model for search.
        token_limit: Max tokens per batch for the map phase.
        max_concurrency: Max parallel LLM calls.
        top_k: Number of top-scored intermediate answers to use in reduction.
        database: Neo4j database name.
    """

    def __init__(
        self,
        driver: Any,
        llm: LLMInterface,
        token_limit: int = 2000,
        max_concurrency: int = 5,
        top_k: int = 10,
        database: str = "neo4j",
    ) -> None:
        self.driver = driver
        self.llm = llm
        self.token_limit = token_limit
        self.max_concurrency = max_concurrency
        self.top_k = top_k
        self.database = database
        self._encoder = tiktoken.get_encoding("cl100k_base")

    async def search(self, query: str) -> str:
        """Perform a global search over all community summaries.

        Args:
            query: The user's search query.

        Returns:
            The final reduced answer.
        """
        logger.info("Starting global search (with scoring) for query: %s", query)


        summaries = self._fetch_all_summaries()
        if not summaries:
            return "No community summaries found in the knowledge graph. Please index a document first."


        batches = self._batch_summaries(summaries)
        logger.info("Map phase: processing %d batches of summaries...", len(batches))
        
        sem = asyncio.Semaphore(self.max_concurrency)
        tasks = [self._map_batch(sem, batch, query) for batch in batches]
        intermediate_results = await asyncio.gather(*tasks)

        # Filter out invalid or low-score results
        parsed_results = []
        for res in intermediate_results:
            parsed = self._parse_json_result(res)
            if parsed and parsed.get("importance_score", 0) > 0:
                parsed_results.append(parsed)
        
        if not parsed_results:
            return "I couldn't find any relevant information in the community summaries to answer your query."

        #TODO(It can be changed to max context window size)
        logger.info("Scoring phase: sorting and filtering top %d results...", self.top_k)
        parsed_results.sort(key=lambda x: x.get("importance_score", 0), reverse=True)
        top_results = parsed_results[:self.top_k]

        logger.info("Reduce phase: combining %d top intermediate responses...", len(top_results))
        final_answer = await self._reduce_answers([r["answer"] for r in top_results], query)
        
        return final_answer

    def _fetch_all_summaries(self) -> List[str]:
        """Fetch all CommunitySummary text from Neo4j."""
        records, _, _ = self.driver.execute_query(
            "MATCH (cs:CommunitySummary) RETURN cs.text AS text",
            database_=self.database,
        )
        return [r["text"] for r in records if r["text"]]

    def _batch_summaries(self, summaries: List[str]) -> List[List[str]]:
        """Batch summaries based on token limit."""
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

    async def _map_batch(self, sem: asyncio.Semaphore, batch: List[str], query: str) -> str:
        """Process a single batch of summaries to get an intermediate answer with score."""
        async with sem:
            context = "\n\n".join([f"--- Summary ---\n{s}" for s in batch])
            prompt = MAP_PROMPT.format(context=context, query=query)
            return await self.llm.ainvoke(prompt)

    def _parse_json_result(self, text: str) -> Dict[str, Any] | None:
        """Robustly parse JSON from LLM response."""
        try:
            # Try direct parse
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or text
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        logger.warning("Failed to parse JSON response from LLM: %s", text[:100])
        return None

    async def _reduce_answers(self, answers: List[str], query: str) -> str:
        """Combine intermediate answers into a final response."""
        if len(answers) == 1:
            return answers[0]

        responses_text = "\n\n".join([f"--- Intermediate Response ---\n{a}" for a in answers])
        prompt = REDUCE_PROMPT.format(responses=responses_text, query=query)
        return await self.llm.ainvoke(prompt)
