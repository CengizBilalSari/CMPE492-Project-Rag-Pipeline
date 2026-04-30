from __future__ import annotations

import json
import logging
import os
from typing import Callable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generates synthetic QA pairs from document text using an LLM."""

    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        self.provider = provider
        self.model = model
        
        if self.provider == "lmstudio":
            self.base_url = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
            if not self.base_url:
                raise ValueError("Missing LMSTUDIO_BASE_URL in .env file for lmstudio provider")
        elif self.provider == "ollama":
            self.base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY in .env file")
            self.client = OpenAI(api_key=api_key)

    # ── Chunk sizes aligned with GraphRAG ingestion pipeline ──
    # The RAG pipeline ingests documents at chunk_size=1000, chunk_overlap=200
    # (see pipeliner/core/config.py → ChunkingConfig).
    # Local evaluation questions must be generated at the SAME granularity so
    # the questions are answerable from a single retrieval hit.
    LOCAL_CHUNK_SIZE = 1000
    LOCAL_CHUNK_OVERLAP = 200

    # Large window for the Map phase of the summarization pipeline.
    # Each map-chunk is summarized independently before being reduced.
    SUMMARY_MAP_CHUNK_SIZE = 8000
    SUMMARY_MAP_OVERLAP = 800

    def generate_from_text(
        self,
        text: str,
        num_questions: int = 10,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> List[dict]:
        """Generate QA pairs using a dual-strategy pipeline.

        Strategy
        --------
        ┌──────────────────────────────────────────────────────────┐
        │  GLOBAL questions  →  Map-Reduce Summarization          │
        │                                                          │
        │  Step 1 (MAP):   Split the full document into large      │
        │       windows (~15 000 chars, 1 500 overlap).            │
        │       For each window, ask the LLM to produce a         │
        │       focused summary that preserves themes,             │
        │       entity relationships, and narrative arcs.          │
        │                                                          │
        │  Step 2 (REDUCE): Concatenate every per-window summary  │
        │       into a single "super-summary" that covers the     │
        │       entire document in condensed form.                 │
        │                                                          │
        │  Step 3 (GENERATE): Feed that super-summary to the LLM  │
        │       and request high-level, cross-cutting questions    │
        │       that require synthesizing multiple sections.       │
        │       These are the questions best suited for Global     │
        │       Search evaluation.                                 │
        ├──────────────────────────────────────────────────────────┤
        │  LOCAL questions   →  Character Chunking (RAG-aligned)   │
        │                                                          │
        │  Split the raw document into 1 000-char chunks with     │
        │  200-char overlap (matching the GraphRAG ingestion       │
        │  pipeline's default ChunkingConfig).                     │
        │                                                          │
        │  For each selected chunk, ask the LLM to generate       │
        │  pinpoint factual questions (entity names, numbers,      │
        │  relationships) that a Local Search retriever should     │
        │  be able to answer from a single chunk hit.              │
        └──────────────────────────────────────────────────────────┘
        """
        def _progress(msg: str) -> None:
            if on_progress:
                on_progress(msg)

        num_global = num_questions // 2
        num_local = num_questions - num_global

        all_qa_pairs = []

        # ================================================================
        # 1. GLOBAL QUESTIONS — Map-Reduce Summarization
        # ================================================================
        # MAP PHASE: split the document into large windows and summarize
        # each one independently via the LLM.
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        map_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.SUMMARY_MAP_CHUNK_SIZE,
            chunk_overlap=self.SUMMARY_MAP_OVERLAP,
        )
        map_chunks = map_splitter.split_text(text)
        logger.info(
            "Summarization MAP phase: %d chunks of ~%d chars",
            len(map_chunks), self.SUMMARY_MAP_CHUNK_SIZE,
        )
        _progress(f"MAP phase: splitting document into {len(map_chunks)} chunks...")

        summaries: list[str] = []
        for idx, chunk in enumerate(map_chunks):
            sys_prompt = """You are an expert summarizer. Summarize the following document chunk, highlighting main themes, relationships, and important narratives.

            Return ONLY a valid JSON object in the following format:
            {
            "summary": "your summary here"
            }"""
            _progress(f"Summarizing chunk {idx + 1}/{len(map_chunks)}...")
            try:
                raw_response = self._call_llm(
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": chunk},
                    ],
                    json_mode=True,
                )
                data = self._parse_json(raw_response)
                summary = data.get("summary", "")
                if isinstance(summary, (dict, list)):
                    import json
                    summary = json.dumps(summary)
                elif not isinstance(summary, str):
                    summary = str(summary)

                if summary:
                    summaries.append(summary)
                    logger.debug("MAP chunk %d/%d summarized (%d chars)", idx + 1, len(map_chunks), len(summary))
            except Exception as exc:
                logger.error("MAP chunk %d failed: %s", idx + 1, exc)

        # REDUCE PHASE: merge all per-chunk summaries into one document
        combined_summary = "\n\n".join(summaries) if summaries else text[:10000]
        logger.info(
            "Summarization REDUCE phase complete: combined summary %d chars",
            len(combined_summary),
        )

        # GENERATE PHASE: produce global questions from the super-summary
        _progress("Generating global questions from summary...")
        if num_global > 0:
            prompt = f"""You are an expert evaluation engineer.

        === DOCUMENT SUMMARY ===
        {combined_summary}
        === END DOCUMENT SUMMARY ===

        Generate exactly {num_global} GLOBAL question-answer pairs from this summary.

        CRITICAL RULES FOR GLOBAL QUESTIONS:
        1. Global questions require high-level synthesis. Ask about overarching themes, continuous narratives, or broad aggregated trends across the document.
        2. The questions should NOT be answerable by locating a single specific sentence. They must require synthesizing multiple sections.
        3. Every question MUST be answerable ONLY from the summary above.
        4. Every answer MUST be derived strictly from information in the summary.

        Return ONLY a valid JSON object:
        {{
        "qa_pairs": [ {{"question": "...", "ground_truth_answer": "..."}} ]
        }}"""

            try:
                raw_response = self._call_llm(
                    messages=[{"role": "user", "content": prompt}],
                    json_mode=True,
                )
                data = self._parse_json(raw_response)

                for p in data.get("qa_pairs", []):
                    if p and p.get("question") and p.get("ground_truth_answer") and len(all_qa_pairs) < num_global:
                        all_qa_pairs.append({
                            "question": f"[Global] {p['question']}",
                            "ground_truth_answer": p["ground_truth_answer"]
                        })
            except Exception as exc:
                logger.error("Failed to parse GLOBAL QA pairs: %s", exc)

        # ================================================================
        # 2. LOCAL QUESTIONS — Character Chunking (RAG-aligned)
        # ================================================================
        # Use the same chunk_size / overlap as the GraphRAG ingestion
        # pipeline so the generated questions match retriever granularity.
        local_chunks = [
            text[i : i + self.LOCAL_CHUNK_SIZE]
            for i in range(0, len(text), self.LOCAL_CHUNK_SIZE - self.LOCAL_CHUNK_OVERLAP)
        ]

        if len(local_chunks) > num_local:
            step = max(1, len(local_chunks) // num_local)
            local_chunks = local_chunks[::step][:num_local]

        q_per_local = max(1, num_local // len(local_chunks)) if local_chunks else 0
        local_rem = num_local % len(local_chunks) if local_chunks else 0

        for i, chunk in enumerate(local_chunks):
            q_count = q_per_local + (1 if i < local_rem else 0)
            if q_count <= 0:
                continue
            _progress(f"Generating local questions ({i + 1}/{len(local_chunks)})...")

            prompt = f"""You are an expert evaluation engineer.

=== DOCUMENT CHUNK ===
{chunk}
=== END DOCUMENT CHUNK ===

Generate exactly {q_count} LOCAL question-answer pairs from this chunk.

CRITICAL RULES FOR LOCAL QUESTIONS:
1. Local questions target specific facts, pinpoint entity relationships, numerical data, or direct statements found in a specific location in the text.
2. The question should be highly focused (e.g., "What is the relationship between X and Y?", or "What specific value did Z achieve?").
3. Every question MUST be answerable ONLY from the document chunk above.
4. Every answer MUST be derived strictly from information in the chunk.

Return ONLY a valid JSON object:
{{
  "qa_pairs": [ {{"question": "...", "ground_truth_answer": "..."}} ]
}}"""

            try:
                raw_response = self._call_llm(
                    messages=[{"role": "user", "content": prompt}],
                    json_mode=True,
                )
                data = self._parse_json(raw_response)

                for p in data.get("qa_pairs", []):
                    if p and p.get("question") and p.get("ground_truth_answer") and len(all_qa_pairs) < num_questions:
                        all_qa_pairs.append({
                            "question": f"[Local] {p['question']}",
                            "ground_truth_answer": p["ground_truth_answer"]
                        })
            except Exception as exc:
                logger.error("Failed to parse LOCAL QA pairs: %s", exc)

        return all_qa_pairs[:num_questions]

    # ── Private helpers ────────────────────────────────────────

    def _call_llm(self, messages: list[dict], json_mode: bool = False) -> str:
        """Unified LLM call supporting OpenAI, LMStudio, and Ollama providers."""
        if self.provider in ["lmstudio", "ollama"]:
            import httpx
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 16384,
            }
            
            endpoint = self.base_url if self.base_url.endswith("/chat/completions") else f"{self.base_url}/chat/completions"
            res = httpx.post(endpoint, json=payload, timeout=120.0)
            try:
                res.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error("HTTP error %s: %s", res.status_code, res.text)
                raise
            return res.json()["choices"][0]["message"]["content"] or ""
        else:
            kwargs = {}
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )
            return resp.choices[0].message.content or ""

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract a JSON object from an LLM response, handling think-tags and markdown fences."""
        import re
        clean = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        clean = re.sub(r"//.*$", "", clean, flags=re.MULTILINE)
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            return json.loads(clean)
        except json.JSONDecodeError:
            # Try to find the first `{` and extract a valid JSON object character-by-character
            start_idx = clean.find('{')
            if start_idx != -1:
                # Try progressively larger substrings until we get a valid JSON
                for end_idx in range(len(clean), start_idx, -1):
                    if clean[end_idx - 1] == '}':
                        try:
                            return json.loads(clean[start_idx:end_idx])
                        except json.JSONDecodeError:
                            continue
            logger.error("JSON parse failed. Raw response was:\n%s", raw)
            raise ValueError("No JSON found in LLM response")
