from __future__ import annotations

import json
import logging
import os
from typing import List

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
            self.base_url = os.getenv("LMSTUDIO_BASE_URL")
            if not self.base_url:
                raise ValueError("Missing LMSTUDIO_BASE_URL in .env file for lmstudio provider")
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY in .env file")
            self.client = OpenAI(api_key=api_key)

    def generate_from_text(
        self,
        text: str,
        num_questions: int = 10,
        chunk_strategy: str = "recursive",
    ) -> List[dict]:
        """Generate QA pairs from raw document text, partitioned into Global and Local questions."""
        import math
        
        num_global = num_questions // 2
        num_local = num_questions - num_global
        
        all_qa_pairs = []
        
        # --- 1. GLOBAL QUESTIONS ---
        if chunk_strategy == "summarization":
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=1500)
            raw_chunks = splitter.split_text(text)
            
            summaries = []
            for raw in raw_chunks:
                sys_prompt = "You are an expert summarizer. Summarize the following document chunk, highlighting main themes, relationships, and important narratives. Return only the summary text."
                try:
                    if self.provider == "lmstudio":
                        import httpx
                        payload = {"model": self.model, "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": raw}], "temperature": 0.0, "max_tokens": 2048}
                        res = httpx.post(self.base_url, json=payload, timeout=120.0)
                        res.raise_for_status()
                        summary = res.json()["choices"][0]["message"]["content"] or ""
                    else:
                        resp = self.client.chat.completions.create(model=self.model, messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": raw}])
                        summary = resp.choices[0].message.content or ""
                    if summary:
                        summaries.append(summary)
                except Exception as exc:
                    logger.error("Failed to summarize chunk: %s", exc)
            
            combined_summary = "\n\n".join(summaries)
            global_chunks = [combined_summary] if combined_summary else [text[:10000]]
            
        elif chunk_strategy == "semantic":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                global_splitter = LCSemanticChunker(embeddings=embeddings)
                docs = global_splitter.create_documents([text])
                global_chunks = [doc.page_content for doc in docs]
            except ImportError:
                logger.error("langchain_experimental or langchain_huggingface is missing.")
                global_chunks = [text[:2000]]
                
        elif chunk_strategy == "recursive":
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            global_splitter = RecursiveCharacterTextSplitter(
                chunk_size=10000,
                chunk_overlap=1500,
                separators=["\n\n", "\n", " ", ""]
            )
            global_chunks = global_splitter.split_text(text)
        else: # "character"
            global_chunk_size = 2000
            global_chunks = [text[i:i+global_chunk_size] for i in range(0, len(text), global_chunk_size)]
        
        if len(global_chunks) > num_global:
            step = max(1, len(global_chunks) // num_global)
            global_chunks = global_chunks[::step][:num_global]
            
        q_per_global = max(1, num_global // len(global_chunks)) if global_chunks else 0
        global_rem = num_global % len(global_chunks) if global_chunks else 0
        
        for i, chunk in enumerate(global_chunks):
            q_count = q_per_global + (1 if i < global_rem else 0)
            if q_count <= 0:
                continue

            prompt = f"""You are an expert evaluation engineer.

=== DOCUMENT CHUNK ===
{chunk}
=== END DOCUMENT CHUNK ===

Generate exactly {q_count} GLOBAL question-answer pairs from this chunk.

CRITICAL RULES FOR GLOBAL QUESTIONS:
1. Global questions require high-level synthesis. Ask about overarching themes, continuous narratives, or broad aggregated trends across multiple parts of the text.
2. The questions should NOT be answerable by locating a single specific sentence. They must require synthesizing the entire chunk.
3. Every question MUST be answerable ONLY from the document chunk above.
4. Every answer MUST be derived strictly from information in the chunk.

Return ONLY a valid JSON object:
{{
  "qa_pairs": [ {{"question": "...", "ground_truth_answer": "..."}} ]
}}"""

            try:
                import re
                if self.provider == "lmstudio":
                    import httpx
                    payload = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 2048
                    }
                    res = httpx.post(self.base_url, json=payload, timeout=120.0)
                    res.raise_for_status()
                    raw_response = res.json()["choices"][0]["message"]["content"] or ""
                else:
                    kwargs = {"response_format": {"type": "json_object"}}
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )
                    raw_response = resp.choices[0].message.content or ""
                clean = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
                if clean.startswith("```"):
                   clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
                
                try:
                    data = json.loads(clean)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                    else:
                        raise ValueError("No JSON found")

                for p in data.get("qa_pairs", []):
                    if p and p.get("question") and p.get("ground_truth_answer") and len(all_qa_pairs) < num_global:
                        all_qa_pairs.append({
                            "question": f"[Global] {p['question']}", 
                            "ground_truth_answer": p["ground_truth_answer"]
                        })
            except Exception as exc:
                logger.error("Failed to parse GLOBAL QA pairs: %s", exc)


        # --- 2. LOCAL QUESTIONS ---
        if chunk_strategy == "semantic":
            try:
                from langchain_huggingface import HuggingFaceEmbeddings
                from langchain_experimental.text_splitter import SemanticChunker as LCSemanticChunker
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                local_splitter = LCSemanticChunker(embeddings=embeddings)
                docs = local_splitter.create_documents([text])
                local_chunks = [doc.page_content for doc in docs]
            except ImportError:
                local_chunks = [text[:500]]
                
        elif chunk_strategy in ("recursive", "summarization"):
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            local_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=400,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            local_chunks = local_splitter.split_text(text)
        else: # "character"
            local_chunk_size = 500
            local_chunks = [text[i:i+local_chunk_size] for i in range(0, len(text), local_chunk_size)]
        
        if len(local_chunks) > num_local:
            step = max(1, len(local_chunks) // num_local)
            local_chunks = local_chunks[::step][:num_local]
            
        q_per_local = max(1, num_local // len(local_chunks)) if local_chunks else 0
        local_rem = num_local % len(local_chunks) if local_chunks else 0
        
        for i, chunk in enumerate(local_chunks):
            q_count = q_per_local + (1 if i < local_rem else 0)
            if q_count <= 0:
                continue

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
                import re
                if self.provider == "lmstudio":
                    import httpx
                    payload = {
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 2048
                    }
                    res = httpx.post(self.base_url, json=payload, timeout=120.0)
                    res.raise_for_status()
                    raw_response = res.json()["choices"][0]["message"]["content"] or ""
                else:
                    kwargs = {"response_format": {"type": "json_object"}}
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        **kwargs
                    )
                    raw_response = resp.choices[0].message.content or ""
                clean = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
                if clean.startswith("```"):
                   clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
                
                try:
                    data = json.loads(clean)
                except json.JSONDecodeError:
                    match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
                    if match:
                        data = json.loads(match.group(0))
                    else:
                        raise ValueError("No JSON found")

                for p in data.get("qa_pairs", []):
                    if p and p.get("question") and p.get("ground_truth_answer") and len(all_qa_pairs) < num_questions:
                        all_qa_pairs.append({
                            "question": f"[Local] {p['question']}", 
                            "ground_truth_answer": p["ground_truth_answer"]
                        })
            except Exception as exc:
                logger.error("Failed to parse LOCAL QA pairs: %s", exc)

        return all_qa_pairs[:num_questions]
