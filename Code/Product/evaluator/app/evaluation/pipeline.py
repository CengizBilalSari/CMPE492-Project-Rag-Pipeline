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
    """Generates synthetic QA pairs from document text using an LLM.

    Adapted from the original EvaluationPipeline's generator to accept
    raw text strings (e.g. fetched from Supabase) instead of file paths.
    """

    def __init__(self, model: str = "gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in .env file")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate_from_text(
        self,
        text: str,
        num_questions: int = 10,
    ) -> List[dict]:
        """Generate QA pairs from raw document text.

        Returns list of dicts with keys: question, ground_truth_answer
        """
        prompt = f"""You are an expert evaluation engineer.

=== DOCUMENT TEXT ===
{text[:8000]}
=== END DOCUMENT ===

Generate exactly {num_questions} question-answer pairs from this document.

CRITICAL RULES:
1. Every question MUST be answerable ONLY from the document text above.
2. Every answer MUST be derived strictly from information in the document.
3. Vary question types: include high-level synthesis ("Global"), entity-linking ("Local Direct"), and pinpoint facts ("One Fact").
4. Questions should test different aspects of the document.

Return ONLY a valid JSON object:
{{
  "qa_pairs": [
    {{"question": "...", "ground_truth_answer": "..."}},
    ...
  ]
}}"""

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        try:
            data = json.loads(raw)
            pairs = data.get("qa_pairs", [])
            return [
                {"question": p["question"], "ground_truth_answer": p["ground_truth_answer"]}
                for p in pairs
                if p.get("question") and p.get("ground_truth_answer")
            ]
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to parse generated QA pairs: %s", exc)
            return []
