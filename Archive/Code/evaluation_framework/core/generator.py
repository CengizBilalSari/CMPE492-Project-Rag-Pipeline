import json
import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
try:
    from groq import Groq
except Exception:  
    Groq = None
from .models import QAEntry

load_dotenv()


class QuestionLibraryGenerator:
    def __init__(self, provider: str, model: str):
        provider = provider.lower()
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("Missing OPENAI_API_KEY in .env file")
            self.client = OpenAI(api_key=api_key)
        elif provider == "groq":
            if Groq is None:
                raise ImportError("groq is not installed. Install with `pip install groq`.")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Missing GROQ_API_KEY in .env file")
            self.client = Groq(api_key=api_key)
        else:
            raise ValueError(f"Unsupported llm_provider: {provider}")
        self.provider = provider
        self.model = model

    def generate_suite(self, corpus_desc: str, K: int, N: int, M: int) -> List[QAEntry]:
        persona_prompt = (
            f"Corpus Description: {corpus_desc}\n"
            f"Generate {K} distinct user personas that represent a realistic cross-section of potential users for this data"
        )
        personas = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": persona_prompt}]
        ).choices[0].message.content

        qa_prompt = f"""
Act as an expert evaluation engineer.

=== CORPUS CONTENT ===
{corpus_desc}
=== END CORPUS ===

=== USER PERSONAS ===
{personas}
=== END PERSONAS ===

Generate {N} high-level tasks. For each task, generate {M} Q&A pairs.

CRITICAL RULES:
1. Every question MUST be answerable ONLY from the CORPUS CONTENT above.
2. Every answer MUST be derived strictly from information found in the CORPUS CONTENT.
3. The persona determines the STYLE and PERSPECTIVE of the question (e.g., technical vs. simple language), but the TOPIC must come from the corpus.
4. Do NOT invent facts about the persona (e.g., their skills, background, research goals). The persona is just a lens through which to phrase the question.
5. Do NOT ask questions about the persona themselves.

You must strictly vary the 'question_type' based on these definitions:
- 'Global': High-level synthesis or "what is the main theme" questions about the corpus.
- 'Local Direct': Requires connecting 2-3 specific entities or sections from the corpus.
- 'One Fact': A pinpoint lookup of a single, specific detail from the corpus.

Output MUST be a valid JSON object following this structure:
{{
  "library": [
    {{
      "persona": "The persona name and a bit detail of it (e.g., 'Alice, a data scientist with 5 years of experience')",
      "task": "The overarching objective",
      "question_type": "Global | Local Direct | One Fact",
      "question": "The specific inquiry (must be answerable from the corpus)",
      "answer": "The detailed ground-truth answer (must come from the corpus)"
    }}
  ]
}}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": qa_prompt}],
            response_format={"type": "json_object"},
        )
        payload = json.loads(response.choices[0].message.content)
        entries = []
        for item in payload.get("library", []):
            entries.append(
                QAEntry(
                    question_type=item.get("question_type", ""),
                    source=self.model,
                    question=item.get("question", ""),
                    answer=item.get("answer", ""),
                    persona=item.get("persona"),
                    task=item.get("task"),
                )
            )
        return entries
