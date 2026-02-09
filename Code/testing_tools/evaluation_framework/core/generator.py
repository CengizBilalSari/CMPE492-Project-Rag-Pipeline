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
Act as an expert evaluation engineer. Based on these user personas:
{personas}

Generate {N} high-level tasks. For each task, generate {M} Q&A pairs. 
You must strictly vary the 'question_type' based on these definitions:
- 'Global': High-level synthesis or "what is the main theme" questions.
- 'Local Direct': Requires connecting 2-3 specific entities or sections.
- 'One Fact': A pinpoint lookup of a single, specific detail.

Constraint: Ensure the 'answer' is comprehensive and derived strictly from the implied context of the corpus.

Output MUST be a valid JSON object following this structure:
{{
  "library": [
    {{
      "persona": "The persona name and a bit detail of it (e.g., 'Alice, a data scientist with 5 years of experience')",
      "task": "The overarching objective",
      "question_type": "Global | Local Direct | One Fact",
      "question": "The specific inquiry",
      "answer": "The detailed ground-truth answer"
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
