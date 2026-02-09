import csv
import json
from dataclasses import asdict
from typing import Iterable

from .models import QAEntry


class CSVWriter:
    def __init__(self, output_path: str):
        self.output_path = output_path

    def write(self, entries: Iterable[QAEntry]) -> None:
        fieldnames = [
            "question_type",
            "source",
            "question",
            "answer",
            "persona",
            "task",
            "chunk_id",
            "chunk",
            "file_path",
            "retrieved_contexts",
        ]
        with open(self.output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for entry in entries:
                row = asdict(entry)
                row["retrieved_contexts"] = json.dumps(row.get("retrieved_contexts") or [])
                writer.writerow(row)
