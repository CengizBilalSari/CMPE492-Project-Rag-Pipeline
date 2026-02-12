import logging
from typing import List, Optional

from loaders.registry import build_default_loader_registry
from splitters.splitters import build_splitter
from .generator import QuestionLibraryGenerator
from .models import QAEntry, PipelineConfig
from .rag import RAGClient


class EvaluationPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.loader_registry = build_default_loader_registry()
        self.splitter = build_splitter(
            config.splitter_type, config.chunk_size, config.chunk_overlap
        )
        self.generator = QuestionLibraryGenerator(config.llm_provider, config.llm_model)
        self.rag_client = (
            RAGClient(config.rag_endpoint_url, config.rag_http_method, config.rag_top_k)
            if config.rag_endpoint_url
            else None
        )

    def run(
        self,
        file_path: str,
        human_labeled: Optional[List[QAEntry]] = None,
    ) -> List[QAEntry]:
        raw_text = self.loader_registry.load(file_path)
        chunks = self.splitter.split(raw_text)
        all_entries: List[QAEntry] = []

        for idx, chunk in enumerate(chunks):
            logging.info("Generating questions for chunk %s/%s", idx + 1, len(chunks))
            entries = self.generator.generate_suite(
                chunk,
                K=self.config.personas_k,
                N=self.config.tasks_n,
                M=self.config.qas_m,
            )
            for entry in entries:
                entry.chunk_id = idx
                entry.chunk = chunk
                entry.file_path = file_path
                if self.rag_client:
                    entry.retrieved_contexts = self.rag_client.retrieve(entry.question)
            all_entries.extend(entries)

        if human_labeled:
            for entry in human_labeled:
                entry.source = entry.source or "human"
                entry.file_path = entry.file_path or file_path
            all_entries.extend(human_labeled)

        return all_entries
