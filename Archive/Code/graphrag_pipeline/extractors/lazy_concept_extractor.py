import logging
from typing import List
from itertools import combinations
import asyncio

from graphrag_pipeline.extractors.entity_relationship_extractor import ExtractionResult, Entity, Relation

logger = logging.getLogger(__name__)

class LazyConceptExtractor:
    def __init__(self, nlp_model: str = "en_core_web_lg", max_concurrency: int = 5):
        import spacy
        self.max_concurrency = max_concurrency
        try:
            self.nlp = spacy.load(nlp_model)
        except OSError:
            logger.warning("spacy %s model not found. Attempting download via cli...", nlp_model)
            from spacy.cli import download
            download(nlp_model)
            self.nlp = spacy.load(nlp_model)

    async def extract_from_chunks(self, chunks: List[str]) -> List[ExtractionResult]:
        results = []
        for i, chunk in enumerate(chunks):
            await asyncio.sleep(0)
            
            doc = self.nlp(chunk)
            
            concepts = set()
            for chunk_spacy in doc.noun_chunks:
                text = chunk_spacy.text.strip().lower()
                is_valid = (
                    text and 
                    2 < len(text) < 50 and 
                    chunk_spacy.root.pos_ != "PRON" and
                    not all(token.is_stop or token.is_punct or token.is_digit or token.is_space for token in chunk_spacy)
                )
                if is_valid:
                    concepts.add(text)
                    
            for ent in doc.ents:
                text = ent.text.strip().lower()
                is_valid = (
                    text and 
                    2 < len(text) < 50 and 
                    not all(token.is_stop or token.is_punct or token.is_digit or token.is_space for token in ent)
                )
                if is_valid:
                    concepts.add(text)
            
            entities = [
                Entity(name=concept, type="Concept", description="")
                for concept in concepts
            ]
            
            relations = []
            seen_pairs = set()
            
            for sent in doc.sents:
                sent_text = sent.text.lower()
                sent_concepts = [c for c in concepts if c in sent_text]
                for c1, c2 in combinations(set(sent_concepts), 2):
                    pair = tuple(sorted([c1, c2]))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        relations.append(
                            Relation(subject=c1, object=c2, predicate="CO_OCCURS")
                        )
            
            results.append(ExtractionResult(
                chunk_index=i,
                chunk_text=chunk,
                entities=entities,
                relations=relations
            ))
            
        return results
