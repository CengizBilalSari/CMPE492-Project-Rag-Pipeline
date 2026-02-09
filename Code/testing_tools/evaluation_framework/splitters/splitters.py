import re
from typing import List

import tiktoken


class BaseSplitter:
    def split(self, text: str) -> List[str]:
        raise NotImplementedError


class TokenSplitter(BaseSplitter):
    def __init__(self, chunk_size: int, overlap: int, encoding_name: str = "cl100k_base"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def split(self, text: str) -> List[str]:
        tokens = self.tokenizer.encode(text)
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(tokens), step):
            chunk_tokens = tokens[i : i + self.chunk_size]
            chunks.append(self.tokenizer.decode(chunk_tokens))
            if i + self.chunk_size >= len(tokens):
                break
        return chunks


class CharSplitter(BaseSplitter):
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str) -> List[str]:
        chunks = []
        step = max(1, self.chunk_size - self.overlap)
        for i in range(0, len(text), step):
            chunk = text[i : i + self.chunk_size]
            chunks.append(chunk)
            if i + self.chunk_size >= len(text):
                break
        return chunks


class SentenceSplitter(BaseSplitter):
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.sentence_re = re.compile(r"(?<=[.!?])\s+")

    def split(self, text: str) -> List[str]:
        sentences = self.sentence_re.split(text.strip())
        chunks = []
        current = ""
        for sent in sentences:
            if not sent:
                continue
            candidate = f"{current} {sent}".strip()
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            current = sent
        if current:
            chunks.append(current)
        if self.overlap > 0 and len(chunks) > 1:
            overlapped = []
            for i, chunk in enumerate(chunks):
                if i == 0:
                    overlapped.append(chunk)
                    continue
                prev = overlapped[-1]
                prefix = prev[-self.overlap :]
                overlapped.append(f"{prefix} {chunk}".strip())
            chunks = overlapped
        return chunks


def build_splitter(splitter_type: str, chunk_size: int, overlap: int) -> BaseSplitter:
    splitter_type = splitter_type.lower()
    if splitter_type == "token":
        return TokenSplitter(chunk_size, overlap)
    if splitter_type == "char":
        return CharSplitter(chunk_size, overlap)
    if splitter_type == "sentence":
        return SentenceSplitter(chunk_size, overlap)
    raise ValueError(f"Unsupported splitter_type: {splitter_type}")
