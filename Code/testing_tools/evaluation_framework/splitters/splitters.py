from typing import List

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)


class BaseSplitter:
    def split(self, text: str) -> List[str]:
        raise NotImplementedError


class TokenSplitter(BaseSplitter):
    def __init__(self, chunk_size: int, overlap: int):
        self._splitter = TokenTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )

    def split(self, text: str) -> List[str]:
        return self._splitter.split_text(text)


class CharSplitter(BaseSplitter):
    def __init__(self, chunk_size: int, overlap: int):
        self._splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap, separator=""
        )

    def split(self, text: str) -> List[str]:
        return self._splitter.split_text(text)


class SentenceSplitter(BaseSplitter):
    def __init__(self, chunk_size: int, overlap: int):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=[". ", "! ", "? ", "\n\n", "\n", " ", ""],
        )

    def split(self, text: str) -> List[str]:
        return self._splitter.split_text(text)


def build_splitter(splitter_type: str, chunk_size: int, overlap: int) -> BaseSplitter:
    splitter_type = splitter_type.lower()
    if splitter_type == "token":
        return TokenSplitter(chunk_size, overlap)
    if splitter_type == "char":
        return CharSplitter(chunk_size, overlap)
    if splitter_type == "sentence":
        return SentenceSplitter(chunk_size, overlap)
    raise ValueError(f"Unsupported splitter_type: {splitter_type}")
