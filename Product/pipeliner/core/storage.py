from __future__ import annotations

import logging
from pathlib import Path

from .config import StorageConfig

logger = logging.getLogger(__name__)

STORED_PATH_PREFIX = "/docs"


class LocalDocumentStore:
    """Stores uploaded documents on the local filesystem under {base_path}/{chat_id}/{filename}.

    The `path` persisted in Postgres is always the canonical `/docs/{chat_id}/{filename}` form
    (independent of where base_path actually points), so it stays portable across environments.
    """

    def __init__(self, config: StorageConfig) -> None:
        self.base_path = Path(config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, chat_id: str, filename: str, data: bytes) -> str:
        chat_dir = self.base_path / chat_id
        chat_dir.mkdir(parents=True, exist_ok=True)
        dest = chat_dir / filename
        dest.write_bytes(data)
        logger.info("Saved document: %s (%d bytes)", dest, len(data))
        return f"{STORED_PATH_PREFIX}/{chat_id}/{filename}"

    def read(self, stored_path: str) -> bytes:
        rel = stored_path.removeprefix(f"{STORED_PATH_PREFIX}/")
        full = self.base_path / rel
        if not full.exists():
            raise FileNotFoundError(f"Document not found on disk: {full}")
        return full.read_bytes()
