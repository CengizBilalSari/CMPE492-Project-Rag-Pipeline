"""Microsoft GraphRAG pipeline wrapper.

Wraps the ``graphrag`` library's CLI-based indexing pipeline so it can be
used as a drop-in alternative to the custom Neo4j-based GraphRAGPipeline.

The library handles its own:
  - Chunking
  - Entity / relationship extraction
  - Community detection (Leiden)
  - Community summarisation
  - Embedding

All we need to do is:
  1. Prepare a workspace directory with input text + settings.yaml
  2. Run ``graphrag index --root <workspace>``
  3. Stream stdout/stderr back as status messages
  4. Record the workspace path in PostgreSQL for the evaluator
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import textwrap
import uuid
from pathlib import Path
from typing import AsyncGenerator

from core.config import PipelineConfig

logger = logging.getLogger(__name__)


def _build_settings_yaml(config: PipelineConfig) -> str:
    """Generate a ``settings.yaml`` compatible with the graphrag library.

    We map the user's LLM provider/model selection onto the graphrag
    config format.  For providers other than OpenAI we set the
    ``api_base`` to point at the compatible endpoint.
    """
    provider = config.llm.provider.value  # "openai" | "lmstudio" | "ollama"
    model = config.llm.model

    # Determine API base URL for non-OpenAI providers
    if provider == "lmstudio":
        api_base = os.getenv("LMSTUDIO_BASE_URL", "http://host.docker.internal:1234/v1")
        # Strip /v1 suffix if present — graphrag adds it internally
        api_base = api_base.rstrip("/").removesuffix("/v1")
        model_provider = "openai"  # LMStudio exposes OpenAI-compatible API
    elif provider == "ollama":
        api_base = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434/v1")
        api_base = api_base.rstrip("/").removesuffix("/v1")
        model_provider = "openai"
    else:
        api_base = ""
        model_provider = "openai"

    api_base_line = f'    api_base: "{api_base}"' if api_base else ""

    return textwrap.dedent(f"""\
        completion_models:
          default_completion_model:
            model: "{model}"
            model_provider: "{model_provider}"
            api_key: ${{GRAPHRAG_API_KEY}}
        {api_base_line}
            max_retries: 5

        embedding_models:
          default_embedding_model:
            model: "text-embedding-3-small"
            model_provider: "{model_provider}"
            api_key: ${{GRAPHRAG_API_KEY}}
        {api_base_line}

        chunks:
          size: {config.chunking.chunk_size}
          overlap: {config.chunking.chunk_overlap}

        input:
          type: text
          base_dir: "input"
          file_pattern: ".*\\\\.txt$$"

        storage:
          type: file
          base_dir: "output"

        reporting:
          type: file
          base_dir: "logs"
    """)


class MSGraphRAGPipeline:
    """Wraps the Microsoft GraphRAG library for graph indexing.

    Same streaming interface as ``GraphRAGPipeline.run()`` so the
    WebSocket route can use either one interchangeably.
    """

    def __init__(
        self,
        config: PipelineConfig,
        chat_id: str = "",
        document_id: str = "",
    ) -> None:
        self.config = config
        self.chat_id = chat_id
        self.document_id = document_id
        self._workspace: Path | None = None

    # ── public ────────────────────────────────────────────────────

    async def run(
        self,
        document_text: str,
        doc_title: str = "",
        doc_source: str = "",
    ) -> AsyncGenerator[str, None]:
        """Run the MS GraphRAG indexing pipeline, yielding status messages."""

        yield "Initializing Microsoft GraphRAG pipeline..."

        # 1. Create workspace ------------------------------------------------
        workspace = self._create_workspace(document_text, doc_title)
        self._workspace = workspace
        yield f"Workspace created: {workspace}"

        # 2. Record run in PostgreSQL ----------------------------------------
        run_id = self._record_run(workspace)
        yield "Pipeline run recorded in database."

        # 3. Run graphrag index ----------------------------------------------
        yield "Starting graphrag index (this may take several minutes)..."
        try:
            async for line in self._run_indexing(workspace):
                yield line

            self._complete_run(run_id)
            yield "Microsoft GraphRAG indexing completed successfully."
            yield "Pipeline completed."

        except Exception as exc:
            self._fail_run(run_id, str(exc))
            raise

    def close(self) -> None:
        """No persistent connections to close."""

    # ── workspace setup ───────────────────────────────────────────

    def _create_workspace(self, document_text: str, doc_title: str) -> Path:
        """Create a workspace directory with input text + settings.yaml."""
        base = Path(self.config.storage.ms_graphrag_base)
        base.mkdir(parents=True, exist_ok=True)

        ws_name = hashlib.md5(
            f"{self.chat_id}:{self.document_id}:{doc_title}".encode()
        ).hexdigest()
        workspace = base / ws_name
        workspace.mkdir(parents=True, exist_ok=True)

        # Write input text
        input_dir = workspace / "input"
        input_dir.mkdir(exist_ok=True)
        (input_dir / "document.txt").write_text(document_text, encoding="utf-8")

        # Write settings.yaml
        settings = _build_settings_yaml(self.config)
        (workspace / "settings.yaml").write_text(settings, encoding="utf-8")

        # Write .env with API key
        api_key = os.getenv("OPENAI_API_KEY", "") or "sk-dummy"
        (workspace / ".env").write_text(
            f"GRAPHRAG_API_KEY={api_key}\n", encoding="utf-8"
        )

        return workspace

    # ── subprocess execution ──────────────────────────────────────

    async def _run_indexing(self, workspace: Path) -> AsyncGenerator[str, None]:
        """Run ``graphrag index --root <workspace>`` and stream output."""
        cmd = ["graphrag", "index", "--root", str(workspace)]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ, "GRAPHRAG_API_KEY": os.getenv("OPENAI_API_KEY", "")},
        )

        assert process.stdout is not None
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                logger.info("[ms-graphrag] %s", decoded)
                yield decoded

        returncode = await process.wait()
        if returncode != 0:
            raise RuntimeError(
                f"graphrag index failed with exit code {returncode}"
            )

    # ── database tracking ─────────────────────────────────────────

    def _record_run(self, workspace: Path) -> str:
        """Insert a row into ms_graphrag_runs."""
        from core.db import connection

        run_id = str(uuid.uuid4())
        try:
            with connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO ms_graphrag_runs
                            (id, chat_id, document_id, workspace_path, status)
                        VALUES (%s, %s, %s, %s, 'INDEXING')
                        """,
                        (run_id, self.chat_id, self.document_id or None, str(workspace)),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Failed to record MS GraphRAG run: %s", exc)
        return run_id

    def _complete_run(self, run_id: str) -> None:
        from core.db import connection

        try:
            with connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE ms_graphrag_runs
                        SET status = 'COMPLETED', completed_at = now()
                        WHERE id = %s
                        """,
                        (run_id,),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Failed to complete MS GraphRAG run: %s", exc)

    def _fail_run(self, run_id: str, error: str) -> None:
        from core.db import connection

        try:
            with connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE ms_graphrag_runs
                        SET status = 'FAILED', error = %s, completed_at = now()
                        WHERE id = %s
                        """,
                        (error, run_id),
                    )
                conn.commit()
        except Exception as exc:
            logger.warning("Failed to fail MS GraphRAG run: %s", exc)
