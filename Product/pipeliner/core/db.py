from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def _dsn() -> str:
    return os.getenv(
        "DATABASE_URL",
        "postgresql://graphrag:graphrag@postgres:5432/graphrag",
    )


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(_dsn(), min_size=1, max_size=10, kwargs={"row_factory": dict_row})
        _pool.wait()
        logger.info("Postgres connection pool initialized.")
    return _pool


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    with get_pool().connection() as conn:
        yield conn
