"""SQLite engine and session management.

* Foreign keys are enforced on every connection (``PRAGMA foreign_keys=ON``).
* WAL journaling lets readers continue while one writer commits.
* ``session_scope`` commits on success and rolls back on any exception.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Iterator
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings

_lock = threading.Lock()
_engine: Engine | None = None
_engine_path: Path | None = None
_factory: sessionmaker[Session] | None = None


def _on_connect(dbapi_connection, _connection_record) -> None:  # noqa: ANN001 - DB-API objects
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=15000")
    finally:
        cursor.close()


def get_engine() -> Engine:
    """Return the engine for the configured database file (created lazily)."""
    global _engine, _engine_path, _factory
    path = get_settings().database_path.resolve()
    with _lock:
        if _engine is None or _engine_path != path:
            if _engine is not None:
                _engine.dispose()
            path.parent.mkdir(parents=True, exist_ok=True)
            engine = create_engine(
                f"sqlite:///{path.as_posix()}",
                connect_args={"check_same_thread": False, "timeout": 30},
            )
            event.listen(engine, "connect", _on_connect)
            _engine, _engine_path = engine, path
            _factory = sessionmaker(bind=engine, expire_on_commit=False)
        return _engine


def reset_engine() -> None:
    """Dispose the engine (tests use this to release SQLite file handles)."""
    global _engine, _engine_path, _factory
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine, _engine_path, _factory = None, None, None


def new_session() -> Session:
    get_engine()
    assert _factory is not None
    return _factory()


def begin_write(session: Session) -> None:
    """Take SQLite's write lock for the rest of ``session``'s transaction (``BEGIN IMMEDIATE``).

    pysqlite only sends ``BEGIN`` before the first INSERT or UPDATE, so reads
    that decide what to write (a "did it change?" check, the next version
    number, a quota) would otherwise run outside the write transaction. Call
    this right before such check-then-write sequences; keep heavy work (file
    I/O, scoring) before it, because every other writer waits while the lock is
    held. A no-op when the connection already holds a write transaction. The
    lock is released by the session's commit or rollback.
    """
    driver = session.connection().connection.driver_connection
    if not driver.in_transaction:
        session.connection().exec_driver_sql("BEGIN IMMEDIATE")


@contextlib.contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope: commit on success, roll back on any exception."""
    session = new_session()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
