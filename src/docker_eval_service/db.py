"""Database setup: connect, apply SQL migrations, fail loudly.

Migrations are plain SQL files in ``migrations/`` (``001_*.sql``,
``002_*.sql``, ...). Each is applied exactly once, tracked in the
``schema_migrations`` table. If Postgres cannot be reached, or a
migration fails, this raises — the service must never start against
a half-initialised database, and it must never silently fall back
to another store.
"""

from __future__ import annotations

import os
from pathlib import Path

def _candidate_dirs() -> list[Path]:
    """Where migration files may live, in preference order.

    The service runs in two layouts: from a repo checkout (editable
    install), where migrations sit at the project root, and from the
    Docker image, where they are copied to /app/migrations and the
    code lives in site-packages. ``MIGRATIONS_DIR`` overrides both.
    """
    package_root = Path(__file__).resolve().parent.parent.parent
    candidates = []
    if os.environ.get("MIGRATIONS_DIR"):
        candidates.append(Path(os.environ["MIGRATIONS_DIR"]))
    candidates.append(package_root / "migrations")
    candidates.append(Path.cwd() / "migrations")
    return candidates


def resolve_migrations_dir() -> Path:
    """First candidate directory that exists and holds ``*.sql`` files.

    Raises RuntimeError when none does — the service must fail
    loudly rather than start with no schema.
    """
    for candidate in _candidate_dirs():
        if candidate.is_dir() and list(candidate.glob("*.sql")):
            return candidate
    raise RuntimeError(
        "No migration files found. Looked in: "
        + ", ".join(str(c) for c in _candidate_dirs())
        + ". Set MIGRATIONS_DIR to the directory holding the *.sql files."
    )


def list_migrations(migrations_dir: Path | None = None) -> list[Path]:
    """Migration files in application order (sorted by filename)."""
    directory = migrations_dir or resolve_migrations_dir()
    return sorted(p for p in directory.glob("*.sql"))


def get_database_url() -> str:
    """Read DATABASE_URL from the environment.

    Raises RuntimeError when it is missing — the service has no
    default database and no fallback.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Set it to a PostgreSQL connection "
            "string, e.g. postgresql://eval:evalpass@postgres:5432/evaldb. "
            "The service does not fall back to any other database."
        )
    return url


def apply_migrations(conninfo: str, migrations_dir: Path | None = None) -> list[str]:
    """Apply every pending migration. Returns the versions applied.

    Raises the underlying database error on failure — callers must
    treat that as fatal.
    """
    import psycopg

    directory = migrations_dir or resolve_migrations_dir()
    applied: list[str] = []
    with psycopg.connect(conninfo, autocommit=True) as conn:
        with conn.cursor() as cur:
            # The tracking table must exist before we can ask it
            # anything; the first migration creates it, so create it
            # here too if this is a brand-new database.
            cur.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "  version TEXT PRIMARY KEY,"
                "  applied_at TIMESTAMPTZ NOT NULL DEFAULT now()"
                ")"
            )
            cur.execute("SELECT version FROM schema_migrations")
            done = {row[0] for row in cur.fetchall()}
        for path in list_migrations(directory):
            version = path.stem
            if version in done:
                continue
            sql = path.read_text(encoding="utf-8")
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (version) VALUES (%s)",
                    (version,),
                )
            applied.append(version)
    return applied


def init_database(conninfo: str) -> list[str]:
    """Connect and apply migrations. Raises on any failure."""
    try:
        return apply_migrations(conninfo)
    except Exception as exc:
        raise RuntimeError(
            f"Could not initialise the database at startup: {exc}. "
            "Check DATABASE_URL and that PostgreSQL is running and reachable."
        ) from exc
