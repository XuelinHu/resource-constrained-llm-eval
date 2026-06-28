from __future__ import annotations

import psycopg
from psycopg import sql

from .config import settings
from .database import Base, engine


def main() -> None:
    admin_dsn = (
        f"host={settings.db_host} port={settings.db_port} user={settings.db_user} "
        f"password={settings.db_password} dbname=postgres"
    )
    with psycopg.connect(admin_dsn, autocommit=True) as connection:
        exists = connection.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (settings.db_name,)
        ).fetchone()
        if not exists:
            connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(settings.db_name)))
            print(f"created database: {settings.db_name}")
        else:
            print(f"database exists: {settings.db_name}")
    Base.metadata.create_all(bind=engine)
    print("database schema ready")


if __name__ == "__main__":
    main()
