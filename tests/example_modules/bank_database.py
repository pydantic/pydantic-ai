from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

DB_SCHEMA = """
CREATE TABLE customers (
    id INT PRIMARY KEY,
    name TEXT NOT NULL
);

INSERT INTO customers (id, name) VALUES (123, 'John'), (456, 'Bob'), (789, 'Charlie');

CREATE TABLE balances (
    id SERIAL PRIMARY KEY,
    customer_id INT NOT NULL REFERENCES customers(id),
    balance FLOAT NOT NULL,
    pending_balance FLOAT NOT NULL
);

INSERT INTO balances (customer_id, balance, pending_balance) VALUES (123, 123.45, 123.45);
"""


# pyright: reportUnknownVariableType=false
# pyright: reportUnknownMemberType=false
@dataclass
class DatabaseConnPostgres:
    _conn: Any

    @classmethod
    @asynccontextmanager
    async def connect(cls):
        import asyncpg

        server_dsn = 'postgres://postgres:postgres@localhost:5432'
        database = 'bank'
        conn = await asyncpg.connect(server_dsn)
        try:
            db_exists = await conn.fetchval('SELECT 1 FROM pg_database WHERE datname = $1', database)
            if not db_exists:
                await conn.execute(f'CREATE DATABASE {database}')
        finally:
            await conn.close()

        conn = await asyncpg.connect(f'{server_dsn}/{database}')
        try:
            if not db_exists:
                await conn.execute(DB_SCHEMA)

            yield cls(conn)
        finally:
            await conn.close()

    async def customer_name(self, *, id: int) -> str | None:
        return await self._conn.fetchval('SELECT name FROM customers WHERE id = $1', id)

    async def customer_balance(self, *, id: int, include_pending: bool) -> float:
        row = await self._conn.fetchrow('SELECT balance, pending_balance FROM balances WHERE customer_id = $1', id)
        if row is None:
            raise ValueError('Customer not found')
        else:
            return row['pending_balance' if include_pending else 'balance']


@dataclass
class DatabaseConnNoop:
    @classmethod
    @asynccontextmanager
    async def connect(cls):
        yield cls()

    async def customer_name(self, *, id: int) -> str | None:
        if id == 123:
            return 'John'

    async def customer_balance(self, *, id: int, include_pending: bool) -> float:
        if id == 123:
            return 123.45
        else:
            raise ValueError('Customer not found')


# to keep tests running quickly we use a no-op database connection for tests
DatabaseConn = DatabaseConnNoop
