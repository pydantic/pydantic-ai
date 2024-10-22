"""Example demonstrating how to use Pydantic AI to generate SQL queries based on user input.

Run with:

    uv run --extra examples -m examples.sql_gen
"""

import asyncio
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from typing import Annotated, Any, cast

import asyncpg
import logfire
from annotated_types import MinLen
from devtools import debug

from pydantic_ai import Agent, CallContext, ModelRetry
from pydantic_ai.agent import KnownModelName

# 'if-token-present' means nothing will be sent (and the example wil work) if you don't have logfire set up
logfire.configure()

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    created_at timestamptz,
    start_timestamp timestamptz,
    end_timestamp timestamptz,
    trace_id text,
    span_id text,
    parent_span_id text,
    level log_level,
    span_name text,
    message text,
    attributes_json_schema text,
    attributes jsonb,
    tags text[],
    is_exception boolean,
    otel_status_message text,
    service_name text
);
"""


@dataclass
class Response:
    sql_query: Annotated[str, MinLen(1)]


@dataclass
class Deps:
    conn: asyncpg.Connection


model = cast(KnownModelName, os.getenv('PYDANTIC_AI_MODEL', 'gemini-1.5-flash'))
agent: Agent[Deps, Response] = Agent(model, result_type=Response)


@agent.system_prompt
async def system_prompt() -> str:
    return f"""\
Given the following PostgreSQL table of records, your job is to write a SQL query that suits the user's request.

{DB_SCHEMA}

today's date = {date.today()}

Example
    request: show me records where foobar is false
    response: SELECT * FROM records WHERE attributes->>'foobar' = false
Example
    request: show me records where attributes include the key "foobar"
    response: SELECT * FROM records WHERE attributes ? 'foobar'
Example
    request: show me records from yesterday
    response: SELECT * FROM records WHERE start_timestamp::date > CURRENT_TIMESTAMP - INTERVAL '1 day'
Example
    request: show me error records with the tag "foobar"
    response: SELECT * FROM records WHERE level = 'error' and 'foobar' = ANY(tags)
"""


@agent.result_validator
async def validate_result(ctx: CallContext[Deps], result: Response) -> Response:
    result.sql_query = result.sql_query.replace('\\', '')
    lower_query = result.sql_query.lower()
    if not lower_query.startswith('select'):
        raise ModelRetry('Please a SELECT query')

    try:
        await ctx.deps.conn.execute(f'EXPLAIN {result.sql_query}')
    except asyncpg.exceptions.PostgresError as e:
        raise ModelRetry(f'Invalid query: {e}') from e
    else:
        return result


async def main():
    if len(sys.argv) == 1:
        prompt = 'show me logs from yesterday, with level "error"'
    else:
        prompt = sys.argv[1]

    async with database_connect('postgresql://postgres@localhost', 'pydantic_ai_sql_gen') as conn:
        deps = Deps(conn)
        result = await agent.run(prompt, deps=deps)
    debug(result.response.sql_query)


# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
@asynccontextmanager
async def database_connect(server_dsn: str, database: str) -> AsyncGenerator[Any, None]:
    with logfire.span('check and create DB'):
        conn = await asyncpg.connect(server_dsn)
        try:
            db_exists = await conn.fetchval('SELECT 1 FROM pg_database WHERE datname = $1', database)
            if not db_exists:
                await conn.execute(f'CREATE DATABASE {database}')
        finally:
            await conn.close()

    conn = await asyncpg.connect(f'{server_dsn}/{database}')
    try:
        with logfire.span('create schema'):
            async with conn.transaction():
                if not db_exists:
                    await conn.execute(
                        "CREATE TYPE log_level AS ENUM ('debug', 'info', 'warning', 'error', 'critical')"
                    )
                await conn.execute(DB_SCHEMA)
        yield conn
    finally:
        await conn.close()


if __name__ == '__main__':
    asyncio.run(main())
