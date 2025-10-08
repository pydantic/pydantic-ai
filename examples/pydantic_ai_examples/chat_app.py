"""Simple chat app example build with FastAPI.

Run with:

    uv run -m pydantic_ai_examples.chat_app
"""

from __future__ import annotations as _annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import fastapi
import logfire
from fastapi import Depends, Request, Response

from pydantic_ai import Agent, RunContext
from pydantic_ai.vercel_ai.starlette import StarletteChat

from .sqlite_database import Database

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

THIS_DIR = Path(__file__).parent
sql_schema = """
create table if not exists memory(
    id integer primary key,
    user_id integer not null,
    value text not null,
    unique(user_id, value)
);"""


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect(sql_schema) as db:
        yield {'db': db}


@dataclass
class Deps:
    conn: Database
    user_id: int


chat_agent = Agent(
    'openai:gpt-4.1',
    deps_type=Deps,
    instructions="""
You are a helpful assistant.

Always reply with markdown. ALWAYS use code fences for code examples and lines of code.
""",
)


@chat_agent.tool
async def record_memory(ctx: RunContext[Deps], value: str) -> str:
    """Use this tool to store information in memory."""
    await ctx.deps.conn.execute(
        'insert into memory(user_id, value) values(?, ?) on conflict do nothing',
        ctx.deps.user_id,
        value,
        commit=True,
    )
    return 'Value added to memory.'


@chat_agent.tool
async def retrieve_memories(ctx: RunContext[Deps], memory_contains: str) -> str:
    """Get all memories about the user."""
    rows = await ctx.deps.conn.fetchall(
        'select value from memory where user_id = ? and value like ?',
        ctx.deps.user_id,
        f'%{memory_contains}%',
    )
    return '\n'.join([row[0] for row in rows])


starlette_chat = StarletteChat(chat_agent)
app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


async def get_db(request: Request) -> Database:
    return request.state.db


@app.options('/api/chat')
def options_chat():
    pass


@app.post('/api/chat')
async def get_chat(request: Request, database: Database = Depends(get_db)) -> Response:
    return await starlette_chat.dispatch_request(request, deps=Deps(database, 123))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'pydantic_ai_examples.chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )
