"""Simple chat app example build with FastAPI.

Run with:

    uv run -m pydantic_ai_examples.chat_app
"""

from __future__ import annotations as _annotations

import asyncio
import sqlite3
from collections.abc import AsyncIterator
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Callable, TypeVar

import fastapi
import logfire
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from opentelemetry.instrumentation.sqlite3 import SQLite3Instrumentor
from pydantic import Field, TypeAdapter
from typing_extensions import ParamSpec

from pydantic_ai import Agent
from pydantic_ai.messages import (
    Message,
    MessagesTypeAdapter,
    ModelTextResponse,
    UserPrompt,
)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

agent = Agent('openai:gpt-4o')
database: Database | None = None


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    global database

    async with Database.connect() as db:
        database = db
        yield
        database = None


app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


@app.get('/')
async def index() -> HTMLResponse:
    return HTMLResponse((THIS_DIR / 'chat_app.html').read_bytes())


@app.get('/chat_app.ts')
async def main_ts() -> Response:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    return Response((THIS_DIR / 'chat_app.ts').read_bytes(), media_type='text/plain')


@app.get('/chat/')
async def get_chat() -> Response:
    assert database is not None, 'Database not initialised'
    msgs = await database.get_messages()
    return Response(
        b'\n'.join(MessageTypeAdapter.dump_json(m) for m in msgs),
        media_type='text/plain',
    )


@app.post('/chat/')
async def post_chat(prompt: Annotated[str, fastapi.Form()]) -> StreamingResponse:
    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # stream the user prompt so that can be displayed straight away
        yield MessageTypeAdapter.dump_json(UserPrompt(content=prompt)) + b'\n'
        # get the chat history so far to pass as context to the agent
        assert database is not None, 'Database not initialised'
        messages = await database.get_messages()
        # run the agent with the user prompt and the chat history
        async with agent.run_stream(prompt, message_history=messages) as result:
            async for text in result.stream(debounce_by=0.01):
                # text here is a `str` and the frontend wants
                # JSON encoded ModelTextResponse, so we create one
                m = ModelTextResponse(content=text, timestamp=result.timestamp())
                yield MessageTypeAdapter.dump_json(m) + b'\n'

        # add new messages (e.g. the user prompt and the agent response in this case) to the database
        await database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type='text/plain')


THIS_DIR = Path(__file__).parent
MessageTypeAdapter: TypeAdapter[Message] = TypeAdapter(
    Annotated[Message, Field(discriminator='role')]
)
P = ParamSpec('P')
R = TypeVar('R')


@dataclass
class Database:
    """Rudimentary database to store chat messages in SQLite."""

    con: sqlite3.Connection
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, file: Path = THIS_DIR / '.chat_app_messages.sqlite'
    ) -> AsyncIterator[Database]:
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        con = await loop.run_in_executor(executor, cls._connect, file)
        slf = cls(con, loop, executor)
        try:
            yield slf
        finally:
            await slf._asyncify(con.close)

    @staticmethod
    def _connect(file: Path) -> sqlite3.Connection:
        con = sqlite3.connect(str(file))
        SQLite3Instrumentor().instrument_connection(con)  # type: ignore
        con.execute(
            'CREATE TABLE IF NOT EXISTS messages (id INT PRIMARY KEY, message_list TEXT);'
        )
        con.commit()
        return con

    async def add_messages(self, messages: bytes):
        await self._asyncify(
            self.con.execute,
            'INSERT INTO messages (message_list) VALUES (?);',
            (messages,),
        )
        await self._asyncify(self.con.commit)

    async def get_messages(self) -> list[Message]:
        c = await self._asyncify(
            self.con.execute, 'SELECT message_list FROM messages order by id desc'
        )
        rows = await self._asyncify(c.fetchall)
        messages: list[Message] = []
        for row in rows:
            messages.extend(MessagesTypeAdapter.validate_json(row[0]))
        return messages

    async def _asyncify(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        assert kwargs == {}, 'kwargs not supported'
        return await self._loop.run_in_executor(self._executor, func, *args)  # type: ignore


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'pydantic_ai_examples.chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )
