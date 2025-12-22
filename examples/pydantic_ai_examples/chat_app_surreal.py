"""Simple chat app example build with FastAPI using SurrealDB embedded.

Install SurrealDB from the optional dependency included in pydantic-ai-examples:

    uv sync --package pydantic-ai-examples --extra surrealdb

Set up your OpenAI API key:

    export OPENAI_API_KEY=your-api-key

Or, store it in a .env file, and add `--env-file .env` to your `uv run` commands.

Run with:

    uv run -m pydantic_ai_examples.chat_app_surreal
"""

from __future__ import annotations as _annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

import fastapi
import logfire
from fastapi import Depends, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from surrealdb import AsyncEmbeddedSurrealConnection, AsyncSurreal
from typing_extensions import TypedDict

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UnexpectedModelBehavior,
    UserPromptPart,
)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')
logfire.instrument_pydantic_ai()

agent = Agent('openai:gpt-5')
THIS_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(_app: fastapi.FastAPI):
    async with Database.connect() as db:
        yield {'db': db}


app = fastapi.FastAPI(lifespan=lifespan)
logfire.instrument_fastapi(app)


@app.get('/')
async def index() -> FileResponse:
    return FileResponse((THIS_DIR / 'chat_app.html'), media_type='text/html')


@app.get('/chat_app.ts')
async def main_ts() -> FileResponse:
    """Get the raw typescript code, it's compiled in the browser, forgive me."""
    return FileResponse((THIS_DIR / 'chat_app.ts'), media_type='text/plain')


async def get_db(request: Request) -> Database:
    return request.state.db


@app.get('/chat/')
async def get_chat(database: Database = Depends(get_db)) -> Response:
    msgs = await database.get_messages()
    return Response(
        b'\n'.join(json.dumps(to_chat_message(m)).encode('utf-8') for m in msgs),
        media_type='text/plain',
    )


class ChatMessage(TypedDict):
    """Format of messages sent to the browser."""

    role: Literal['user', 'model']
    timestamp: str
    content: str


def to_chat_message(m: ModelMessage) -> ChatMessage:
    first_part = m.parts[0]
    if isinstance(m, ModelRequest):
        if isinstance(first_part, UserPromptPart):
            assert isinstance(first_part.content, str)
            return {
                'role': 'user',
                'timestamp': first_part.timestamp.isoformat(),
                'content': first_part.content,
            }
    elif isinstance(m, ModelResponse):
        if isinstance(first_part, TextPart):
            return {
                'role': 'model',
                'timestamp': m.timestamp.isoformat(),
                'content': first_part.content,
            }
    raise UnexpectedModelBehavior(f'Unexpected message type for chat app: {m}')


@app.post('/chat/')
async def post_chat(
    prompt: Annotated[str, fastapi.Form()], database: Database = Depends(get_db)
) -> StreamingResponse:
    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # stream the user prompt so that can be displayed straight away
        yield (
            json.dumps(
                {
                    'role': 'user',
                    'timestamp': datetime.now(tz=timezone.utc).isoformat(),
                    'content': prompt,
                }
            ).encode('utf-8')
            + b'\n'
        )
        # get the chat history so far to pass as context to the agent
        messages = await database.get_messages()
        # run the agent with the user prompt and the chat history
        async with agent.run_stream(prompt, message_history=messages) as result:
            async for text in result.stream_output(debounce_by=0.01):
                # text here is a `str` and the frontend wants
                # JSON encoded ModelResponse, so we create one
                m = ModelResponse(parts=[TextPart(text)], timestamp=result.timestamp())
                yield json.dumps(to_chat_message(m)).encode('utf-8') + b'\n'

        # add new messages (e.g. the user prompt and the agent response in this case) to the database
        await database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type='text/plain')


@dataclass
class Database:
    """Database to store chat messages in SurrealDB embedded.

    Uses file-based persistence to store messages across sessions.
    """

    db: AsyncEmbeddedSurrealConnection
    namespace: str = 'chat_app'
    database: str = 'messages'

    @classmethod
    @asynccontextmanager
    async def connect(
        cls, db_path: Path = THIS_DIR / '.chat_app_messages_surrealdb'
    ) -> AsyncIterator[Database]:
        """Connect to SurrealDB embedded database.

        Uses file-based persistence so messages are saved across sessions.
        """
        with logfire.span('connect to DB'):
            db_url = f'file://{db_path}'
            # Use async context manager to properly manage the connection
            # The connection stays open for the entire lifespan of the FastAPI app
            async with AsyncSurreal(db_url) as db:
                if not isinstance(db, AsyncEmbeddedSurrealConnection):
                    raise ValueError(
                        f'Expected AsyncEmbeddedSurrealConnection, got {type(db)}'
                    )
                slf = cls(db)
                # Set namespace and database
                await slf.db.use(slf.namespace, slf.database)
                # Create table schema if it doesn't exist
                await slf._initialize_schema()
                # Yield the database instance - connection stays open until lifespan ends
                yield slf
                # Connection will be closed automatically when the async with block exits

    async def _initialize_schema(self) -> None:
        """Initialize the messages table schema."""
        # Define table if it doesn't exist
        # SurrealDB will create the table automatically on first insert,
        # but we can define it explicitly for better control
        await self.db.query('DEFINE TABLE message SCHEMALESS;')

    async def add_messages(self, messages: bytes) -> None:
        """Add new messages to the database.

        Messages are stored as JSON in the message_list field.
        """
        # Decode the bytes to get the JSON string
        messages_json = messages.decode('utf-8')
        # Validate it's valid JSON (will raise if invalid)
        json.loads(messages_json)

        # Create a record with the message list
        # Using a timestamp-based ID and created_at field for proper ordering
        now = datetime.now(timezone.utc)
        await self.db.create(
            'message',
            {
                'message_list': messages_json,
                'created_at': now.isoformat(),
            },
        )

    async def get_messages(self) -> list[ModelMessage]:
        """Retrieve all messages from the database, ordered by creation time."""
        # Query all messages ordered by created_at timestamp
        result = await self.db.query(
            'SELECT message_list, created_at FROM message ORDER BY created_at ASC;'
        )

        messages: list[ModelMessage] = []
        if isinstance(result, list):
            for record in result:
                if isinstance(record, dict) and 'message_list' in record:
                    # Parse the JSON string and extend the messages list
                    messages.extend(
                        ModelMessagesTypeAdapter.validate_json(
                            str(record['message_list'])
                        )
                    )
        else:
            raise ValueError(f'Expected list, got {type(result)}')

        return messages


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'pydantic_ai_examples.chat_app_surreal:app',
        reload=True,
        reload_dirs=[str(THIS_DIR)],
    )
