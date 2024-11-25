"""Simple chat app example build with FastAPI.

Run with:

    uv run -m pydantic_ai_examples.chat_app
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import fastapi
import logfire
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import Field, TypeAdapter

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

app = fastapi.FastAPI()
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
    msgs = database.get_messages()
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
        messages = list(database.get_messages())
        # run the agent with the user prompt and the chat history
        async with agent.run_stream(prompt, message_history=messages) as result:
            async for text in result.stream(debounce_by=0.01):
                # text here is a `str` and the frontend wants
                # JSON encoded ModelTextResponse, so we create one
                m = ModelTextResponse(content=text, timestamp=result.timestamp())
                yield MessageTypeAdapter.dump_json(m) + b'\n'

        # add new messages (e.g. the user prompt and the agent response in this case) to the database
        database.add_messages(result.new_messages_json())

    return StreamingResponse(stream_messages(), media_type='text/plain')


THIS_DIR = Path(__file__).parent
MessageTypeAdapter: TypeAdapter[Message] = TypeAdapter(
    Annotated[Message, Field(discriminator='role')]
)


@dataclass
class Database:
    """Very rudimentary database to store chat messages in a JSON lines file."""

    file: Path = THIS_DIR / '.chat_app_messages.jsonl'

    def add_messages(self, messages: bytes):
        with self.file.open('ab') as f:
            f.write(messages + b'\n')

    def get_messages(self) -> Iterator[Message]:
        if self.file.exists():
            with self.file.open('rb') as f:
                for line in f:
                    if line:
                        yield from MessagesTypeAdapter.validate_json(line)


database = Database()


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'pydantic_ai_examples.chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)]
    )
