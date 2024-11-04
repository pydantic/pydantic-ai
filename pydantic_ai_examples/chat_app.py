"""Simple chat app example build with FastAPI.

Run with:

    uv run -m pydantic_ai_examples.chat_app
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import fastapi
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import Field, TypeAdapter

from pydantic_ai import Agent
from pydantic_ai.messages import Message, MessagesTypeAdapter, UserPrompt

agent = Agent('openai:gpt-4o', deps=None)

app = fastapi.FastAPI()


@app.get('/')
async def index() -> HTMLResponse:
    return HTMLResponse((THIS_DIR / 'chat_app.html').read_bytes())


@app.get('/chat/')
async def get_chat() -> Response:
    msgs = database.get_messages()
    return Response(b'\n'.join(MessageTypeAdapter.dump_json(m) for m in msgs), media_type='text/plain')


@app.post('/chat/')
async def post_chat(prompt: Annotated[str, fastapi.Form()]) -> StreamingResponse:
    async def stream_messages():
        """Streams new line delimited JSON `Message`s to the client."""
        # stream the user prompt so that can be displayed straight away
        yield MessageTypeAdapter.dump_json(UserPrompt(content=prompt)) + b'\n'
        # get the chat history so far to pass as context to the agent
        messages = list(database.get_messages())
        response = await agent.run(prompt, message_history=messages)
        # add new messages (e.g. the user prompt and the agent response in this case) to the database
        database.add_messages(response.new_messages_json())
        # stream the last message which will be the agent response, we can't just yield `new_messages_json()`
        # since we already stream the user prompt
        yield MessageTypeAdapter.dump_json(response.all_messages()[-1]) + b'\n'

    return StreamingResponse(stream_messages(), media_type='text/plain')


THIS_DIR = Path(__file__).parent
MessageTypeAdapter: TypeAdapter[Message] = TypeAdapter(Annotated[Message, Field(discriminator='role')])


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

    uvicorn.run('pydantic_ai_examples.chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)])
