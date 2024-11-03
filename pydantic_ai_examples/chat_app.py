from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import fastapi
from fastapi.responses import HTMLResponse
from pydantic import Field, TypeAdapter

from pydantic_ai import Agent
from pydantic_ai.messages import Message, MessagesTypeAdapter

agent = Agent('openai:gpt-4o', deps=None)

app = fastapi.FastAPI()


@app.get('/')
async def index() -> HTMLResponse:
    return HTMLResponse((THIS_DIR / 'chat_app.html').read_bytes())


@app.get('/chat/')
async def get_chat() -> fastapi.Response:
    messages = list(database.get_messages())
    messages = MessagesTypeAdapter.dump_json(messages)
    return fastapi.Response(content=messages, media_type='application/json')


@app.post('/chat/')
async def post_chat(prompt: Annotated[str, fastapi.Form()]) -> fastapi.Response:
    messages = list(database.get_messages())
    response = await agent.run(prompt, message_history=messages)
    response_messages: list[Message] = []
    for message in response.message_history:
        if message.role != 'system':
            database.add_message(message)
            response_messages.append(message)
    messages = MessagesTypeAdapter.dump_json(response_messages)
    return fastapi.Response(content=messages, media_type='application/json')


THIS_DIR = Path(__file__).parent
MessageTypeAdapter: TypeAdapter[Message] = TypeAdapter(Annotated[Message, Field(discriminator='role')])


@dataclass
class Database:
    """Very rudimentary database to store chat messages in a JSON lines file."""

    file: Path = THIS_DIR / '.chat_app_messages.json'

    def add_message(self, message: Message):
        with self.file.open('ab') as f:
            f.write(MessageTypeAdapter.dump_json(message) + b'\n')

    def get_messages(self) -> Iterator[Message]:
        if self.file.exists():
            with self.file.open('rb') as f:
                for line in f:
                    if line:
                        yield MessageTypeAdapter.validate_json(line)


database = Database()


if __name__ == '__main__':
    import uvicorn

    uvicorn.run('pydantic_ai_examples.chat_app:app', reload=True, reload_dirs=[str(THIS_DIR)])
