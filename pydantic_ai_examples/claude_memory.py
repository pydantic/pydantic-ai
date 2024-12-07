r"""MCP server implementation to give Claude desktop memory.

Copy the appropriate {PROVIDER}_API_KEY into an .env file, for example:

    » echo OPENAI_API_KEY=$OPENAI_API_KEY > .env

Install the server with:

    » uvx fastmcp install pydantic_ai_examples/claude_memory.py -f .env --with-editable .

Open Claude desktop and look for the 🔨 in the bottom right.

    » open -a Claude

Review or edit your claude_desktop_config.json.

    » vim  ~/Library/Application\ Support/Claude/claude_desktop_config.json

Delete the memory file to start fresh.

    » rm ~/.fastmcp/memory.json
"""

import os
from pathlib import Path
from typing import Annotated, TypedDict

from fastmcp import FastMCP
from pydantic import Field
from pydantic_core import from_json, to_json

from pydantic_ai import Agent

server = FastMCP('Memory server', dependencies=['pydantic_ai_slim[openai]'])

MEMORY_PATH = Path(os.getenv('MEMORY_PATH') or '~/.fastmcp/memory.json').expanduser()

ConciseString = Annotated[str, Field(description='concise')]


# we need an actual schema but want to let the AI namespace
class MemoryKV(TypedDict):
    keys: list[Annotated[ConciseString, Field(description='one word, lowercase')]]
    values: list[ConciseString]


Biographer = Agent[None, MemoryKV](
    'openai:gpt-4o',
    system_prompt='categorize memories in bins, consolidate items to save space.',
    result_type=MemoryKV,
)


@server.tool()  # Claude will call this
async def save_memories(memories: list[ConciseString]) -> str:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = await Biographer.run(f'Memories: {memories}', deps=None)
    mapping = dict(zip(result.data['keys'], result.data['values']))
    MEMORY_PATH.write_bytes(to_json(mapping))
    return f'Saved {len(memories)} memories to bins: {" | ".join(mapping.keys())}'


def _display_memory() -> dict[str, list[str]]:
    if MEMORY_PATH.exists():
        return from_json(MEMORY_PATH.read_bytes())
    return {}


@server.tool()  # Claude will call this
def display_memory() -> dict[str, list[str]]:
    return _display_memory()


@server.resource('memory://user')  # allow user to copy/paste this
def memory_of_user() -> dict[str, list[str]]:
    return _display_memory()
