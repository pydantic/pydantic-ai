"""Cache LLM outputs by exact input, so an unchanged fleet costs nothing to re-mine and gives the same drafts.

That is what makes `--watch` cheap, and what lets it write only when something actually changed: the same prompts
produce byte-identical proposals instead of a fresh paraphrase every cycle.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pydantic_ai import Agent

PATH = Path(__file__).parent / '.cache' / 'llm.json'


async def run_cached[OutputT: BaseModel](
    agent: Agent[Any, OutputT], prompt: str, *, output_type: type[OutputT], deps: Any = None
) -> OutputT:
    key = hashlib.sha256(
        json.dumps(
            [agent.name, getattr(agent.model, 'model_name', agent.model), agent._instructions, prompt],  # pyright: ignore[reportPrivateUsage]
            default=str,
        ).encode()
    ).hexdigest()
    cache: dict[str, str] = json.loads(PATH.read_text()) if PATH.exists() else {}
    if key in cache:
        return output_type.model_validate_json(cache[key])
    result = await agent.run(prompt, deps=deps)
    # Re-read before saving: concurrent runs (drafts are gathered) each add their own key.
    cache = json.loads(PATH.read_text()) if PATH.exists() else {}
    cache[key] = result.output.model_dump_json()
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(cache))
    return result.output
