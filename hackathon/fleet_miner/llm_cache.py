"""Cache LLM outputs by exact input, so an unchanged fleet costs nothing to re-mine and gives the same drafts.

That is what makes `--watch` cheap, and what lets it write only when something actually changed: the same prompts
produce byte-identical proposals instead of a fresh paraphrase every cycle.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pydantic_ai import Agent, AgentRunResult

CACHE_DIR = Path(os.environ.get('FLEET_MINER_CACHE_DIR') or Path(__file__).parent / '.cache')
PATH = CACHE_DIR / 'llm.json'


@dataclass
class Usage:
    """LLM spend for one mining run, so cost per run can be reported."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_hits: int = 0

    def add(self, result: AgentRunResult[Any]) -> None:
        usage = result.usage() if callable(result.usage) else result.usage
        self.requests += usage.requests
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens

    def reset(self) -> None:
        self.requests = self.input_tokens = self.output_tokens = self.cached_hits = 0

    def __str__(self) -> str:
        return (
            f'{self.requests} LLM requests, {self.input_tokens:,} in / {self.output_tokens:,} out tokens, '
            f'{self.cached_hits} cache hits'
        )


USAGE = Usage()


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
        USAGE.cached_hits += 1
        return output_type.model_validate_json(cache[key])
    result = await agent.run(prompt, deps=deps)
    USAGE.add(result)
    # Re-read before saving: concurrent runs (drafts are gathered) each add their own key.
    cache = json.loads(PATH.read_text()) if PATH.exists() else {}
    cache[key] = result.output.model_dump_json()
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(cache))
    return result.output
