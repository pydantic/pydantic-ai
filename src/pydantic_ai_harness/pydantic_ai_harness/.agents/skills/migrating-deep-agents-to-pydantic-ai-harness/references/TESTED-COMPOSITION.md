# Tested Composition

Use this target smoke test only when a local coding-agent composition is plausible. `Coder(workspace)` is shorter when its documented defaults match the source.

The repository's skill-example test executes this example and forbids execution or lint skips, so imports and public constructor signatures cannot drift unnoticed. It does not prove behavioral parity with a source application.

```python
import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness.compaction import SlidingWindowCompaction
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.repo_context import RepoContext
from pydantic_ai_harness.skills import Skills
from pydantic_ai_harness.subagents import SubAgent, SubAgents


async def stream(_messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
    yield 'Workspace ready.'

worker = Agent(TestModel(call_tools=[]), name='researcher', description='Research a bounded question.')

with TemporaryDirectory() as workspace:
    root = Path(workspace)
    skill = root / 'skills' / 'inspect-workspace'
    skill.mkdir(parents=True)
    (skill / 'SKILL.md').write_text(
        '---\nname: inspect-workspace\ndescription: Inspect a workspace.\n---\nUse read-only tools.',
        encoding='utf-8',
    )
    migrated = Agent(
        FunctionModel(stream_function=stream),
        output_type=str,
        instructions='Work only inside the configured workspace.',
        capabilities=[
            Planning(),
            FileSystem(root, read_only=True),
            RepoContext(workspace_dir=root),
            Skills(root / 'skills'),
            SubAgents(agents=[SubAgent(worker, max_calls=1)], agent_folders=None),
            SlidingWindowCompaction(max_messages=20, keep_messages=10),
        ],
    )
    result = asyncio.run(migrated.run('Describe the available workspace tools.'))
    assert isinstance(result.output, str)
    assert result.all_messages()
```
