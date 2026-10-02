"""The helper that runs a docs page's blocks against a sandbox provider's live service."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from pydantic_ai.workspaces import WorkspaceRef

from ._docs_examples import documented_cleanup, python_blocks, run_block
from ._temporal import skip_temporal_sandbox_on_314

_PAGE = """
```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.coder import Coder

agent = Agent('anthropic:claude-opus-5-5', capabilities=[LocalWorkspace({root!r}), Coder()])
result = agent.run_sync('Look around.')
followup = agent.run_sync('And again.', message_history=result.all_messages())
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

Agent('openai:gpt-6', capabilities=[LocalWorkspace({root!r}), Shell()]).run_sync('Run something.')
Agent('openai:gpt-6', capabilities=[LocalWorkspace({root!r}), FileSystem()]).run_sync('Write something.')
```

```python
import ast
import re
from pathlib import Path


async def cleanup(ref) -> None:
    Path(ref.id, 'cleaned').touch()
```
"""


def test_blocks_run_their_agents_through_the_tools_and_hand_every_workspace_to_cleanup(tmp_path: Path) -> None:
    page = tmp_path / 'page.md'
    page.write_text(_PAGE.format(root=str(tmp_path)))
    blocks = python_blocks(str(page))
    coder, split, _ = blocks
    cleanup = documented_cleanup(blocks, 'cleanup')

    _, runs = run_block(coder, cleanup=cleanup)
    assert [run.used_sandbox for run in runs] == [True, True]
    assert set(runs[0].outputs) == {'shell', 'write_file', 'read_file'}
    assert runs[0].ref == WorkspaceRef(provider='local', id=str(tmp_path))
    assert (tmp_path / 'cleaned').exists()

    _, runs = run_block(split, cleanup=cleanup)
    assert [(run.used_sandbox, set(run.outputs)) for run in runs] == [
        (True, {'run_command'}),
        (True, {'write_file', 'read_file'}),
    ]


def test_sprites_durable_example_is_module_level_and_runnable() -> None:
    root = Path(__file__).parents[2]
    for path in (
        root / 'docs/harness/sprites-sandbox.md',
        root / 'src/pydantic_ai_harness/pydantic_ai_harness/sprites_sandbox/README.md',
    ):
        section = path.read_text().split('## Durable execution\n', 1)[1].split('\n## ', 1)[0]
        code = re.search(r'```python[^\n]*\n(.*?)\n```', section, re.DOTALL)
        assert code is not None
        source = code.group(1)
        assert '...' not in source
        tree = ast.parse(source)
        assert any(
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'agent' for t in node.targets)
            for node in tree.body
        )
        assert 'TemporalDurability()' in source
        assert 'Coder()' in source
        assert 'PydanticAIPlugin()' in source
        assert 'workflows=[' in source


_TEMPORAL_PAGE = """
```python
import asyncio
import uuid

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.durable_exec.temporal import PydanticAIPlugin, PydanticAIWorkflow, TemporalDurability
from pydantic_ai_harness.coder import Coder
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

agent = Agent(
    'anthropic:claude-opus-5-5',
    name='docs_example_coder',
    capabilities=[LocalWorkspace({root!r}), Coder(), TemporalDurability()],
)


@workflow.defn
class CoderWorkflow(PydanticAIWorkflow):
    __pydantic_ai_agents__ = [agent]

    @workflow.run
    async def run(self, prompt: str) -> str:
        return (await agent.run(prompt)).output


async def main() -> None:
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin()])
    async with Worker(client, task_queue='coder', workflows=[CoderWorkflow]):
        print(
            await client.execute_workflow(
                CoderWorkflow.run, 'Look around.', id=f'coder-{{uuid.uuid4()}}', task_queue='coder'
            )
        )


if __name__ == '__main__':
    asyncio.run(main())
```
"""


@pytest.mark.temporal
@pytest.mark.xdist_group(name='harness-temporal')
@skip_temporal_sandbox_on_314
def test_temporal_block_runs_its_workflow_against_a_local_dev_server(tmp_path: Path) -> None:
    pytest.importorskip('temporalio')
    page = tmp_path / 'page.md'
    page.write_text(_TEMPORAL_PAGE.format(root=str(tmp_path)))
    (example,) = python_blocks(str(page))

    async def cleanup(ref: WorkspaceRef) -> None:
        Path(ref.id, 'cleaned').touch()

    _, runs = run_block(example, cleanup=cleanup)
    assert [(run.used_sandbox, set(run.outputs)) for run in runs] == [(True, {'shell', 'write_file', 'read_file'})]
    assert (tmp_path / 'cleaned').exists()
