---
title: Recover a partially applied file migration
description: Record tool effects, reconcile ambiguous writes, and resume a migration without repeating completed changes.
---

# Recover a partially applied file migration

A process can fail after replacing a file but before recording the tool result. Persist each step, annotate the intended side effect, make the migration idempotent, and inspect unresolved effects before continuing.

```bash
pip/uv-add "pydantic-ai-slim[anthropic]" "pydantic-ai-harness==0.31.0"
export ANTHROPIC_API_KEY=your-api-key
```

```python {call_name="build_agent" dunder_name="not_main" noqa="I001"}
import json
import os
import stat
import tempfile
from pathlib import Path
from uuid import uuid4

from pydantic_ai_harness.step_persistence import (
    SqliteStepStore,
    StepPersistence,
    StepStore,
    annotate_tool_effect,
    continue_run,
)
from pydantic_ai import Agent, AgentRunResult, RunContext
from pydantic_ai.models import Model

DEFAULT_MODEL = os.environ.get('PYDANTIC_AI_MODEL', 'anthropic:claude-fable-5')


def build_agent(
    model: Model | str = DEFAULT_MODEL,
    *,
    workspace: Path | None = None,
    store: StepStore | None = None,
) -> Agent[object, str]:
    """Build an agent that migrates JSON configuration files idempotently."""
    root = (workspace or Path.cwd()).resolve()
    step_store = store or SqliteStepStore(database=root / '.migration-steps.db')
    agent: Agent[object, str] = Agent(
        model,
        name='config_migrator',
        instructions=(
            'Migrate every requested configuration file to schema version 2. '
            'Call migrate_file once per path and report files already at version 2 as reconciled.'
        ),
        capabilities=[StepPersistence(store=step_store)],
    )

    @agent.tool
    async def migrate_file(ctx: RunContext[object], path: str) -> str:
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            return f'Refused path outside workspace: {path}'

        data = json.loads(target.read_text())
        if data.get('schema_version') == 2:
            return f'{path} is already at schema version 2; no write performed.'

        await annotate_tool_effect(
            step_store,
            ctx,
            idempotency_key=f'config-v2:{target.relative_to(root)}',
            effect_summary=f'Atomically replace {path} with schema version 2.',
        )
        data['schema_version'] = 2
        descriptor, temporary_name = tempfile.mkstemp(prefix=f'.{target.name}.', dir=target.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, 'w') as file:
                file.write(json.dumps(data, indent=2) + '\n')
            temporary.chmod(stat.S_IMODE(target.stat().st_mode))
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return f'Migrated {path} to schema version 2.'

    return agent


async def resume_migration(
    agent: Agent[object, str],
    store: StepStore,
    *,
    failed_run_id: str,
) -> AgentRunResult[str]:
    """Inspect a failed run and resume from its latest settled checkpoint."""
    unresolved = await store.list_unresolved_tool_effects(run_id=failed_run_id)
    unsafe = [effect for effect in unresolved if effect.idempotency_key is None]
    if unsafe:
        details = ', '.join(f'{effect.tool_name}:{effect.effect_summary}' for effect in unsafe)
        raise RuntimeError(f'Reconcile non-idempotent side effects before resuming: {details}')

    events = await store.list_events(run_id=failed_run_id)
    failed = ', '.join(
        event.tool_name or 'unknown tool' for event in events if event.kind == 'tool_call_failed'
    ) or 'none recorded'
    history = await continue_run(store, run_id=failed_run_id, include_interrupted=True)
    return await agent.run(
        f'Resume the migration. Re-read each target before retrying; failed tools: {failed}.',
        message_history=history,
        run_id=f'{failed_run_id}-recovery-{uuid4().hex[:12]}',
    )
```

`StepPersistence` records checkpoints; it does not make arbitrary effects transactional. Refuse unresolved effects without an idempotency key, and review each keyed operation before deciding that retry is safe. Here the tool can reconcile an ambiguous write because it re-reads the file and treats schema version 2 as success. Including the interrupted frontier lets the resumed run close or retry the in-flight call, while the exclusive temporary file and atomic replacement preserve the original file mode.

## Related

See [Harness step persistence](https://pydantic.dev/docs/ai/harness/step-persistence/) for recovery states and continuation APIs.
