import asyncio
import os
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

import pytest
from inline_snapshot import snapshot
from pydantic_ai import (
    Agent,
    ModelRequest,
    ModelResponse,
    RequestUsage,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_harness import Coder
from pydantic_ai_harness.planning import _types as planning_types
from pydantic_ai_harness.repo_context import AgentContextInventory, AssetRoot

if TYPE_CHECKING:

    def IsDatetime(*args: Any, **kwargs: Any) -> datetime: ...
    def IsInstance(expected_type: type[RequestUsage], **kwargs: Any) -> RequestUsage: ...
    def IsStr(*args: Any, **kwargs: Any) -> str: ...
else:
    from dirty_equals import IsDatetime, IsInstance, IsStr

pytestmark = pytest.mark.anyio

_SNAPSHOT_OUTPUT_ENVIRONMENT_VARIABLES = (
    'BUILD_ID',
    'BUILD_NUMBER',
    'CI',
    'CLICOLOR_FORCE',
    'FORCE_COLOR',
    'GITHUB_ACTIONS',
    'PY_COLORS',
)


@pytest.mark.vcr
async def test_coder_completes_task(
    tmp_path: Path, allow_model_requests: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('ANTHROPIC_API_KEY', os.environ.get('ANTHROPIC_API_KEY', 'replay-key'))
    # Plan step ids are `uuid4().hex[:8]`, and the model quotes them back in later `write_plan` and
    # `update_task_status*` calls. A cassette therefore bakes in the ids of the run that recorded it,
    # and on replay a fresh `uuid4()` makes every one of those calls miss its step. Counting ids keep
    # the recorded trajectory replayable. `hex[:8]` is the top 32 bits, so the counter goes there.
    ids = count()
    monkeypatch.setattr(planning_types, 'uuid4', lambda: UUID(int=next(ids) << 96))
    for environment_variable in ('CLICOLOR_FORCE', 'FORCE_COLOR', 'PY_COLORS'):
        monkeypatch.setenv(environment_variable, '1')
    # The agent's `run_command` snapshots subprocess output. These environment variables alter that output.
    for environment_variable in _SNAPSHOT_OUTPUT_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(environment_variable, raising=False)
    assert all(
        environment_variable not in os.environ for environment_variable in _SNAPSHOT_OUTPUT_ENVIRONMENT_VARIABLES
    )
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    (workspace / 'AGENTS.md').write_text(
        'Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.\n'
    )
    (workspace / 'README.md').write_text(
        '# Incident desk\n\nIncidents enter through `cli.py`, public report composition lives in `service.py`, '
        'visibility rules live in `policy.py`, and each output format has its own renderer module. Internal incidents '
        'must not appear in exports unless a caller explicitly opts in.\n'
    )
    (workspace / 'incidents.py').write_text(
        'from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Incident:\n'
        '    title: str\n    severity: str\n    internal: bool = False\n'
    )
    (workspace / 'policy.py').write_text(
        'from incidents import Incident\n\n\ndef visible_incidents(\n'
        '    incidents: list[Incident], *, include_internal: bool\n) -> list[Incident]:\n'
        '    if include_internal:\n        return incidents\n'
        '    return [incident for incident in incidents if not incident.internal]\n'
    )
    (workspace / 'text_renderer.py').write_text(
        'from incidents import Incident\n\n\ndef render_text(incidents: list[Incident]) -> str:\n'
        "    return '\\n'.join(f'[{item.severity.upper()}] {item.title}' for item in incidents)\n"
    )
    (workspace / 'service.py').write_text(
        'from incidents import Incident\nfrom policy import visible_incidents\nfrom text_renderer import render_text\n\n\n'
        'def render_report(incidents: list[Incident], *, include_internal: bool = False) -> str:\n'
        '    visible = visible_incidents(incidents, include_internal=include_internal)\n'
        '    return render_text(visible)\n'
    )
    (workspace / 'cli.py').write_text(
        'from incidents import Incident\nfrom service import render_report\n\n\ndef run(\n'
        '    incidents: list[Incident], *, include_internal: bool = False\n) -> str:\n'
        '    return render_report(incidents, include_internal=include_internal)\n'
    )
    (workspace / 'dashboard.py').write_text(
        'from incidents import Incident\nfrom service import render_report\n\n\ndef incident_card(items: list[Incident]) -> str:\n'
        '    return render_report(items)\n'
    )
    (workspace / 'digest.py').write_text(
        'from incidents import Incident\nfrom service import render_report\n\n\ndef internal_digest(items: list[Incident]) -> str:\n'
        '    return render_report(items, include_internal=True)\n'
    )
    (workspace / 'test_reports.py').write_text(
        'from cli import run\nfrom incidents import Incident\n\n\ndef test_text_report_hides_internal_by_default() -> None:\n'
        "    incidents = [Incident('API down', 'critical'), Incident('Secret host', 'warning', internal=True)]\n"
        "    assert run(incidents) == '[CRITICAL] API down'\n\n\ndef test_text_report_can_include_internal() -> None:\n"
        "    incidents = [Incident('Secret host', 'warning', internal=True)]\n"
        "    assert run(incidents, include_internal=True) == '[WARNING] Secret host'\n"
    )

    agent = Agent('anthropic:claude-fable-5', capabilities=[Coder(workspace)])
    prompt = (
        'Add machine-readable incident reports end to end. Treat mapping the public service, its consumers, '
        'the privacy policy, and renderer boundaries as a separate read-only compatibility investigation, then '
        'use those findings during implementation so existing callers stay intact. Callers of both the public '
        'service and CLI adapter should be able to select `format_name="json"`; JSON contains an `incidents` array '
        'of objects with `title` and `severity`, in input order. Preserve text as the default and the existing '
        'internal-incident privacy behavior for both formats. Follow the repository architecture, cover the service '
        'and CLI paths including explicit internal opt-in, and run the full test suite.'
    )
    result = await agent.run(prompt)

    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Add machine-readable incident reports end to end. Treat mapping the public service, its consumers, the privacy policy, and renderer boundaries as a separate read-only compatibility investigation, then use those findings during implementation so existing callers stay intact. Callers of both the public service and CLI adapter should be able to select `format_name="json"`; JSON contains an `incidents` array of objects with `title` and `severity`, in input order. Preserve text as the default and the existing internal-incident privacy behavior for both formats. Follow the repository architecture, cover the service and CLI paths including explicit internal opt-in, and run the full test suite.',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.
</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.

You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature='CAISuAIKjwEIERgCKkBTJSNR/w2HWksMrCWGirI74UnO81II1GagS8QOFYvUTfJJZR7NeuY9lDYFeFnwBy/EVisFLGYZcAdNiaeVgfbHMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiqAG9/ObUBhIMk/tCATYKFBawHZsaGgyaVvRnaWllKxO6K88iMDJ2UDv1Bsjtmys6sZ0KU6PDdwBijUqTBD96XaqFHy8mRYF/QEAwX+OpNxpBHFeX+ypWMN/s8MJQLxgBQI/RuSy6CowLcPMGwmDk/0WHth3E+DaLBelQ0aJ3HZHPxmS/IBKTcf6K9pVe/5yt81ZXLCYs+Vhswy1+Nae54bS93OzwRXpjW30P6DcYAQ==',
                        provider_name='anthropic',
                    ),
                    ToolCallPart(
                        tool_name='inventory_agent_context',
                        args='',
                        tool_call_id='toolu_01Qbov3cx389nmfxqJmST8f6',
                    ),
                    ToolCallPart(
                        tool_name='list_directory',
                        args='{"path": "."}',
                        tool_call_id='toolu_011FXeZL5wwv2CtE7PFBzDmn',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011CegzPWFwLFyUnYJK3g1ES',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='inventory_agent_context',
                        content=AgentContextInventory(
                            roots=[
                                AssetRoot(root='.claude', exists=False),
                                AssetRoot(root='.agents', exists=False),
                                AssetRoot(
                                    root='.codex',
                                    exists=False,
                                    notes='Codex uses TOML config; assets are derived from the .claude/.agents setup.',
                                ),
                                AssetRoot(
                                    root='.grok',
                                    exists=False,
                                    notes='Grok setup is derived from the .claude/.agents setup.',
                                ),
                            ]
                        ),
                        tool_call_id='toolu_01Qbov3cx389nmfxqJmST8f6',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='list_directory',
                        content="""\
AGENTS.md  (104 bytes)
README.md  (272 bytes)
cli.py  (222 bytes)
dashboard.py  (148 bytes)
digest.py  (173 bytes)
incidents.py  (136 bytes)
policy.py  (257 bytes)
service.py  (306 bytes)
test_reports.py  (463 bytes)
text_renderer.py  (170 bytes)\
""",
                        tool_call_id='toolu_011FXeZL5wwv2CtE7PFBzDmn',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.
</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.

You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='write_plan',
                        args='{"items": [{"content":"Run read-only compatibility investigation of service, consumers, privacy policy, renderer boundaries","status":"in_progress","active_form":"Investigating compatibility surface"},{"content":"Implement JSON renderer and format selection in service/CLI preserving defaults","status":"pending","active_form":"Implementing JSON format support"},{"content":"Add tests covering service and CLI paths incl. internal opt-in, run full suite","status":"pending","active_form":"Adding tests and running suite"}]}',
                        tool_call_id='toolu_01HcMQogbWszgbEPqQqWCuYc',
                    ),
                    ToolCallPart(
                        tool_name='delegate_task',
                        args='{"agent_name": "explorer", "task": "Read-only investigation of a small Python repo at the root directory. Files: AGENTS.md, README.md, cli.py, dashboard.py, digest.py, incidents.py, policy.py, service.py, test_reports.py, text_renderer.py. Do NOT modify anything.\\n\\nReport back, quoting relevant code:\\n1. The public service in service.py: its function signatures, parameters, default values, and exact behavior.\\n2. All consumers of the service (cli.py, dashboard.py, digest.py, any others found via grep) — how they call the service, which parameters they pass, and what they expect back.\\n3. The privacy policy in policy.py — how internal incidents are filtered/handled, defaults for including internal incidents.\\n4. Renderer boundaries: text_renderer.py interface (function names, input/output shapes), and how the service invokes it.\\n5. incidents.py data model (fields like title, severity, internal flag).\\n6. test_reports.py — existing tests and expectations.\\n7. Any architectural conventions implied (e.g., domain policy separate from presentation), and what changes would risk breaking existing callers if we add a format_name=\\"json\\" option (default \\"text\\") to the service and CLI adapter."}',
                        tool_call_id='toolu_01F2AkG2CAUZbNbDBcx1wkvZ',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011CegzPrmj6hUr64bKR1DF7',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='write_plan',
                        content="""\
Plan updated: 3 step(s).

1. [~] Run read-only compatibility investigation of service, consumers, privacy policy, renderer boundaries
2. [ ] Implement JSON renderer and format selection in service/CLI preserving defaults
3. [ ] Add tests covering service and CLI paths incl. internal opt-in, run full suite
(0/3 completed)\
""",
                        tool_call_id='toolu_01HcMQogbWszgbEPqQqWCuYc',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='delegate_task',
                        content="""\
Investigation complete (read-only). Note: `inventory_agent_context` reports no `.claude`/`.agents`/`.codex`/`.grok` directories exist — the only assistant instructions are in `AGENTS.md` ("Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.").

## 1. Public service — `service.py` (8 lines)

```python
from incidents import Incident
from policy import visible_incidents
from text_renderer import render_text


def render_report(incidents: list[Incident], *, include_internal: bool = False) -> str:
    visible = visible_incidents(incidents, include_internal=include_internal)
    return render_text(visible)
```

- **Signature:** `render_report(incidents: list[Incident], *, include_internal: bool = False) -> str`
- `include_internal` is **keyword-only** and defaults to **`False`** (internal incidents hidden by default).
- **Behavior:** filters via `policy.visible_incidents`, then delegates rendering to `text_renderer.render_text`. Returns a plain string. No format selection exists today — text is hardcoded.

## 2. Consumers of the service

Grep for `render_report` finds exactly three consumers:

**`cli.py:5-8`** — pass-through adapter, forwards the flag, expects `str`:
```python
def run(
    incidents: list[Incident], *, include_internal: bool = False
) -> str:
    return render_report(incidents, include_internal=include_internal)
```

**`dashboard.py:5-6`** — relies on the **default** (`include_internal=False`), expects `str`:
```python
def incident_card(items: list[Incident]) -> str:
    return render_report(items)
```

**`digest.py:5-6`** — explicitly opts into internal incidents, expects `str`:
```python
def internal_digest(items: list[Incident]) -> str:
    return render_report(items, include_internal=True)
```

## 3. Privacy policy — `policy.py`

```python
def visible_incidents(
    incidents: list[Incident], *, include_internal: bool
) -> list[Incident]:
    if include_internal:
        return incidents
    return [incident for incident in incidents if not incident.internal]
```

- Keyword-only `include_internal` with **no default here** — the policy layer forces callers (the service) to decide explicitly; the `False` default lives in `service.py`/`cli.py`.
- When `False`, any `Incident` with `internal=True` is dropped before rendering. README.md line 3 states the invariant: "Internal incidents must not appear in exports unless a caller explicitly opts in."

## 4. Renderer boundary — `text_renderer.py`

```python
def render_text(incidents: list[Incident]) -> str:
    return '\\n'.join(f'[{item.severity.upper()}] {item.title}' for item in incidents)
```

- **Input:** already-filtered `list[Incident]`; **Output:** newline-joined lines of `[SEVERITY] title` (severity uppercased). Empty list → `''`.
- The renderer knows nothing about the `internal` flag or policy — the service invokes it only with `visible` incidents (`service.py:7-8`). README confirms the convention: "each output format has its own renderer module."

## 5. Data model — `incidents.py`

```python
@dataclass(frozen=True)
class Incident:
    title: str
    severity: str
    internal: bool = False
```

Frozen (immutable) dataclass; `internal` defaults to `False` (public).

## 6. Tests — `test_reports.py`

Both tests go through the CLI adapter `run`, pinning both the default behavior and the exact text format:

```python
def test_text_report_hides_internal_by_default() -> None:
    incidents = [Incident('API down', 'critical'), Incident('Secret host', 'warning', internal=True)]
    assert run(incidents) == '[CRITICAL] API down'


def test_text_report_can_include_internal() -> None:
    incidents = [Incident('Secret host', 'warning', internal=True)]
    assert run(incidents, include_internal=True) == '[WARNING] Secret host'
```

Run with `pytest -q` per AGENTS.md.

## 7. Architectural conventions & risks of adding `format_name="json"`

**Conventions in evidence:**
- **Layering** (README.md + AGENTS.md): entry point (`cli.py`) → composition (`service.py`) → domain policy (`policy.py`) → presentation (`text_renderer.py`). Policy is deliberately separated from rendering; the renderer never sees internal incidents.
- **Keyword-only privacy flag** everywhere; the safe default (`False`) is set at the public boundary, while `policy.py` requires an explicit value.
- **Per-format renderer modules** ("each output format has its own renderer module") — a JSON format should get a new `json_renderer.py`, not be bolted into `render_text`.

**What would break callers if done wrong:**
1. **Changing defaults or making the parameter positional.** `format_name` must be keyword-only with default `"text"`; e.g. `def render_report(incidents, *, include_internal=False, format_name="text")`. `dashboard.incident_card` calls `render_report(items)` with no kwargs and both tests assert the exact text output (`'[CRITICAL] API down'`), so any change to the default format or text shape breaks them.
2. **Altering `include_internal` semantics or its `False` default** — `digest.py` depends on opt-in working; `dashboard.py` and `test_text_report_hides_internal_by_default` depend on the hidden-by-default behavior (a privacy regression, not just a test failure).
3. **Filtering after/inside the renderer.** JSON rendering must receive the same `visible = visible_incidents(...)` output; serializing raw incidents in a JSON path would leak `internal=True` items and violate the README invariant.
4. **Changing the return type** — all three consumers and both tests expect `str`; JSON output should be a serialized string (`json.dumps`), not a dict/list.
5. **CLI adapter parity** — `cli.run` mirrors the service signature; it should add the same keyword-only `format_name: str = "text"` and forward it, leaving existing call sites (`run(incidents)`, `run(incidents, include_internal=True)`) untouched.\
""",
                        tool_call_id='toolu_01F2AkG2CAUZbNbDBcx1wkvZ',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.
</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.

You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature='CAIS+gIKjwEIERgCKkAW6awWgQiZksI6NGcPNQXdYK4ydAeMKVA7EtcZ5hroBqjx6oUebs77T9NN+/zX8egEAPaWBiFz1jcC29YEYMeZMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiqAH5/ObUBhIMzDQzSLQcQ8IXeSdnGgx7qAGJB5Vx9y4xapMiMATUcCPEfaApJusVZLDYV8ho9BvkV76HPyeSwdjIgoGMwPXS53N5moCf5MsVkPNI/yqXAZ2V1HqIB6KLInp2yqAE3gFRbokJUxqM7q3dN793kIWP9krDCFFaPiSH60VrvmhknATdQ+gTNe8dT1obhK6osERiIt77sniXgib/LoF0qhgoCXn8UPd92Aqgg1g+IPhj5IQjUWKPTvWv944QRKnsF2YRAqweL6M2UikWfsT+1xgHJLXtOk1TC16eyFuFvchq3TjNA6lB1x8YAQ==',
                        provider_name='anthropic',
                    ),
                    ThinkingPart(
                        content='',
                        signature='CAISjwQKkAEIERgCKkAwt/9n+dmCsHbf97HP7BBKVRFg7lPrpPcc/wX9dfAGEguh5vTbUYwA9u6Dh2ampAwPz86hhiccRovZXSgrbNXOMg5jbGF1ZGUtZmFibGUtNTgBQgluYXJyYXRpb25aJDQ0YWU2NzZjLTk1OGYtNGQ2OC05MTA4LWVhZTlkZTdiMzY2YqgB+fzm1AYSDLre/l2ouTxHqFDxMhoMQSEa8Cq36ENwzNKBIjCHk8shXt+zbXeWSb/ou5ebiYKfYQSU0zb/yUyNI2FkheWKLtV9Rl2uBRibWc1v6a0qqwJAzUuq0cT0sQMS6LlV1OQkRg1c7gnzkUajYZKut//x2QPrkVw47rfROxgIg5oj46wp8GqSaDyAo3RAPbCroHGIAMY5GXUmyNWJllGaYl46NJqUhxOss9+jKsUBOwY7FouVUBksdlScErkuXmi1k1J1Dh1uaNwhZfzAnwUhqTPB3bBRlQN1ypdeRBXFnOQlM/K/A8Q9hXLC0FNqa8vF/G/h5F1Q48zis4A3cxQd905pWN89PZVklXWXYblaXc4EQunm7riB+WcPzGnsJ9Ky4kAFT/OJOff/KeDsV4o8vMu/oG/CT4AwmwhbPJwwk6zak4XrlZRXfrllurd/bCjJw/qwbh7fqHejRZjkUF3AzGbSKyvvenlbozwE+Wpi5Snf235niyzGiMD2pwuGYxgB',
                        provider_name='anthropic',
                    ),
                    ToolCallPart(
                        tool_name='update_task_statuses',
                        args='{"updates": [{"task_id":"task-1","status":"completed"},{"task_id":"task-2","status":"in_progress"}]}',
                        tool_call_id='toolu_01KBYwbgXH7uKhA8NMisBMDg',
                    ),
                    ToolCallPart(
                        tool_name='read_plan',
                        args='',
                        tool_call_id='toolu_01VuFi7L9SVt7VkxS57ekaa6',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011CegzTmaJ6twXKdXiohzxf',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='update_task_statuses',
                        content="""\
No changes applied. Errors:
- Step with id 'task-1' not found.
- Step with id 'task-2' not found.\
""",
                        tool_call_id='toolu_01KBYwbgXH7uKhA8NMisBMDg',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='read_plan',
                        content=IsStr(
                            regex="""\
Current\\ plan:\\
1\\.\\ \\[\\~\\]\\ \\[[0-9a-f]{8}\\]\\ Run\\ read\\-only\\ compatibility\\ investigation\\ of\\ service,\\ consumers,\\ privacy\\ policy,\\ renderer\\ boundaries\\
2\\.\\ \\[\\ \\]\\ \\[[0-9a-f]{8}\\]\\ Implement\\ JSON\\ renderer\\ and\\ format\\ selection\\ in\\ service/CLI\\ preserving\\ defaults\\
3\\.\\ \\[\\ \\]\\ \\[[0-9a-f]{8}\\]\\ Add\\ tests\\ covering\\ service\\ and\\ CLI\\ paths\\ incl\\.\\ internal\\ opt\\-in,\\ run\\ full\\ suite\\
\\
Summary:\\ 0\\ completed,\\ 1\\ in\\ progress,\\ 2\\ pending\
"""
                        ),
                        tool_call_id='toolu_01VuFi7L9SVt7VkxS57ekaa6',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.
</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.

You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='update_task_statuses',
                        args=IsStr(
                            regex='\\{"updates":\\ \\[\\{"task_id":"[0-9a-f]{8}","status":"completed"\\},\\{"task_id":"[0-9a-f]{8}","status":"in_progress"\\}\\]\\}'
                        ),
                        tool_call_id='toolu_012o5fAGmtk14brogBaWQpSe',
                    ),
                    ToolCallPart(
                        tool_name='read_file',
                        args='{"path": "service.py"}',
                        tool_call_id='toolu_01LRk9ronME7JPeyh19kfPR6',
                    ),
                    ToolCallPart(
                        tool_name='read_file',
                        args='{"path": "cli.py"}',
                        tool_call_id='toolu_018ZJ5899wSxJSzGoLfCfiuM',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011CegzUFWp7Z24YyqPXTjb2',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='update_task_statuses',
                        content=IsStr(
                            regex="""\
Updated\\ 2\\ step\\(s\\):\\
\\-\\ \\[[0-9a-f]{8}\\]\\ Run\\ read\\-only\\ compatibility\\ investigation\\ of\\ service,\\ consumers,\\ privacy\\ policy,\\ renderer\\ boundaries\\ \\->\\ completed\\
\\-\\ \\[[0-9a-f]{8}\\]\\ Implement\\ JSON\\ renderer\\ and\\ format\\ selection\\ in\\ service/CLI\\ preserving\\ defaults\\ \\->\\ in_progress\
"""
                        ),
                        tool_call_id='toolu_012o5fAGmtk14brogBaWQpSe',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='read_file',
                        content="""\
[service.py | 8 lines | hash:2d0256ca2541]
     1	from incidents import Incident
     2	from policy import visible_incidents
     3	from text_renderer import render_text
     4	
     5	
     6	def render_report(incidents: list[Incident], *, include_internal: bool = False) -> str:
     7	    visible = visible_incidents(incidents, include_internal=include_internal)
     8	    return render_text(visible)
""",
                        tool_call_id='toolu_01LRk9ronME7JPeyh19kfPR6',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='read_file',
                        content="""\
[cli.py | 8 lines | hash:9cb48c8b54cf]
     1	from incidents import Incident
     2	from service import render_report
     3	
     4	
     5	def run(
     6	    incidents: list[Incident], *, include_internal: bool = False
     7	) -> str:
     8	    return render_report(incidents, include_internal=include_internal)
""",
                        tool_call_id='toolu_018ZJ5899wSxJSzGoLfCfiuM',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.
</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.

You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='write_file',
                        args='{"path": "json_renderer.py", "content": "import json\\n\\nfrom incidents import Incident\\n\\n\\ndef render_json(incidents: list[Incident]) -> str:\\n    return json.dumps(\\n        {\\n            \'incidents\': [\\n                {\'title\': item.title, \'severity\': item.severity}\\n                for item in incidents\\n            ]\\n        }\\n    )\\n"}',
                        tool_call_id='toolu_01W9Uy9ovRRv3rVMB7Vw4EG9',
                    ),
                    ToolCallPart(
                        tool_name='write_file',
                        args='{"path": "service.py", "content": "from incidents import Incident\\nfrom json_renderer import render_json\\nfrom policy import visible_incidents\\nfrom text_renderer import render_text\\n\\n_RENDERERS = {\\n    \'text\': render_text,\\n    \'json\': render_json,\\n}\\n\\n\\ndef render_report(\\n    incidents: list[Incident],\\n    *,\\n    include_internal: bool = False,\\n    format_name: str = \'text\',\\n) -> str:\\n    try:\\n        renderer = _RENDERERS[format_name]\\n    except KeyError:\\n        raise ValueError(f\'unknown format_name: {format_name!r}\') from None\\n    visible = visible_incidents(incidents, include_internal=include_internal)\\n    return renderer(visible)\\n", "expected_hash": "2d0256ca2541"}',
                        tool_call_id='toolu_01MQ5ysQeQcEkXUHDSJRfCC6',
                    ),
                    ToolCallPart(
                        tool_name='write_file',
                        args='{"path": "cli.py", "content": "from incidents import Incident\\nfrom service import render_report\\n\\n\\ndef run(\\n    incidents: list[Incident],\\n    *,\\n    include_internal: bool = False,\\n    format_name: str = \'text\',\\n) -> str:\\n    return render_report(\\n        incidents, include_internal=include_internal, format_name=format_name\\n    )\\n", "expected_hash": "9cb48c8b54cf"}',
                        tool_call_id='toolu_015Dg7za9ncPhAacEouoaoB3',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011CegzUb3K6NKs3FT6o8iMR',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='write_file',
                        content='Wrote 290 chars (14 lines) to json_renderer.py. [hash:a48d755e3cd3]',
                        tool_call_id='toolu_01W9Uy9ovRRv3rVMB7Vw4EG9',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='write_file',
                        content='Wrote 604 chars (23 lines) to service.py. [hash:0e4696eab4f6]',
                        tool_call_id='toolu_01MQ5ysQeQcEkXUHDSJRfCC6',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='write_file',
                        content='Wrote 301 chars (13 lines) to cli.py. [hash:a3b143673288]',
                        tool_call_id='toolu_015Dg7za9ncPhAacEouoaoB3',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.
</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.

You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature='CAISvwIKjwEIERgCKkAGHztwC3UrnwOHIFA3D4FJDA8sAPohb1uIJ87EDtfVUowac8DT5M/1Vm7JFF/f5t1TBgH8JbzYnELdcUJt+4YaMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiqAGM/ebUBhIMwaMhNLqMlptl3QxWGgwoBgivZliXZGQS6jMiMN6FJ5cm6/Iqb6nS1Uzi3ObvWThNFymoOQTVPjdzXzc6RACDQ+bBgCBjymfHHO3vcipdEPQU/iPpst6NHvDqBfRTIRRbyQSvYV2v6WdpqRWn/NMddoblC62mJ36+MUGSsSnXdBUji0FewYZblqiiAXf2SKyzrkr4unlBpjEtJd8A8X/X6M8jYG3M1M8YFTppGAE=',
                        provider_name='anthropic',
                    ),
                    ToolCallPart(
                        tool_name='read_file',
                        args='{"path": "test_reports.py"}',
                        tool_call_id='toolu_01241hWGvoGUUtbdkhKEzAEf',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011CegzVFpSonDMdFGAhrjUp',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='read_file',
                        content="""\
[test_reports.py | 12 lines | hash:5e03fb920a9d]
     1	from cli import run
     2	from incidents import Incident
     3	
     4	
     5	def test_text_report_hides_internal_by_default() -> None:
     6	    incidents = [Incident('API down', 'critical'), Incident('Secret host', 'warning', internal=True)]
     7	    assert run(incidents) == '[CRITICAL] API down'
     8	
     9	
    10	def test_text_report_can_include_internal() -> None:
    11	    incidents = [Incident('Secret host', 'warning', internal=True)]
    12	    assert run(incidents, include_internal=True) == '[WARNING] Secret host'
""",
                        tool_call_id='toolu_01241hWGvoGUUtbdkhKEzAEf',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.
</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.

You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature='CAISpgIKjwEIERgCKkAcGJG7/lb6qLTtg5EY3gulRVVJa5E9wj8TOkZLk2yNUqV5TkgFpSmf/Vi6S3aW/+GGLS2dLV8LjrqhPSBz/STpMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiqAGR/ebUBhIMnKwBTPK9Qi/cMBYiGgxK2cQSf2owRua1OjoiMErQaaMNWEcPB6pJkc74b2x0fWdOJl5lXNC742zy2U9wMErsBxF637x6Toe6eHSKwSpEP4xfQP44lFTW1EZS8O3Ma146XSa3831hy8Su22x+R0azoL9krOFZSD1Ag55bJgsUqQLkJVomOPgdnxb508lwP4xgvyYYAQ==',
                        provider_name='anthropic',
                    ),
                    ToolCallPart(
                        tool_name='update_task_statuses',
                        args=IsStr(
                            regex='\\{"updates":\\ \\[\\{"task_id":"[0-9a-f]{8}","status":"completed"\\},\\{"task_id":"[0-9a-f]{8}","status":"in_progress"\\}\\]\\}'
                        ),
                        tool_call_id='toolu_01WD8BJdfob8pBNrAaPhL58f',
                    ),
                    ToolCallPart(
                        tool_name='edit_file',
                        args='{"path": "test_reports.py", "old_text": "from cli import run\\nfrom incidents import Incident\\n", "new_text": "import json\\n\\nfrom cli import run\\nfrom incidents import Incident\\nfrom service import render_report\\n", "expected_hash": "5e03fb920a9d"}',
                        tool_call_id='toolu_01Dt7vxWB9U9xNMuHkJexcM2',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011CegzVZusffDKMsiHkgDfK',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='update_task_statuses',
                        content=IsStr(
                            regex="""\
Updated\\ 2\\ step\\(s\\):\\
\\-\\ \\[[0-9a-f]{8}\\]\\ Implement\\ JSON\\ renderer\\ and\\ format\\ selection\\ in\\ service/CLI\\ preserving\\ defaults\\ \\->\\ completed\\
\\-\\ \\[[0-9a-f]{8}\\]\\ Add\\ tests\\ covering\\ service\\ and\\ CLI\\ paths\\ incl\\.\\ internal\\ opt\\-in,\\ run\\ full\\ suite\\ \\->\\ in_progress\
"""
                        ),
                        tool_call_id='toolu_01WD8BJdfob8pBNrAaPhL58f',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='edit_file',
                        content='Edited test_reports.py. [hash:ed22ff01a563]',
                        tool_call_id='toolu_01Dt7vxWB9U9xNMuHkJexcM2',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.
</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.

You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='edit_file',
                        args="{\"path\": \"test_reports.py\", \"old_text\": \"def test_text_report_can_include_internal() -> None:\\n    incidents = [Incident('Secret host', 'warning', internal=True)]\\n    assert run(incidents, include_internal=True) == '[WARNING] Secret host'\\n\", \"new_text\": \"def test_text_report_can_include_internal() -> None:\\n    incidents = [Incident('Secret host', 'warning', internal=True)]\\n    assert run(incidents, include_internal=True) == '[WARNING] Secret host'\\n\\n\\ndef test_service_json_report_hides_internal_by_default() -> None:\\n    incidents = [\\n        Incident('API down', 'critical'),\\n        Incident('Secret host', 'warning', internal=True),\\n        Incident('Slow queries', 'minor'),\\n    ]\\n    payload = json.loads(render_report(incidents, format_name='json'))\\n    assert payload == {\\n        'incidents': [\\n            {'title': 'API down', 'severity': 'critical'},\\n            {'title': 'Slow queries', 'severity': 'minor'},\\n        ]\\n    }\\n\\n\\ndef test_service_json_report_can_include_internal() -> None:\\n    incidents = [\\n        Incident('API down', 'critical'),\\n        Incident('Secret host', 'warning', internal=True),\\n    ]\\n    payload = json.loads(\\n        render_report(incidents, include_internal=True, format_name='json')\\n    )\\n    assert payload == {\\n        'incidents': [\\n            {'title': 'API down', 'severity': 'critical'},\\n            {'title': 'Secret host', 'severity': 'warning'},\\n        ]\\n    }\\n\\n\\ndef test_cli_json_report_hides_internal_by_default() -> None:\\n    incidents = [\\n        Incident('API down', 'critical'),\\n        Incident('Secret host', 'warning', internal=True),\\n    ]\\n    payload = json.loads(run(incidents, format_name='json'))\\n    assert payload == {\\n        'incidents': [{'title': 'API down', 'severity': 'critical'}]\\n    }\\n\\n\\ndef test_cli_json_report_can_include_internal() -> None:\\n    incidents = [\\n        Incident('Secret host', 'warning', internal=True),\\n        Incident('API down', 'critical'),\\n    ]\\n    payload = json.loads(run(incidents, include_internal=True, format_name='json'))\\n    assert payload == {\\n        'incidents': [\\n            {'title': 'Secret host', 'severity': 'warning'},\\n            {'title': 'API down', 'severity': 'critical'},\\n        ]\\n    }\\n\\n\\ndef test_text_remains_default_format() -> None:\\n    incidents = [Incident('API down', 'critical')]\\n    assert run(incidents) == render_report(incidents) == '[CRITICAL] API down'\\n\"}",
                        tool_call_id='toolu_01HqiBxNbLgYmkUaokW95eVS',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011CegzW4teijuCSztG7gwyJ',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='edit_file',
                        content='Edited test_reports.py. [hash:187d7d563c64]',
                        tool_call_id='toolu_01HqiBxNbLgYmkUaokW95eVS',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.
</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.

You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='run_command',
                        args='{"command": "pytest -q"}',
                        tool_call_id='toolu_019Auj79SpCZfWoPrUaEex9Y',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011CegzWqbjCw2j6asK7UkzS',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='run_command',
                        content=IsStr(regex='(?s)\\[stdout\\].*\\b7 passed\\b.*'),
                        tool_call_id='toolu_019Auj79SpCZfWoPrUaEex9Y',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.
</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.

You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature='CAISpQIKjwEIERgCKkAM+/wREiaEvIU7LNKpnsnBHdO9s9P8xip1iu14cKEGkkcU+LNhTX1bNjDeXAsLOOGFZqj1Wpzs/a4Hu9GXvbiBMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiqAGk/ebUBhIMvA7+xJMqdxt5nPXvGgyI2CPkJV1rIjLGM54iMPklgdVbC2wmWrmNXIrwm56S5n6m5VrROJAP6LMB40LsWT7qIi4RgwCKPGugSp36gipDRU3pgXhyZuOdVoiE4tM25/aWghwvv/ObW8Fwwbay8UdqMQqYAc42SgnLif24K3qjJndv/kOd96am+TCj8UpVMixsaBgB',
                        provider_name='anthropic',
                    ),
                    ToolCallPart(
                        tool_name='update_task_status',
                        args=IsStr(regex='\\{"task_id":\\ "[0-9a-f]{8}",\\ "status":\\ "completed"\\}'),
                        tool_call_id='toolu_01Go6DqeVPV5isoRSvUiUQBV',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011CegzX8HZsRaB79XPkMqzq',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='update_task_status',
                        content="Updated step 'Add tests covering service and CLI paths incl. internal opt-in, run full suite' status to 'completed'.",
                        tool_call_id='toolu_01Go6DqeVPV5isoRSvUiUQBV',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                instructions="""\
<context-file path="AGENTS.md">
Run tests with `pytest -q`. Keep domain policy separate from presentation and preserve public defaults.
</context-file>

Call `inventory_agent_context` to map where this repo keeps its coding-assistant setup (instruction dirs, skills, sub-agents, and hooks) so you can read and translate it.

You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` as soon as it is fully done. Pass the full plan every time you call `write_plan`. Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move steps between statuses, and `read_plan` to see step ids before a granular edit.

You can delegate self-contained tasks to these sub-agents using the `delegate_task` tool. Each runs in its own fresh context and does not see this conversation, so pass everything it needs.

Available sub-agents:
- explorer: Explore the codebase and answer questions without modifying anything\
""",
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content='',
                        signature='CAISpgIKjwEIERgCKkAqABtq4tv2sK8i4PVZ6/Jt5zvgg+QguWvFpI73j1ijRsr+Sxj9k4/XkboP/wn82MSdhUT0tiQKyPMQ2kOCguXyMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiqAGx/ebUBhIMQOmmdphcuVlKE2HiGgxTU+eQCgmbVklPbeIiMM/zbi/MfGofx/dBowR837236okSebMAD8hhW7OTyxcMHhx/AQNB89zCYIOOL8cDkCpEDgi3mfdr8Ix1/26tHwMFXCG2RRHtBplrhrmi3Pov4mzyyS4AGBvIcdyTPGUDnFoTNRr7xjhKwr3Knnvq3eqQX8G2ytIYAQ==',
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
All done — machine-readable incident reports are wired in end to end. Summary of what happened:

## 1. Compatibility investigation (read-only, via explorer sub-agent)

Key findings that shaped the implementation:
- **Layering:** `cli.py` → `service.render_report` → `policy.visible_incidents` → renderer; the README convention is "each output format has its own renderer module," and renderers never see unfiltered incidents.
- **Callers to protect:** `dashboard.incident_card` calls `render_report(items)` with no kwargs (relies on all defaults), `digest.internal_digest` opts into internals, and `cli.run` mirrors the service signature. All expect a `str` return.
- **Privacy invariant:** the `False` default for `include_internal` lives at the public boundary; `policy.py` requires an explicit value; internal incidents must never reach a renderer without opt-in.

## 2. Implementation

- **`json_renderer.py` (new):** `render_json(incidents) -> str` — `json.dumps` of `{"incidents": [{"title": ..., "severity": ...}, ...]}` preserving input order. Pure presentation; no knowledge of the `internal` flag.
- **`service.py`:** added keyword-only `format_name: str = 'text'` with a renderer registry (`text` / `json`); unknown formats raise `ValueError`. The privacy filter runs **before** format dispatch, so both formats share the identical `visible_incidents` output. Signature, return type, and both defaults are unchanged for existing callers.
- **`cli.py`:** mirrors the service — keyword-only `format_name='text'` forwarded through, existing call shapes (`run(incidents)`, `run(incidents, include_internal=True)`) untouched.
- `dashboard.py` and `digest.py` needed no changes and keep working via defaults.

## 3. Tests

Added five tests alongside the two existing ones:
- Service JSON: hides internal by default, includes it with explicit opt-in (order-preserving).
- CLI JSON: same pair through the adapter.
- A guard asserting text remains the default for both `run` and `render_report`.

`pytest -q`: **7 passed** — including the two pre-existing text tests, confirming defaults and privacy behavior are fully preserved.

(Note: `inventory_agent_context` found no `.claude`/`.agents`/`.codex`/`.grok` setup in this repo; the only conventions were in `AGENTS.md` and `README.md`, both of which were followed.)\
"""
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_011CegzXPfnTET53CiHy2gfF',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert (workspace / 'json_renderer.py').is_file()
    assert 'format_name' in (workspace / 'service.py').read_text()
    assert 'format_name' in (workspace / 'cli.py').read_text()
    completed = await asyncio.create_subprocess_exec(
        'pytest', '-q', cwd=workspace, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    output, _ = await completed.communicate()
    assert completed.returncode == 0, output.decode()
    tool_names = {
        part.tool_name for message in result.all_messages() for part in message.parts if isinstance(part, ToolCallPart)
    }
    assert {'write_plan', 'delegate_task', 'write_file', 'edit_file', 'run_command'} <= tool_names
