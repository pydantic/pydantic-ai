import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
                        signature='CAISpwIKiAEIEBgCKkAk1XO/eoBeSG50HAxPVClm6IHVzzs7ZyrMCzX8kvZTblNnUWMRwPVHfeTXNkBI8+k7wQE08bv3RKZmhMp5p51HMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiEgwGiI993f53kP+eUUAaDBTHmN5f8AVcJaipfiIwC+LoARVvpoV9TG/x6SfKPp1cu7mYtvwFN4Kbt6QYbF5bqI6/5g4H2ItU4SJYqooCKkw75YhQ46mEfVIpALTHFiRow28RkOL90s1Y9Dtkfxuz695sPdkYATXxG6QjfnMi2uDwxH9Jt2YgVqNZVlArzSvSwm85oiaNIH1t6A95GAE=',
                        provider_name='anthropic',
                    ),
                    ToolCallPart(
                        tool_name='inventory_agent_context',
                        args={},
                        tool_call_id='toolu_01BqLzyjHnafXwTxPCSwH6xf',
                    ),
                    ToolCallPart(
                        tool_name='list_directory',
                        args={'path': '.'},
                        tool_call_id='toolu_01JzqeGbs5V2ES4WzgC4dYNM',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NTcSntjGYGgpb3CLMo',
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
                        tool_call_id='toolu_01BqLzyjHnafXwTxPCSwH6xf',
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
                        tool_call_id='toolu_01JzqeGbs5V2ES4WzgC4dYNM',
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
                        args={
                            'items': [
                                {
                                    'content': 'Run read-only compatibility investigation of the public service, consumers, privacy policy, and renderer boundaries',
                                    'status': 'in_progress',
                                    'active_form': 'Investigating compatibility surface',
                                },
                                {
                                    'content': 'Implement JSON renderer and format selection in service and CLI, preserving text default and privacy behavior',
                                    'status': 'pending',
                                    'active_form': 'Implementing JSON format support',
                                },
                                {
                                    'content': 'Add tests for service and CLI paths incl. internal opt-in, and run the full test suite',
                                    'status': 'pending',
                                    'active_form': 'Testing',
                                },
                            ]
                        },
                        tool_call_id='toolu_01XsB9H9T9D9KAyFb4ZNABPU',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NTx4zdtRT4b8Bn8pEh',
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

1. [~] Run read-only compatibility investigation of the public service, consumers, privacy policy, and renderer boundaries
2. [ ] Implement JSON renderer and format selection in service and CLI, preserving text default and privacy behavior
3. [ ] Add tests for service and CLI paths incl. internal opt-in, and run the full test suite
(0/3 completed)\
""",
                        tool_call_id='toolu_01XsB9H9T9D9KAyFb4ZNABPU',
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
                        signature='CAIS7AIKiAEIEBgCKkCFmSQHLMRAAa/vJSSwrFdcYbWc3sMn2zOBubNZGAP+6/OQW/bfilU77JagNHCUFKmK5ZHVH5RsWJwpN7/PMt0wMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiEgwp7OggVqWhvI4yjrwaDEcjJakEkkUBZzEHiSIwK7TfAhnTs9GT/mdaGvDklYecRFV3gI/gSOUeUCvsHNWiC+xQ6UnSSE0UXg4LE7E6KpABM3vKckrakssvs8t5EuGH8/Htz6k4/QnPnVIRsSkHG4Ox1r27zmbDbJ4qSPy9lnc6l5ADr9VtFEWOZfdDln3F8krOqqIPdvus6y4GHj4IWMftFUSjH3qHMMDxeXbfYT726hHC83aIBaTgitDjyGKG7ehSVF/ANIQ4xoBkfsIWsf9uL+yjZyUruTq64XvWxiVJGAE=',
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="The repo is small; per the task, I'll delegate the read-only compatibility investigation to the explorer sub-agent."
                    ),
                    ToolCallPart(
                        tool_name='delegate_task',
                        args={
                            'agent_name': 'explorer',
                            'task': """\
Read-only investigation of a small Python repo (files: cli.py, dashboard.py, digest.py, incidents.py, policy.py, service.py, text_renderer.py, test_reports.py, README.md, AGENTS.md). Do NOT modify anything. I plan to add a machine-readable (JSON) incident report format end to end, where callers of the public service and CLI adapter can pass format_name="json", producing JSON with an `incidents` array of objects with `title` and `severity` in input order, while text stays the default format and existing internal-incident privacy behavior is preserved.

Report back with:
1. The public service entry point(s): exact function signatures, parameters, default values, and return types in service.py.
2. All consumers/callers of the public service (cli.py, dashboard.py, digest.py, tests): how they call it, what arguments they pass, and what they expect back — anything that would break if signatures/defaults change.
3. The privacy policy logic in policy.py: exactly how internal incidents are filtered/included, what "internal opt-in" looks like (parameter names, defaults), and where in the pipeline filtering happens (service vs renderer).
4. The renderer boundary: how text_renderer.py works, its input data shape (from incidents.py), its output format, and how the service selects/invokes it — is there a format registry or is text hardcoded?
5. The incident data model in incidents.py: fields, especially title, severity, and internal flags.
6. Existing tests in test_reports.py: what behaviors they lock in.
7. Architecture conventions implied by AGENTS.md/README.md (domain policy separate from presentation, preserve public defaults) and your recommendation for where a JSON renderer and format selection should live to fit the existing structure without breaking callers.

Quote relevant code snippets verbatim so I can implement without re-reading everything.\
""",
                        },
                        tool_call_id='toolu_01BR58PvyLqqJu1LYtXCZ48Z',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NUJLPr8tCPyJoLSSXj',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='delegate_task',
                        content="""\
I've read the entire repo (it's small — every file fits below). Note: `inventory_agent_context` shows **no coding-assistant setup exists** (`.claude`, `.agents`, `.codex`, `.grok` all `exists: false`), so the only conventions come from `AGENTS.md` and `README.md`.

---

## 1. Public service entry point — `service.py`

There is exactly one entry point, and text rendering is **hardcoded** at the end of it:

```python
# service.py (entire file)
from incidents import Incident
from policy import visible_incidents
from text_renderer import render_text


def render_report(incidents: list[Incident], *, include_internal: bool = False) -> str:
    visible = visible_incidents(incidents, include_internal=include_internal)
    return render_text(visible)
```

- **Signature:** `render_report(incidents: list[Incident], *, include_internal: bool = False) -> str`
- `include_internal` is keyword-only, defaults to `False` (internal incidents hidden by default).
- Returns `str`. There is **no `format_name` parameter yet** and no registry.

## 2. Consumers/callers of the service

**`cli.py`** — a thin pass-through adapter, keyword-only, same default:

```python
# cli.py
from incidents import Incident
from service import render_report


def run(
    incidents: list[Incident], *, include_internal: bool = False
) -> str:
    return render_report(incidents, include_internal=include_internal)
```

**`dashboard.py`** — relies on the **default** `include_internal=False` (and implicitly on the default output being text):

```python
# dashboard.py
def incident_card(items: list[Incident]) -> str:
    return render_report(items)
```

**`digest.py`** — explicit internal opt-in via keyword:

```python
# digest.py
def internal_digest(items: list[Incident]) -> str:
    return render_report(items, include_internal=True)
```

**`test_reports.py`** — calls only `cli.run` (see §6).

**Breakage risks if you change signatures/defaults:**
- `dashboard.incident_card` passes no keywords → any new required parameter, or changing the default format away from text, breaks it.
- Both keyword call sites use `include_internal=` — don't rename it or make it positional.
- All callers annotate/expect `-> str`, so `render_report` must keep returning `str` for JSON too (a JSON string, not a dict).
- Safe change: add `format_name: str = "text"` as an additional keyword-only param with a default.

## 3. Privacy policy — `policy.py`

```python
# policy.py
def visible_incidents(
    incidents: list[Incident], *, include_internal: bool
) -> list[Incident]:
    if include_internal:
        return incidents
    return [incident for incident in incidents if not incident.internal]
```

- Internal opt-in is the keyword-only param `include_internal` (**no default here** — the default `False` lives only at the public boundary in `service.render_report` / `cli.run`).
- Filtering keeps input order (list comprehension preserves ordering).
- **Filtering happens in the service layer, before rendering**: `service.render_report` calls `visible_incidents(...)` first, then hands the already-filtered list to the renderer. Renderers never see internal incidents unless opted in. Your JSON renderer must be wired the same way (receive `visible`, not the raw list) so privacy is preserved automatically.

## 4. Renderer boundary — `text_renderer.py`

```python
# text_renderer.py (entire file)
from incidents import Incident


def render_text(incidents: list[Incident]) -> str:
    return '\\n'.join(f'[{item.severity.upper()}] {item.title}' for item in incidents)
```

- **Input shape:** `list[Incident]` (already policy-filtered).
- **Output:** `str`, one line per incident, `[SEVERITY_UPPERCASED] title`, `\\n`-joined.
- **Selection:** there is **no format registry** — `service.py` imports `render_text` directly and calls it unconditionally. Format selection/dispatch does not exist yet; you'll be adding it.

## 5. Incident data model — `incidents.py`

```python
# incidents.py (entire file)
from dataclasses import dataclass


@dataclass(frozen=True)
class Incident:
    title: str
    severity: str
    internal: bool = False
```

- Frozen dataclass; fields: `title: str`, `severity: str`, `internal: bool = False`. Positional construction `Incident('API down', 'critical')` is used in tests. Your JSON objects need `title` and `severity` (raw, not uppercased — the uppercasing is a text-renderer presentation choice, not domain data); `internal` should presumably not leak into JSON output.

## 6. Locked-in behaviors — `test_reports.py`

```python
from cli import run
from incidents import Incident


def test_text_report_hides_internal_by_default() -> None:
    incidents = [Incident('API down', 'critical'), Incident('Secret host', 'warning', internal=True)]
    assert run(incidents) == '[CRITICAL] API down'


def test_text_report_can_include_internal() -> None:
    incidents = [Incident('Secret host', 'warning', internal=True)]
    assert run(incidents, include_internal=True) == '[WARNING] Secret host'
```

Locked in:
1. `cli.run(incidents)` with no keywords produces **text** output and **hides internal** incidents → text must remain the default format and `include_internal=False` must stay the default.
2. Exact text format `[CRITICAL] API down` (uppercased severity in brackets).
3. `include_internal=True` reveals internal incidents.

Run with `pytest -q` (per AGENTS.md).

## 7. Conventions and recommendation

**AGENTS.md:** `Keep domain policy separate from presentation and preserve public defaults.`

**README.md:** `Incidents enter through cli.py, public report composition lives in service.py, visibility rules live in policy.py, and each output format has its own renderer module. Internal incidents must not appear in exports unless a caller explicitly opts in.`

The README explicitly states "**each output format has its own renderer module**", so the fit-the-structure design is:

1. **New module `json_renderer.py`**, mirroring `text_renderer.py`:
   ```python
   import json
   from incidents import Incident

   def render_json(incidents: list[Incident]) -> str:
       return json.dumps({'incidents': [
           {'title': i.title, 'severity': i.severity} for i in incidents
       ]})
   ```
   It takes the already-filtered `list[Incident]` and returns `str` — same contract as `render_text`. Input order is preserved because policy filtering and this comprehension both preserve order. Do **not** put `include_internal`/policy logic here (policy ≠ presentation).

2. **Format selection lives in `service.py`** (that's the "public report composition" layer). Add a small renderer mapping there and extend the signature backward-compatibly:
   ```python
   def render_report(incidents: list[Incident], *, include_internal: bool = False,
                     format_name: str = 'text') -> str:
       visible = visible_incidents(incidents, include_internal=include_internal)
       return _RENDERERS[format_name](visible)
   ```
   Keyword-only + defaults `include_internal=False`, `format_name='text'` preserves every existing caller (`dashboard.incident_card` passes nothing; `digest` and `cli` pass only `include_internal=`) and both existing tests.

3. **Thread `format_name` through `cli.run`** the same way (keyword-only, default `'text'`), forwarding to `render_report`.

4. **Do not touch** `policy.py`, `incidents.py`, `dashboard.py`, `digest.py`, or `text_renderer.py`. Privacy is preserved for free because filtering happens in the service before any renderer runs.

5. Add tests mirroring the existing pair: JSON hides internal by default, JSON includes internal on opt-in, and text remains the default when `format_name` is omitted.\
""",
                        tool_call_id='toolu_01BR58PvyLqqJu1LYtXCZ48Z',
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
                        signature='CAISzQIKiAEIEBgCKkDMNtDlve9KbBVe9AxtkOTz3AUMVTOvfZHoE7ceSTohcOgnEbvlPpAQYJzeyst18dmEzvaqAw8ONcA8+DCskrn+Mg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiEgyMsywHgMhmN57AsKIaDEt8Lx/UYESKJiNVcSIwJy8uvBSpWCB1xNHB5swBG/lK3iLxSz8ZXkozL8UHatgi0pfI5QebJCnBrbtS/fQzKnJydUnmEBtmbqhfJvpcUdkOiJfx5Ja7q3q3MvvymAcb+I5HBGoqzDd7LuebB2Ckx1ftHHVUtoVK6aw60mmTrVh/2+rFn4lIcLErx2HbW1e4VDIyLzI8txlsksg7m72YuxqUX/KTT1zwgRhDugb1FAou2osYAQ==',
                        provider_name='anthropic',
                    ),
                    ToolCallPart(
                        tool_name='update_task_statuses',
                        args={
                            'updates': [
                                {'task_id': '1', 'status': 'completed'},
                                {'task_id': '2', 'status': 'in_progress'},
                            ]
                        },
                        tool_call_id='toolu_013MRqVM4b92B5eopr8iHmBg',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NYBKakQPsdydhzLcZP',
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
- Step with id '1' not found.
- Step with id '2' not found.\
""",
                        tool_call_id='toolu_013MRqVM4b92B5eopr8iHmBg',
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
                        tool_name='read_plan',
                        args={},
                        tool_call_id='toolu_01RLNK6ekeGX7djUqAerSCHB',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NYUYvGE2e6VPtRz5e1',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='read_plan',
                        content=IsStr(
                            regex="""\
Current\\ plan:\\
1\\.\\ \\[\\~\\]\\ \\[[0-9a-f]{8}\\]\\ Run\\ read\\-only\\ compatibility\\ investigation\\ of\\ the\\ public\\ service,\\ consumers,\\ privacy\\ policy,\\ and\\ renderer\\ boundaries\\
2\\.\\ \\[\\ \\]\\ \\[[0-9a-f]{8}\\]\\ Implement\\ JSON\\ renderer\\ and\\ format\\ selection\\ in\\ service\\ and\\ CLI,\\ preserving\\ text\\ default\\ and\\ privacy\\ behavior\\
3\\.\\ \\[\\ \\]\\ \\[[0-9a-f]{8}\\]\\ Add\\ tests\\ for\\ service\\ and\\ CLI\\ paths\\ incl\\.\\ internal\\ opt\\-in,\\ and\\ run\\ the\\ full\\ test\\ suite\\
\\
Summary:\\ 0\\ completed,\\ 1\\ in\\ progress,\\ 2\\ pending\
"""
                        ),
                        tool_call_id='toolu_01RLNK6ekeGX7djUqAerSCHB',
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
                        tool_name='update_task_statuses',
                        args={
                            'updates': [
                                {'task_id': 'c0f1bc3e', 'status': 'completed'},
                                {'task_id': '7ac6b145', 'status': 'in_progress'},
                            ]
                        },
                        tool_call_id='toolu_01DUT7MjCMUKxS4HcmdvbFtW',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NYiAUFqeZDf8VPGVXR',
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
- Step with id 'c0f1bc3e' not found.
- Step with id '7ac6b145' not found.\
""",
                        tool_call_id='toolu_01DUT7MjCMUKxS4HcmdvbFtW',
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
                        signature='CAIStQIKiAEIEBgCKkD/VmqAQvmMmBrY+9Bo31/VDeRJmXbBr48wsio9MsgJ4YDtm/7V6Jt3FaC92sdvBlbMQJXNp11kePPNfXf3AKWpMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiEgzNFimLq7KpYxv9u8AaDII/Ar4rTRNTcjhRYyIwaw8y54SUwMRP+SyYiEdyR/kG1UO8f0k98PmkMD83Hrga1mBP9V11P6WocuGObuT4Klrhrzgk/x+4UvfqUnLYTz1MNSHXcZ3bFNaB37cGmR5PWCCMsgOZVufOo/lzkA5LSby16AvT7MI+oI+ydTOrDionJqVqzRGTtwBY1C5416hF6pT81ccJNZy4cKQYAQ==',
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content='Now implementing. First the new JSON renderer module, then wiring format selection into the service and CLI:'
                    ),
                    ToolCallPart(
                        tool_name='write_file',
                        args={
                            'path': 'json_renderer.py',
                            'content': """\
import json

from incidents import Incident


def render_json(incidents: list[Incident]) -> str:
    return json.dumps(
        {
            'incidents': [
                {'title': item.title, 'severity': item.severity}
                for item in incidents
            ]
        }
    )
""",
                        },
                        tool_call_id='toolu_01Vct7gH4SrzsBuqTgEdrHgt',
                    ),
                    ToolCallPart(
                        tool_name='write_file',
                        args={
                            'path': 'service.py',
                            'content': """\
from typing import Callable

from incidents import Incident
from json_renderer import render_json
from policy import visible_incidents
from text_renderer import render_text

_RENDERERS: dict[str, Callable[[list[Incident]], str]] = {
    'text': render_text,
    'json': render_json,
}


def render_report(
    incidents: list[Incident],
    *,
    include_internal: bool = False,
    format_name: str = 'text',
) -> str:
    try:
        renderer = _RENDERERS[format_name]
    except KeyError:
        raise ValueError(f'unknown format: {format_name!r}') from None
    visible = visible_incidents(incidents, include_internal=include_internal)
    return renderer(visible)
""",
                        },
                        tool_call_id='toolu_01VqLXEEb4CUhau7bHvgfqTS',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NYyo4zseN96FodQxd6',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='write_file',
                        content='Wrote 290 chars (14 lines) to json_renderer.py. [hash:a48d755e3cd3]',
                        tool_call_id='toolu_01Vct7gH4SrzsBuqTgEdrHgt',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='write_file',
                        content='Wrote 672 chars (25 lines) to service.py. [hash:a617f3812afa]',
                        tool_call_id='toolu_01VqLXEEb4CUhau7bHvgfqTS',
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
                    TextPart(content='Continuing implementation — threading `format_name` through the CLI adapter:'),
                    ToolCallPart(
                        tool_name='write_file',
                        args={
                            'path': 'cli.py',
                            'content': """\
from incidents import Incident
from service import render_report


def run(
    incidents: list[Incident],
    *,
    include_internal: bool = False,
    format_name: str = 'text',
) -> str:
    return render_report(
        incidents,
        include_internal=include_internal,
        format_name=format_name,
    )
""",
                        },
                        tool_call_id='toolu_01MhKuPqgXVqMeP2aoHRWru3',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NZXypRcMn6qZUMedr8',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='write_file',
                        content='Wrote 318 chars (15 lines) to cli.py. [hash:95b106f6ca26]',
                        tool_call_id='toolu_01MhKuPqgXVqMeP2aoHRWru3',
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
                        signature='CAISvgIKiAEIEBgCKkDm2C+76TSSuVg0yxNtQimuqFCwvHFyohfkuFltg5vI53S/qLZNTz2F2ydWRoFaKCd4jR0NUgp3VOwaFWB5/HfFMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiEgxj53sP/cMMGw+SWy0aDLEG8iGvgeprjQOLiSIwT7JhuutCyDOLPdqpWANBnwntjA3Wp1U+AzEjnc8pUCARObTMhh57EaZIJHuRZZIMKmMLiMlsqioFIHlnpK22nc6fgQCaDkijwEQp44tKRlw3xxlv0Pia+YAwB98TPA4X1ScdN9w5w0A9Q0JHEadBAOAZblLBCdq0b3FUiHQ32pRZGMEh33qwF33kYVsKaJA/4keb9GEYAQ==',
                        provider_name='anthropic',
                    ),
                    TextPart(content='Implementation is complete. Moving to tests:'),
                    ToolCallPart(
                        tool_name='update_task_statuses',
                        args={
                            'updates': [
                                {'task_id': '7ac6b145', 'status': 'completed'},
                                {'task_id': 'b033c624', 'status': 'in_progress'},
                            ]
                        },
                        tool_call_id='toolu_01HSrcV2dei9JuwZyHxajR1o',
                    ),
                    ToolCallPart(
                        tool_name='read_file',
                        args={'path': 'test_reports.py'},
                        tool_call_id='toolu_019SUo2oTjE8xDxuGKPbrD8h',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NZx5f3vRT8RV2knZaX',
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
- Step with id '7ac6b145' not found.
- Step with id 'b033c624' not found.\
""",
                        tool_call_id='toolu_01HSrcV2dei9JuwZyHxajR1o',
                        timestamp=IsDatetime(),
                    ),
                    ToolReturnPart(
                        tool_name='read_file',
                        content="""\
[test_reports.py | 12 lines | hash:5e03fb920a9d]
     1	from cli import run
     2	from incidents import Incident
     3\t
     4\t
     5	def test_text_report_hides_internal_by_default() -> None:
     6	    incidents = [Incident('API down', 'critical'), Incident('Secret host', 'warning', internal=True)]
     7	    assert run(incidents) == '[CRITICAL] API down'
     8\t
     9\t
    10	def test_text_report_can_include_internal() -> None:
    11	    incidents = [Incident('Secret host', 'warning', internal=True)]
    12	    assert run(incidents, include_internal=True) == '[WARNING] Secret host'
""",
                        tool_call_id='toolu_019SUo2oTjE8xDxuGKPbrD8h',
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
                        args={
                            'path': 'test_reports.py',
                            'old_text': """\
from cli import run
from incidents import Incident


def test_text_report_hides_internal_by_default() -> None:\
""",
                            'new_text': """\
import json

from cli import run
from incidents import Incident
from service import render_report


def test_text_report_hides_internal_by_default() -> None:\
""",
                        },
                        tool_call_id='toolu_014ijHqv2JEJVRsj1zAqumCR',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NaKWFqn9WQmsQKtpV6',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='edit_file',
                        content='Edited test_reports.py. [hash:ed22ff01a563]',
                        tool_call_id='toolu_014ijHqv2JEJVRsj1zAqumCR',
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
                        signature='CAISpAIKiAEIEBgCKkDCgLhuNb0rWIHcxCBzPJaaIx16TdND9Cf/7oknhpnLDo65AXSi2JZuil9e1+qrvsQIav+8hjBiXUs65sxohRqYMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiEgw3nLncBvAOE3S1Yd0aDBd/+XxSeFN91+mJ+yIw2oZN6j7jxcuGjN2DGLdWLQremMrAlLr2yoVw3zafDOKqpf6Io0upCn1eOs2JKOnqKknsJh5pZTp/9fVV1zRZiBinRwMiQS0Qs6khihBMY+OIuI1lk8BnZhk5HDeAZrcvo81WO6mn9BOMw/N3HL+jllXJoSoMa6EtbmyYGAE=',
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content='Now appending the JSON tests covering service and CLI paths, order preservation, and privacy defaults/opt-in:'
                    ),
                    ToolCallPart(
                        tool_name='edit_file',
                        args={
                            'path': 'test_reports.py',
                            'old_text': """\
def test_text_report_can_include_internal() -> None:
    incidents = [Incident('Secret host', 'warning', internal=True)]
    assert run(incidents, include_internal=True) == '[WARNING] Secret host'\
""",
                            'new_text': """\
def test_text_report_can_include_internal() -> None:
    incidents = [Incident('Secret host', 'warning', internal=True)]
    assert run(incidents, include_internal=True) == '[WARNING] Secret host'


def test_service_json_report_hides_internal_by_default() -> None:
    incidents = [
        Incident('API down', 'critical'),
        Incident('Secret host', 'warning', internal=True),
        Incident('Latency spike', 'minor'),
    ]
    payload = json.loads(render_report(incidents, format_name='json'))
    assert payload == {
        'incidents': [
            {'title': 'API down', 'severity': 'critical'},
            {'title': 'Latency spike', 'severity': 'minor'},
        ]
    }


def test_service_json_report_can_include_internal() -> None:
    incidents = [
        Incident('API down', 'critical'),
        Incident('Secret host', 'warning', internal=True),
    ]
    payload = json.loads(
        render_report(incidents, include_internal=True, format_name='json')
    )
    assert payload == {
        'incidents': [
            {'title': 'API down', 'severity': 'critical'},
            {'title': 'Secret host', 'severity': 'warning'},
        ]
    }


def test_cli_json_report_hides_internal_by_default() -> None:
    incidents = [
        Incident('API down', 'critical'),
        Incident('Secret host', 'warning', internal=True),
    ]
    payload = json.loads(run(incidents, format_name='json'))
    assert payload == {
        'incidents': [{'title': 'API down', 'severity': 'critical'}]
    }


def test_cli_json_report_can_include_internal() -> None:
    incidents = [
        Incident('Secret host', 'warning', internal=True),
        Incident('API down', 'critical'),
    ]
    payload = json.loads(run(incidents, include_internal=True, format_name='json'))
    assert payload == {
        'incidents': [
            {'title': 'Secret host', 'severity': 'warning'},
            {'title': 'API down', 'severity': 'critical'},
        ]
    }


def test_service_text_remains_default_format() -> None:
    incidents = [Incident('API down', 'critical')]
    assert render_report(incidents) == '[CRITICAL] API down'\
""",
                        },
                        tool_call_id='toolu_01L1zaTYZ3D6TLodiDayQd3b',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NadeumDMvW5mkQThKV',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='edit_file',
                        content='Edited test_reports.py. [hash:24c2441d15d6]',
                        tool_call_id='toolu_01L1zaTYZ3D6TLodiDayQd3b',
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
                        args={'command': 'pytest -q'},
                        tool_call_id='toolu_01PmYMa2AQarjAbNa9s5gh6U',
                    )
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3NbVXHpiKbdPPvPfXx5',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='run_command',
                        content=IsStr(regex=r'(?s)\[stdout\].*\b7 passed\b.*'),
                        tool_call_id='toolu_01PmYMa2AQarjAbNa9s5gh6U',
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
                        signature='CAISoQIKiAEIEBgCKkAXcUjQOiCpoSNnVLH7G/h3kDk17c/pgbw9W/RcskhR5mZGZguGnV73N3OvUI26SwUVc3yy/feuqAn+giUarzZbMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiEgx0IOMd3PDQHmSZob8aDGarqjn82b8m2dhxLSIwZqTTWAJqdVp0A/+76hrIx515C2O6V/NVcoKQrDIozH5hBOIr+6cH6aPUj9xMW5TvKkasY7fYTx1MT10pU4GBAsrKvgkkDhrh3+V1qPUuq4ObqN45Dm/R6feMLq2w/2VMYX8cpiZTsVLO/FDATTmrRQRcAcGqpT8KGAE=',
                        provider_name='anthropic',
                    ),
                    ToolCallPart(
                        tool_name='update_task_status',
                        args={'task_id': 'b033c624', 'status': 'completed'},
                        tool_call_id='toolu_01QcWWef4Mz43xqyUHAJA1y1',
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'tool_use'},
                provider_response_id='msg_011Ce3Nbn4D6kd7RBNCUZzTB',
                finish_reason='tool_call',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='update_task_status',
                        content="Step with id 'b033c624' not found.",
                        tool_call_id='toolu_01QcWWef4Mz43xqyUHAJA1y1',
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
                        signature='CAISnwIKiAEIEBgCKkCY/OWDo+8mNH/T9QTIgYkb0fK58woV+z1kd0T7w/h7xzE0RaAslwfCaY8IWhldKQGvPBl9VE88cAkzGtJp9OwWMg5jbGF1ZGUtZmFibGUtNTgBQgh0aGlua2luZ1okNDRhZTY3NmMtOTU4Zi00ZDY4LTkxMDgtZWFlOWRlN2IzNjZiEgw2YyQ3yKZsHuIn/30aDP8LCo6X1E/7vgl43iIwQmDpOxRqvUgbOnN08FtipDRm3tefba/FpnQ1AZRb0ueRipAY40aLoKMO+9xc5GPgKkR3pfnELk8tEYIrcSmLO3Nu4WcAyeMHU+KivLx9GZ7AGB/Bf1DGAjrFcdRi5kwy0WBMy95MTnuQ8Zah0SNZc2mJMsp1XRgB',
                        provider_name='anthropic',
                    ),
                    TextPart(
                        content="""\
All done — machine-readable incident reports are wired end to end. Summary:

## Investigation (read-only, via explorer sub-agent)
Mapped the compatibility surface before touching code:
- **Public service**: `service.render_report(incidents, *, include_internal=False) -> str`, with text rendering hardcoded and no format registry.
- **Consumers**: `cli.run` (keyword pass-through), `dashboard.incident_card` (relies on all defaults), `digest.internal_digest` (`include_internal=True`) — so new parameters had to be keyword-only with defaults, and the return type had to stay `str`.
- **Privacy**: `policy.visible_incidents` filters internal incidents in the service layer *before* rendering, so renderers never see unfiltered data.
- **Architecture** (README/AGENTS.md): each output format gets its own renderer module; policy stays separate from presentation.

## Implementation
- **`json_renderer.py`** (new): mirrors `text_renderer.py`'s contract — takes the already-policy-filtered `list[Incident]`, returns a JSON string with an `incidents` array of `{title, severity}` objects in input order. No policy logic, and the `internal` flag never leaks into output.
- **`service.py`**: added a `_RENDERERS` registry (`text`, `json`) and a keyword-only `format_name: str = 'text'` parameter; unknown formats raise a clear `ValueError`. Privacy filtering still happens once, before the renderer runs, so both formats share identical privacy behavior.
- **`cli.py`**: threads `format_name` through to the service with the same default.
- **Untouched**: `policy.py`, `incidents.py`, `text_renderer.py`, `dashboard.py`, `digest.py` — every existing caller and default is preserved.

## Tests
Added five tests alongside the two existing ones (which lock in text-by-default and privacy defaults):
- Service JSON path: hides internal by default (with order preserved across the filtered gap), includes internal on explicit opt-in.
- CLI JSON path: same pair, including explicit `include_internal=True` opt-in.
- Text remains the default format when `format_name` is omitted on the service.

**Full suite: `pytest -q` → 7 passed.**\
"""
                    ),
                ],
                usage=IsInstance(RequestUsage),
                model_name='claude-fable-5',
                timestamp=IsDatetime(),
                provider_name='anthropic',
                provider_url='https://api.anthropic.com',
                provider_details={'finish_reason': 'end_turn'},
                provider_response_id='msg_011Ce3Nc9EvVcvSRPES5zzTU',
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
