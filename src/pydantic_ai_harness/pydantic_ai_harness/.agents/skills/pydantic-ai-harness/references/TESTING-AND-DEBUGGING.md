# Testing and Debugging

Read this file when the user wants offline, deterministic tests for an agent that uses harness
capabilities, or when a harness agent fails at run start or keeps retrying. For core `TestModel`,
`FunctionModel`, `Agent.override`, and `capture_run_messages()` basics, load the
`building-pydantic-ai-agents` skill's `references/TESTING-AND-DEBUGGING.md` instead of repeating them here.

## The Pattern: Script the Model, Use Real Capabilities

Harness capabilities are real code (they write files, run commands, call answerers), so script the
model with a `FunctionModel` and let the capabilities execute against temp directories and in-memory
stores. `Agent.override(model=..., workspace=...)` swaps both without touching the production agent;
`workspace=` takes a backend such as `LocalWorkspaceBackend(tmp_dir)`. This also works on an agent
with `ModalSandbox`/`E2BSandbox`/`SpritesSandbox` attached: no sandbox is created, no factory needed.

- `FunctionModel`: return a `ToolCallPart` on the first request and a `TextPart` once results are in.
- `TestModel(call_tools=[])`: smoke test that the agent builds and runs. Plain `TestModel()` calls every
  tool with generated arguments -- real commands and writes for `Shell`/`FileSystem` -- so avoid it here.
- Build the production agent, and every sub-agent, with `defer_model_check=True` (for example
  `Agent('anthropic:claude-opus-5-5', defer_model_check=True, ...)`): a provider model string
  otherwise needs the provider's API key when the `Agent` is constructed, so importing the module in
  a test without keys raises `UserError`. `TrajectoryJudge(model='provider:...')` builds its judge
  agent at construction and needs the key too; pass `agent=Agent(..., defer_model_check=True,
  instructions=...)` instead (you then write the judge instructions). `Advisor`, `SubAgents(models=)`,
  `SummarizingCompaction(model=)`, `Summarize(model=)`, and `LLMReminder` resolve their model lazily.
- Override a sub-agent's model next to the parent's:
  `with parent.override(model=FunctionModel(script)), child.override(model=TestModel()):`.
- Add a throwaway tool to a production agent for one run:
  `agent.run_sync(..., toolsets=[FunctionToolset([my_tool])])` (`pydantic_ai.toolsets`).
- Hosted integrations (`GitHub`, `Linear`, `LogfireMCP`, ...) need a credential when the `Agent` is
  built. Pass `auth` as a function that reads the token from deps; a test that passes deps without a
  token gets a run with no tools from that integration. See
  [Hosted Integrations](./HOSTED-INTEGRATIONS.md).
- `Advisor` under an overridden `FunctionModel` or `TestModel` runs as a local `advisor` function
  tool, so the scripted model sees and can call a tool named `advisor`.
- Assert through `result.all_messages()`: `ToolCallPart` is in `ModelResponse.parts`, `ToolReturnPart`
  and `RetryPromptPart` in `ModelRequest.parts`.

## FileSystem, Shell, and Coder on a Temp Workspace

`FileSystem`, `Shell`, and `Coder` do all I/O through the run's workspace, so point it at a temp
directory. `LocalWorkspaceBackend` is POSIX-only.

```python
import tempfile
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.workspaces import LocalWorkspaceBackend

from pydantic_ai_harness import FileSystem, Shell

agent = Agent(capabilities=[FileSystem(), Shell()])  # production model set elsewhere


def script(messages, info):
    if len(messages) == 1:
        args = {'path': 'notes.txt', 'content': 'hi\n'}
        return ModelResponse(parts=[ToolCallPart('write_file', args)])
    if len(messages) == 3:  # the write result is in; read the file back
        return ModelResponse(parts=[ToolCallPart('run_command', {'command': 'cat notes.txt'})])
    return ModelResponse(parts=[TextPart('done')])


with tempfile.TemporaryDirectory() as tmp:
    with agent.override(model=FunctionModel(script), workspace=LocalWorkspaceBackend(tmp)):
        result = agent.run_sync('write a note')
    print(repr(Path(tmp, 'notes.txt').read_text()))
    #> 'hi\n'

returns = {
    part.tool_name: part.content
    for message in result.all_messages()
    for part in message.parts
    if isinstance(part, ToolReturnPart)
}
print(repr(returns['run_command']))
#> '[stdout]\nhi\n'
```

- Tools in one response run concurrently, so the example reads the file back in the next model turn; do the same whenever a test needs ordering.
- `Coder()` bundles `FileSystem` (`FILE_TOOL_NAMES`: `read_file`, `write_file`, `edit_file`, `list_files`, `grep`), `Shell` (the persistent `shell` tool only), `RepoContext`, `SubAgents`, and more. Script calls to those tool names. `RepoContext` (in `Coder()` by default) observes run events, which makes model requests stream, so a `FunctionModel` without `stream_function=` fails with `AssertionError`; script it with `stream_function=` (below) or use `Coder(repo_context=False)`. The prebuilt `coder_agent` has no model and attaches `LocalWorkspace` on the current directory, so pass `workspace=` explicitly in tests.
- `Shell()` denies `rm`, `dd`, `shutdown`, and similar by default; a denied command comes back to the model as a retry, not an exception.
- `pytest`'s `tmp_path` fixture works the same way: `LocalWorkspaceBackend(tmp_path)`.

```python
import tempfile

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from pydantic_ai_harness.coder import Coder


async def stream(messages, info):  # yield str for text, {index: DeltaToolCall} for calls
    if len(messages) == 1:
        args = '{"command": "echo hi"}'
        yield {0: DeltaToolCall(name='shell', json_args=args, tool_call_id='c1')}
    else:
        yield 'done'


with tempfile.TemporaryDirectory() as tmp:
    agent = Agent(
        FunctionModel(stream_function=stream),
        capabilities=[LocalWorkspace(tmp), Coder()],
    )
    print(agent.run_sync('say hi').output)
    #> done
```

## AskUser with a Scripted Answerer

`AskUser` has no default answerer, so tests pass one that answers deterministically and records what
it was asked. The model receives `{header: [labels]}`, or `DECLINED` text for `AskUserResponse(cancelled=True)`.

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from pydantic_ai_harness import AskUser
from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserRequest, AskUserResponse

asked: list[AskUserRequest] = []


async def pick_last(request: AskUserRequest) -> AskUserResponse:
    asked.append(request)
    answers = [
        AskUserAnswer(header=q.header, selected=(q.options[-1].label,))
        for q in request.questions
    ]
    return AskUserResponse(answers=tuple(answers))


question = {
    'header': 'Database',
    'question': 'Which database?',
    'options': [{'label': 'SQLite'}, {'label': 'Postgres'}],
}


def script(messages, info):
    if len(messages) == 1:
        args = {'questions': [question]}
        return ModelResponse(parts=[ToolCallPart('ask_user_question', args)])
    return ModelResponse(parts=[TextPart(messages[-1].parts[-1].model_response_str())])


agent = Agent(FunctionModel(script), capabilities=[AskUser(answerer=pick_last)])
print(agent.run_sync('set up storage').output)
#> {"Database":["Postgres"]}
print(asked[0].questions[0].header)
#> Database
```

- Schema violations (fewer than 2 options, duplicate headers, more than 10 questions, control characters) are retries to the model, and the answerer is never called.
- An answerer response that does not fit its request (unknown header, label not offered, several picks on single-select) fails the whole run with `ValueError`, as `check_response` would; decline with `cancelled=True` instead.
- `AskUserRequestedEvent`/`AskUserAnsweredEvent` let an observer capability (`@on_event`) assert on the exchange without being the answerer.

## Memory with an In-Memory Store

`Memory()` defaults to `InMemoryStore`. Seed it with `InMemoryStore({'<scope>/<file>': text})`, where
scope is `agent_name` (default `'main'`), prefixed by `namespace/` when one is set, and read `store.files`
afterwards. The snapshot is injected as a user-role `TextContent` starting with `<memory>`.

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from pydantic_ai_harness import Memory
from pydantic_ai_harness.memory import InMemoryStore

store = InMemoryStore({'main/MEMORY.md': '- prefers tabs'})
injected: list[str] = []


def script(messages, info):
    if len(messages) == 1:
        injected.append(messages[0].parts[-1].content[0].content)
        return ModelResponse(parts=[ToolCallPart('write_memory', {'content': '- uses uv'})])
    return ModelResponse(parts=[TextPart('ok')])


Agent(FunctionModel(script), capabilities=[Memory(store=store)]).run_sync('hi')
print(injected[0].splitlines()[-2])
#> - prefers tabs
print(store.files['main/MEMORY.md'])
"""
- prefers tabs
- uses uv
"""
```

`FileStore` needs a workspace: without one the run fails at start with a `UserError`; attach a
`LocalWorkspace` or pass `FileStore(..., workspace=LocalWorkspaceBackend(tmp))`.

## CodeMode: Drive `run_code` Directly

Script the `run_code` call with the snippet you want to test. This needs the `codemode` extra; nested
calls land in the return part's metadata.

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from pydantic_ai_harness import CodeMode

CODE = """\
import asyncio
a, b = await asyncio.gather(double(x=2), double(x=5))
print('computed')
a + b"""


def script(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('run_code', {'code': CODE})])
    return ModelResponse(parts=[TextPart('done')])


agent = Agent(FunctionModel(script), capabilities=[CodeMode()])


@agent.tool_plain
def double(x: int) -> int:
    return x * 2


result = agent.run_sync('go')
(ret,) = [
    p
    for m in result.all_messages()
    for p in m.parts
    if isinstance(p, ToolReturnPart) and p.tool_name == 'run_code'
]
print(ret.content)
#> {'output': 'computed\n', 'result': 14}
print([call.tool_name for call in ret.metadata['tool_calls'].values()])
#> ['double', 'double']
print(sorted(r.content for r in ret.metadata['tool_returns'].values()))
#> [4, 10]
```

`info.function_tools` inside the model function shows what the model was offered: here only `run_code`,
since `double` moved into the sandbox.

## Capture Spans

In an app, `logfire.configure()` plus `logfire.instrument_pydantic_ai()` traces every run; harness
capabilities add their own spans (`memory.write`, nested tool calls under `run_code`). In tests, attach
the core `Instrumentation` capability with an in-memory exporter:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings

from pydantic_ai_harness import Memory

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
tracing = Instrumentation(settings=InstrumentationSettings(tracer_provider=provider))


def script(messages, info):
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('write_memory', {'content': '- x'})])
    return ModelResponse(parts=[TextPart('ok')])


Agent(FunctionModel(script), capabilities=[Memory(), tracing]).run_sync('hi')
names = [span.name for span in exporter.get_finished_spans()]
print('memory.write' in names, 'execute_tool write_memory' in names)
#> True True
```

With Logfire's pytest plugin, the `capfire` fixture (`logfire.testing.CaptureLogfire`) captures the same
spans: call `Agent.instrument_all(True)` in a fixture, then read `capfire.exporter.exported_spans_as_dict()`.

## Debug Common Failures

**Missing workspace.** `FileSystem`, `Shell`, `Coder`, and `Memory(store=FileStore(...))` fail the run
at start, before any model request:

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError

from pydantic_ai_harness import Shell

try:
    Agent('test', capabilities=[Shell()]).run_sync('hi')
except UserError as e:
    print(str(e).split('. ')[0])
    #> `Shell` needs a workspace, but none is attached to this run
```

Fix: add `LocalWorkspace('.')` (`from pydantic_ai.capabilities import LocalWorkspace`) or a sandbox
capability such as `ModalSandbox()` to `capabilities`, or pass `workspace=` to the run. If the message
says the history "continues in workspace `<provider>:<id>`", the conversation was started elsewhere;
attach the capability for that provider or pass `workspace='new'`.

**Missing extra.** Importing a capability without its extra usually raises `ImportError` naming the package and
the install command, e.g. `pydantic-monty is required for CodeMode. Install it with: pip install
"pydantic-ai-harness[code-mode]"`. Install the extra from that capability's reference or docs page.

**Monty restrictions.** Generated code that breaks a sandbox rule surfaces as a `RetryPromptPart` on
`run_code`, and the run keeps going until `max_retries` is used up. Typical contents:
`ModuleNotFoundError: No module named 'numpy'` (third-party import) and
`RuntimeError: 'datetime.now' is not supported in this environment` (clock without `os_access`). See
[Code Mode](./CODE-MODE.md#sandbox-restrictions) for the rules and how to grant host access.

**Asserting a `SystemReminders` reminder.** It is ephemeral (absent from `result.all_messages()`), so
check it inside the script. `interval` counts model requests from 1; on a firing request,
`messages[-1].parts[-1]` is a `UserPromptPart` whose content list ends with
`'<system-reminder>\n<text>\n</system-reminder>'` (raw text with `Reminder(tag=None)`), after a
`CachePoint` when the request has user content.

**Looks like it never fires.** Context-window-driven capabilities (the compaction family) cannot
resolve a window for `TestModel`, so thresholds default high; pass `context_window=` or
`fallback_context_window=` in the test.

**Token thresholds under a scripted model.** `FunctionModel` and `TestModel` estimate usage by counting
words (splitting on whitespace and `",.:`), and the compaction family anchors its token count on the
latest response's `input_tokens`. A payload with no separators, such as `'x' * 4000`, counts as about one
token once a response follows it, so `max_tokens`/`max_fraction` may never fire. Use real words, or set
the count from the script with `ModelResponse(..., usage=RequestUsage(input_tokens=...))`
(`from pydantic_ai.usage import RequestUsage`); `tokenizer=` measures only the messages after that
anchor. `ToolOutputLimits` measures characters (or ~4 chars/token, or `tokenizer=`, with
`over_tokens=True`) and does not read model usage.

**Summary requests reach the script.** `SummarizingCompaction` without `model=` sends its summary
request to the run's model, so the scripted function receives it too: that call has no function tools
(`info.function_tools == []`) and `info.instructions` starts with
`'You are a context summarization assistant'` (the default `instructions=`). Return the summary text
there, or pass `model=` a separate `FunctionModel`.

## See also

- https://pydantic.dev/docs/ai/guides/testing/
- https://pydantic.dev/docs/ai/core-concepts/workspace/
- https://pydantic.dev/docs/ai/integrations/logfire/
- https://pydantic.dev/docs/ai/harness/filesystem/
- https://pydantic.dev/docs/ai/harness/shell/
- https://pydantic.dev/docs/ai/harness/coder/
- https://pydantic.dev/docs/ai/harness/ask-user/
- https://pydantic.dev/docs/ai/harness/memory/
- https://pydantic.dev/docs/ai/harness/code-mode/
