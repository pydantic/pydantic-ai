# Detect Loops

Notice when an agent repeats the same actions without making progress, and interrupt it before the loop uses up the run's request, token, or cost budget. Use it on agents that run autonomously for many steps.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/detect_loops/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

## Usage

uv:

```bash
uv add pydantic-ai-harness
```

pip:

```bash
pip install pydantic-ai-harness
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import DetectLoops

agent = Agent('anthropic:claude-opus-5-5', capabilities=[DetectLoops()])
```

`DetectLoops` records each model response's tool calls and text as it arrives, and each call's result from the request that follows it. Before each model request, it checks what the run has done so far. When it finds a loop, it adds a system prompt to that request telling the model which loop it is in and asking it to change its approach. The capability adds no tools or instructions, so a run without a loop sends the same requests it would without it.

## What counts as a loop

| Pattern | Reported when | Setting (default) |
|---|---|---|
| `repeated_call` | The same tool call (same tool, same arguments) is made this many times within the last `window` tool calls, whatever it returned. | `repeated_call_threshold` (`5`) |
| `repeated_result` | The same tool call returned the same result each of the last this-many times it was made within `window`. | `repeated_result_threshold` (`3`) |
| `alternating_calls` | Two different tool calls are made in turn, A-B-A-B, for this many A-B cycles. | `alternating_calls_threshold` (`3`) |
| `repeated_text` | This many consecutive responses have no tool calls and the same text, ignoring case and whitespace. This happens when [output validation](https://pydantic.dev/docs/ai/core-concepts/output/#output-validator-functions) keeps asking the model to retry. | `repeated_text_threshold` (`3`) |

`window` (default `10`) is how many of the most recent tool calls the three call-based checks look at. Set a threshold to `None` to turn that check off. A threshold must be at least 2, and a call-based threshold has to fit in `window` (`alternating_calls_threshold` needs twice its value), so a check that could never fire raises `ValueError` when the capability is created.

How calls and results are compared:

- Arguments are compared as canonical JSON with sorted keys, so `{"path": "a", "limit": 1}` and `{"limit": 1, "path": "a"}`, or the same object sent as a JSON string, are the same call.
- A result is the tool return's outcome, its content as structured JSON with sorted keys, and the identifiers of any files it returned. A tool that raises [`ModelRetry`](https://pydantic.dev/docs/ai/tools-toolsets/tools-advanced/#tool-retries) with the same message each time returns the same result, so a call that keeps failing the same way is a `repeated_result` loop.
- Every call in a response counts, in order, including parallel calls in a response that already completes a loop.
- A result is matched to its call only within the request that follows the call's response, so models that reuse tool call ids across responses are still compared correctly.
- Calls and results are recorded when they arrive. `DetectLoops` runs outermost among the agent's capabilities, so it reads a call's result before others, such as [compaction](../compaction/), rewrite the request. Compaction that later drops old messages, or replaces old results and arguments with placeholders, doesn't change what was recorded, so it can't make different calls look alike.
- Only [function tool](https://pydantic.dev/docs/ai/tools-toolsets/tools/) calls count. Calls the provider runs itself ([native tools](https://pydantic.dev/docs/ai/tools-toolsets/native-tools/)) are not compared.

At most one loop is reported per model request, checked in the order `repeated_result`, `alternating_calls`, `repeated_call`, `repeated_text`. The responses that formed it are then not counted again, so the same loop has to build up from scratch before it is reported a second time.

Only the current run's responses are recorded. A run that continues an earlier conversation through `message_history` starts with nothing counted.

### Polling and retries

A tool polled until a job finishes, with arguments that don't change, is a `repeated_call` loop after 5 polls within 10 calls, and a `repeated_result` loop after 3 polls in a row that return the same status. If your agent polls on purpose, raise those thresholds or turn the checks off, or have the tool return something that changes between polls, such as the elapsed time.

## Actions

`action` decides what happens when a loop is detected.

### `'nudge'` (default)

The capability enqueues a [`SystemPromptPart`](https://pydantic.dev/docs/ai/api/pydantic-ai/messages/#pydantic_ai.messages.SystemPromptPart) with [`RunContext.enqueue`](https://pydantic.dev/docs/ai/core-concepts/message-history/#injecting-messages-mid-run), which joins the request being prepared. It keeps the default conversation scope, so like a course correction it keeps applying on later requests instead of being shown once, and later requests send it unchanged. For example:

```text
Your last 3 calls to `read_file` used the same arguments and returned the same result, so calling it again will not change the outcome. Change your approach, or explain what is blocking you.
```

The text names tools using only letters, digits, and `_`; any other character becomes `_`. The real names, and the pattern, are in the part's `metadata`:

```python
{'origin': 'pydantic-ai-harness.detect_loops.nudge.v1', 'pattern': 'repeated_result', 'tool_names': ['read_file']}
```

Find nudges in history by that `origin`, not by their text ([identifying system prompts](https://pydantic.dev/docs/ai/core-concepts/message-history/#identifying-system-prompts)). On a model without inline system prompts, Pydantic AI sends the part as `<system>` user text, as it does any [mid-conversation system prompt](https://pydantic.dev/docs/ai/core-concepts/message-history/#mid-conversation-system-prompts).

A nudge doesn't stop a model that ignores it. Combine it with [`UsageLimits`](https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits) for a hard stop.

### `'error'`

The capability raises `LoopDetectedError`, a subclass of [`AgentRunError`](https://pydantic.dev/docs/ai/api/pydantic-ai/exceptions/#pydantic_ai.exceptions.AgentRunError), which ends the run. The detected loop is on its `loop` attribute:

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness import DetectLoops
from pydantic_ai_harness.detect_loops import LoopDetectedError


def stuck_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'missing.txt'})])


agent = Agent(FunctionModel(stuck_model), capabilities=[DetectLoops(action='error')])


@agent.tool_plain
def read_file(path: str) -> str:
    return 'Error: file not found'


try:
    agent.run_sync('Summarize missing.txt')
except LoopDetectedError as error:
    print(error.loop.pattern, error.loop.tool_names, error.loop.count)
    #> repeated_result ('read_file',) 3
```

### A handler

Pass a sync or async function to decide yourself. It's called with the run context and a `DetectedLoop`: raise from it to stop the run, or call `ctx.enqueue(...)` to steer the model your own way.

```python
import logging

from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import SystemPromptPart
from pydantic_ai_harness import DetectLoops
from pydantic_ai_harness.detect_loops import DetectedLoop

logger = logging.getLogger(__name__)


async def on_loop(ctx: RunContext[object], loop: DetectedLoop) -> None:
    logger.warning('Agent loop: %s %s', loop.pattern, loop.tool_names)
    ctx.enqueue(SystemPromptPart('You are repeating yourself. Summarize what you know and stop.'))


agent = Agent('anthropic:claude-opus-5-5', capabilities=[DetectLoops(action=on_loop)])
```

A `DetectedLoop` has:

- `pattern`: which of the four patterns it is.
- `tool_names`: the tools involved, one or two (none for `repeated_text`).
- `count`: how many calls, cycles, or responses repeated.
- `window`: how many recent tool calls (responses, for `repeated_text`) the check looked at.
- `fingerprints`: short stable hashes of the repeated calls (tool name and canonical arguments) or text. The same call has the same fingerprint in every run. They don't contain the arguments in the clear, but a short argument, such as a user id, can be recovered by hashing candidate values.
- `message`: the text the nudge would send.

## Durable execution

What a run recorded lives in memory, in the instance `for_run` creates for that run. Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) the recording and detection hooks (`after_model_request` and `before_model_request`) run in workflow code, so when Temporal, DBOS, or Prefect replays a run, they run again in the same order over the journaled responses and reach the same decisions. A handler runs there too, so keep it deterministic and free of I/O: log through the engine's workflow logger, or move I/O into a [durable capability operation](https://pydantic.dev/docs/ai/capabilities/custom/#durable-capability-operations). `LoopDetectedError` is an `AgentRunError`, so under Temporal it fails the workflow rather than the workflow task.

## Telemetry

Each detection emits a zero-duration `loop detected` span through `ctx.tracer`, so it shows in the run's trace next to the model request it changed. It's a no-op unless the agent is [instrumented](https://pydantic.dev/docs/ai/integrations/logfire/).

| Attribute | Value |
|---|---|
| `detect_loops.pattern` | The pattern, such as `repeated_result` |
| `detect_loops.action` | `nudge`, `error`, or `handler` |
| `detect_loops.count` | `DetectedLoop.count` |
| `detect_loops.window` | `DetectedLoop.window` |
| `detect_loops.tool_names` | The tools involved |
| `detect_loops.fingerprints` | The hashes of the repeated calls or text, only when `trace_include_content` is enabled |

The span contains no arguments, results, or response text. The fingerprints are left out unless `trace_include_content` is enabled, because a short argument can be recovered from its hash. Checks that find nothing emit nothing.

## Composition

- Several `DetectLoops` on one agent run independently. Each reports the loops its own settings describe, so you can, for example, nudge at the defaults and add a second one with higher thresholds and `action='error'` to stop a run that ignores the nudge.
- [System Reminders](../system_reminders/) re-inject guidance on a cadence or condition, and [Trajectory Judge](../trajectory_judge/) asks a second model whether the run is on track. `DetectLoops` needs no second model and only acts on exact repetition.

`DetectLoops` does not detect a run that keeps exceeding the model's context window, and it doesn't truncate, restart, or roll back the run: compaction and your own handler cover those.

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/detect_loops/).
