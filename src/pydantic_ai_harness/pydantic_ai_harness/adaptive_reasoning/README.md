# Adaptive Reasoning

Give the model more thinking effort on the steps that need it and less on the ones that don't, instead of one effort for the whole run. A step that starts a turn or follows a failed tool call gets deliberate effort; a step that only continues after successful tool results, such as the next read after a file read, gets routine effort. Use it to spend fewer reasoning tokens on the routine steps of a long tool-using run.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/adaptive_reasoning/)

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
from pydantic_ai_harness import AdaptiveReasoning

agent = Agent('anthropic:claude-opus-5-5', capabilities=[AdaptiveReasoning(deliberate='high', routine='low')])
```

`AdaptiveReasoning` sets the unified `thinking` setting for each model request, the same setting the core [Thinking](https://pydantic.dev/docs/ai/capabilities/thinking/) capability sets once per run. Each provider maps the level to its own parameter, and a model that doesn't support thinking ignores it.

| The last request in the history | Effort |
|---|---|
| Only returns tool results, all successful | `routine` (default `'medium'`) |
| Has a user prompt, a failed, retried or denied tool result, or retry feedback on the model's output | `deliberate` (default `'high'`) |

Both fields take any [`ThinkingLevel`](https://pydantic.dev/docs/ai/api/pydantic-ai/settings/#pydantic_ai.settings.ThinkingLevel).

## Custom policy

Pass `select` to replace the default policy. It is called once per step with the run's [`RunContext`](https://pydantic.dev/docs/ai/api/pydantic-ai/tools/#pydantic_ai.tools.RunContext), where the last `ModelRequest` in `ctx.messages` is the one the step answers. When the step resumes a turn the provider paused, `ctx.messages` ends in the suspended `ModelResponse` instead. Return `None` to leave `thinking` as the model, agent and earlier capabilities set it.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.settings import ThinkingLevel
from pydantic_ai_harness import AdaptiveReasoning


def by_step(ctx: RunContext[object]) -> ThinkingLevel | None:
    return 'high' if ctx.run_step <= 2 else 'low'


agent = Agent('anthropic:claude-opus-5-5', capabilities=[AdaptiveReasoning(select=by_step)])
```

## Composition

- `AdaptiveReasoning` overrides a `thinking` set on the model or the agent, and a [Thinking](https://pydantic.dev/docs/ai/capabilities/thinking/) capability listed before it. A `thinking` passed to a run's `model_settings` overrides `AdaptiveReasoning`.
- A provider-specific setting such as `anthropic_thinking` or `openai_reasoning_effort` takes precedence over the unified `thinking` setting, so it also overrides `AdaptiveReasoning`.
- Choosing the model and choosing its effort stay separate: a [model selection capability](https://pydantic.dev/docs/ai/capabilities/select-model/) picks the model for a step, then `AdaptiveReasoning` sets the effort for that model.
- Two `AdaptiveReasoning` on one agent share the default `id`, so they merge into one, and the later one's settings win.

## Prompt caching

Each change of effort is recorded in the request as a [`ModelSettingsDeltaPart`](https://pydantic.dev/docs/ai/api/pydantic-ai/messages/#pydantic_ai.messages.ModelSettingsDeltaPart). A model whose profile sets `supports_mid_conversation_effort` sends the change in place in the conversation, so the cached prefix stays valid. On other models the effort is a request-level parameter, and changing it between requests can invalidate the provider's prompt cache. If that costs more than the reasoning tokens saved, use one effort for the run.

## Durable execution

The default policy reads only the last request in the history, so under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) a replay picks the same effort. A custom `select` should be deterministic for the same reason.

## Telemetry

`AdaptiveReasoning` adds no spans. Each change of effort is in the message history as a `ModelSettingsDeltaPart` and in the event stream as a [`ModelSettingsDeltaEvent`](https://pydantic.dev/docs/ai/api/pydantic-ai/messages/#pydantic_ai.messages.ModelSettingsDeltaEvent).

See the [source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/adaptive_reasoning/).
