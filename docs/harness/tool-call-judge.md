---
title: Tool Call Judge
description: Ask a second model whether a tool call may run, and block it before the tool function executes.
---

# Tool Call Judge

Ask a second model whether a tool call may run, and stop it before the tool function executes.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/tool_call_judge/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

Some tool calls are fine and some are not, and which is which depends on the arguments rather than the tool. `run_shell` is not dangerous; `run_shell(command='rm -rf /')` is. Allowlists and argument validation catch the shapes you anticipated. [Human approval](../deferred-tools.md) catches everything, but needs a person on the other end, which a scheduled or long-running agent does not have.

## The solution

`ToolCallJudge` puts a second model in front of selected tool calls. You give it one risk question; for each selected call it answers `yes`, `no`, or `unsure`. A `yes` blocks the call, a `no` lets it run, and `unsure` follows `on_uncertain`.

The judge runs after the arguments are validated and before the tool function does, so it sees the values the tool would actually receive, and a blocked call leaves no side effects to undo.

For the models in this example, install their provider extra and set an API key:

```bash
pip/uv-add pydantic-ai-harness "pydantic-ai-slim[anthropic]"
```

```bash
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```

```python {names="defined"}
from pydantic_ai import Agent
from pydantic_ai_harness.tool_call_judge import ToolCallJudge

judge = ToolCallJudge(
    'anthropic:claude-haiku-4-5',
    tools=['delete_file'],
    question='Would running this destroy data, spend money, or leak secrets?',
)
agent = Agent('anthropic:claude-fable-5', capabilities=[judge])


@agent.tool_plain
def delete_file(path: str) -> str:
    # Keep the example non-destructive. A real tool enforces its own controls here.
    return f'Deletion requested for {path}'


result = agent.run_sync('Delete ./old-report.txt')
print(result.output)
#> ...
```

A blocked call is not an error. The model receives `denial_message` in place of the tool's result and carries on, so the run continues without the call.

The judging model returns only the literal `yes`, `no`, or `unsure`, not a free-text explanation. That keeps the response contract small and lets typed non-text models answer. Where a provider reports its own answer confidence in response metadata, it is read from there rather than asked for as another output field.

## Which tools are judged

`tools` is a [`ToolSelector`][pydantic_ai.tools.ToolSelector], the same selector the other tool-scoped capabilities take:

| Value | Judges |
|---|---|
| `'all'` (default) | every function tool |
| `['delete_file', 'run_shell']` | tools with those names |
| `{'risk': 'high'}` | tools whose metadata includes those pairs |
| `lambda ctx, tool_def: ...` | whatever the predicate returns |

Calls to tools the selector does not match run unjudged, and cost nothing.

Native tools that the provider executes server-side (hosted web search, code execution) never transit the client, so they are not judged.

## Uncertainty and failure

`on_uncertain` covers every case where the judge did not produce a usable `yes` or `no`: the model answered `unsure`, it raised, it exceeded the run's usage limits, or its output failed validation.

| `on_uncertain` | What happens |
|---|---|
| `'block'` (default) | the call does not run; the model sees `denial_message` |
| `'allow'` | the call runs |
| `'ask'` | the call becomes a human approval request |

`'block'` is the default because a judge that is down or overloaded should not be a way to get a call through. `'ask'` is the right answer when a person is available: an uncertain judge escalates instead of guessing, and the call reaches the caller in the run's [`DeferredToolRequests`](../deferred-tools.md) output. Include `DeferredToolRequests` in the agent's output type when you use it.

```python {names="defined"}
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai_harness.tool_call_judge import ToolCallJudge

judge = ToolCallJudge(
    'anthropic:claude-haiku-4-5',
    tools=['delete_file'],
    question='Would this destroy data?',
    on_uncertain='ask',
)
agent = Agent(
    'anthropic:claude-fable-5',
    capabilities=[judge],
    output_type=[str, DeferredToolRequests],
)
```

## Working with human approval

A tool registered with `requires_approval=True` and a judge are two gates, and both apply. They do not race, because they act at different points in the call's life:

1. The call's arguments are validated and the call is deferred, before any execution hook runs. The judge does not see it and is not billed for it.
2. A person approves or denies it.
3. On denial the call is over. On approval the call is executed, and the judge is asked then, about the same validated arguments the person saw.

So the judge is the second gate, not the first, and the effective rule is that both must allow. The judge can still block a call a person approved; it can never release one a person denied, because a denied call never reaches execution.

That ordering follows from where each gate sits in the agent loop rather than from a policy this capability chose, and it is the conservative direction: adding a judge to a tool that already asks a person can only narrow what runs.

What the judge cannot do is ask again. A person's approval answers the `'ask'` question for that call, so on an approved call an `on_uncertain='ask'` policy allows it rather than raising a second approval request. Without that, a judge that stays uncertain -- one that is down, or one whose question the call genuinely cannot settle -- would hand the same person the same question after every approval and the call would never run. This applies to the judge's own escalation and to a `requires_approval=True` tool alike; `yes` still blocks either.

Use the combination when a person approves the intent and the judge checks the arguments -- a reviewer approving "yes, clean up the old exports" does not want to re-read every path. Use `on_uncertain='ask'` instead of `requires_approval=True` when you want the judge to decide which calls are worth a person's attention.

## What the judge sees

By default, one judgement is a fresh internal-agent run whose entire input is:

- the fixed judge instructions, including your `question`
- a `<tool_call>` block with the tool name and its validated arguments as JSON

The outer agent's conversation, instructions, and tool results are not forwarded. That keeps each judgement small and cheap, and it keeps the judge away from text the agent read from pages, files, and other tool results -- text that can carry instructions aimed at whatever reads it next.

The cost of that default is real: a question like "is this deletion something the user asked for?" cannot be answered from the call alone. Set `include_conversation=True` when the question needs the conversation that led to the call:

```python {names="defined"}
from pydantic_ai_harness.tool_call_judge import ToolCallJudge

judge = ToolCallJudge(
    'anthropic:claude-haiku-4-5',
    tools=['delete_file'],
    question='Does this deletion go beyond what the user asked for?',
    include_conversation=True,
    conversation_window=4_000,
)
```

The judge then also receives a `<conversation>` block: user messages, assistant messages, tool calls, and tool results rendered as a transcript and clamped to the most recent `conversation_window` tokens (estimated at ~4 characters per token). System prompts and thinking parts are left out, so the judge is shown observable behavior rather than the agent's configuration or private reasoning.

Turning it on widens the judge's own prompt-injection surface, because that transcript includes third-party content. The judge instructions label both blocks as untrusted data, and both are escaped before they are embedded. That reduces the risk rather than removing it: content-derived instructions are exactly what a model is least able to discount, which is why the judge is one filter among the controls a sensitive tool needs.

Either way, what the judge is shown leaves the process. The whole validated argument dict goes to the judging model's provider before the verdict comes back, so a call is disclosed even when it is then blocked. Where a tool's arguments can carry credentials or customer data, scope `tools` away from it, or route the judge to a provider you already trust with that data.

## Risk tiers

`ToolCallJudge` asks one question and takes one answer. It has no severity scale and no per-tool thresholds, deliberately: a threshold is only useful when the numbers on either side of it mean something, and a model's self-reported severity for "delete a file" is not calibrated against another model's, another prompt's, or yesterday's.

Tiering is expressed with more than one judge instead. Each instance carries its own selector, question, model, and uncertainty policy, and a call must clear all of them:

```python {names="defined"}
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai_harness.tool_call_judge import ToolCallJudge

agent = Agent(
    'anthropic:claude-fable-5',
    output_type=[str, DeferredToolRequests],
    capabilities=[
        ToolCallJudge(
            'anthropic:claude-haiku-4-5',
            tools={'risk': 'high'},
            question='Is there any chance this is destructive or irreversible?',
            on_uncertain='ask',
        ),
        ToolCallJudge(
            'anthropic:claude-haiku-4-5',
            tools='all',
            question='Is this call clearly outside what the user asked for?',
            on_uncertain='allow',
        ),
    ],
)
```

Destructive tools get a strict question that escalates to a person when the judge is unsure; every tool gets a loose question that does not stop the run when the judge is unavailable. That is what a severity scale would have been used for, with the policy stated instead of encoded in a number.

## Composition

Judges evaluate in capability order and the first block wins: once one judge blocks a call, the remaining judges' hooks do not run and neither does the tool. Order the strict, cheap judge first when the second one's cost matters.

Each judge's model usage is added to the outer run's usage and respects its usage limits, so a judged run's request and token totals include what judging cost. A judgement claims its request on the shared usage before it calls the model, so calls judged in parallel (the model requested several tool calls at once) see each other's in-flight requests and the run's `request_limit` bounds the judges too. A judgement the budget cannot fit is not made, and `on_uncertain` decides the call.

Judging adds a model request in the path of every selected tool call, before the tool runs. Scope `tools` to the calls where the answer can change what happens.

## Durable execution

Inside a durable workflow or flow (Temporal, DBOS, Prefect) the judgement is a durable operation. It is checkpointed under the name `<agent>__capability__<id>.judge`, so a replay reuses the recorded verdict instead of asking the model again: the same call is allowed or blocked the same way every time, and the judge is billed once.

That matters because the judgement is made from `before_tool_execute`, which runs in orchestration context rather than inside a checkpointed unit. An uncheckpointed model call there would be re-made on every replay, paid for each time, and free to answer differently, which would leave the same call allowed on one replay and blocked on the next.

`on_verdict` is not part of the durable operation. It runs from the same hook as the judgement, so a replay calls it again with the recorded verdict. Deduplicate on `ToolCallVerdict.tool_call_id` when the callback writes metrics or records to an external system.

A durable operation is addressed by the capability's `id`, and `ToolCallJudge` has no default one, because several judges on one agent is the normal shape and a shared default would merge them. So a judge on a durable-capable agent needs an explicit `id`, distinct per judge:

```python
from pydantic_ai_harness.tool_call_judge import ToolCallJudge

judge = ToolCallJudge(
    'anthropic:claude-haiku-4-5',
    id='refund-judge',
    tools=['issue_refund'],
    question='Would this refund more than the original charge?',
)
```

Without one, Pydantic AI refuses to bind the agent, naming the fix, rather than silently running the judge uncheckpointed. Outside durable execution nothing changes and `id` stays optional.

## Observability

Every judged call opens a `judge tool call` span on the run's tracer:

| Attribute | Value |
|---|---|
| `tool_call_judge.tool` | tool name |
| `tool_call_judge.tool_call_id` | call identifier |
| `tool_call_judge.model_result` | `yes`, `no`, `unsure`, or `error` |
| `tool_call_judge.verdict` | `allow`, `block`, or `ask` |
| `tool_call_judge.confidence` | answer confidence from provider metadata, when reported |
| `tool_call_judge.error.type` | exception type when the judge failed |
| `tool_call_judge.arguments` | validated arguments, only when `trace_include_content` is on |

The internal judge agent is not instrumented, so its prompt (which contains the call's arguments) is not recorded even when global instrumentation is on. Its usage is still added to the outer run's usage.

A blocked call's `ToolReturnPart` carries the answer, the confidence, and the question under a `tool_call_judge` key in its `metadata`, which the application can read and the model cannot see.

Pass `on_verdict` to observe the same decisions from application code:

```python {names="defined"}
from pydantic_ai_harness.tool_call_judge import ToolCallJudge, ToolCallVerdict

verdicts: list[ToolCallVerdict] = []
judge = ToolCallJudge(
    'anthropic:claude-haiku-4-5',
    tools=['delete_file'],
    question='Would this destroy data?',
    on_verdict=verdicts.append,
)
```

## Safety boundary

A judgement cheap enough to run on every call is exactly why it must not be the only thing between an agent and an irreversible action. `ToolCallJudge` is a filter, not a security boundary: the decision is a model's, on input the agent partly controls. Keep authentication, authorization, argument validation, least privilege, and recovery controls inside or below any tool that destroys data, spends money, exposes secrets, or cannot be undone.

Unlike [`OutputGuardrail`](guardrails.md), this capability does not inspect the agent's final output. It decides on one tool call at a time, at the point where the decision can still prevent the action.

## Agent specs

`ToolCallJudge` publishes an agent-spec schema for a string model identifier, the question, the serializable `tools` forms (`'all'`, a name list, a metadata match), `include_conversation`, `conversation_window`, `on_uncertain`, `denial_message`, and the standard capability fields. A live `Model`, a predicate selector, and the `on_verdict` callback are code-only.

## API reference

::: pydantic_ai_harness.tool_call_judge.ToolCallJudge

::: pydantic_ai_harness.tool_call_judge.ToolCallVerdict
