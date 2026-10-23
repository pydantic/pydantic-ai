
# Approval Reviewer

Let a second model approve the low-risk tool calls that need approval, and leave the rest to a person.

> [!NOTE]
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/approval_reviewer/)

> [!TIP]
> **Approval Reviewer or Tool Call Judge?** Both ask a second model one yes/no question about a tool call, but give it opposite powers. Use [Tool Call Judge](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/tool_call_judge/) when you want a model to say no: its `yes` blocks a call, whether or not a person is watching. Use Approval Reviewer when you want a model to say yes on a person's behalf: it reviews only calls that already need approval, its `no` approves the call, and everything else still goes to the person. A judge cannot do the second job, because it runs when a call executes, after the person has already approved it.

## The problem

[Human approval](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/#human-in-the-loop-tool-approval) is the dependable gate for a tool like `run_shell`, but most of the calls it stops are harmless: `ls`, `git status`, `rg TODO`. A person asked to approve every one of them stops reading the requests, and the request that mattered gets the same click as the rest. Allowlists cut the volume, but which calls are risky depends on the arguments, and a list only covers the shapes you anticipated.

## The solution

`ApprovalReviewer` answers approval requests before a person sees them. You give it one risk question; for each call that needs approval, a second model answers `yes`, `no`, or `unsure`. A `no` approves the call on the person's behalf and it runs in the same run step. Everything else is left for the person, exactly as it would have been without the reviewer.

The reviewer can only take a question off the person's plate. It never denies a call, never runs one itself, and does nothing for calls that do not need approval.

For the models in this example, install their provider extra and set an API key:

uv:

```bash
uv add pydantic-ai-harness "pydantic-ai-slim[anthropic]"
```

pip:

```bash
pip install pydantic-ai-harness "pydantic-ai-slim[anthropic]"
```

```bash
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```

```python
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai_harness.approval_reviewer import ApprovalReviewer

reviewer = ApprovalReviewer(
    'anthropic:claude-haiku-5-5',
    tools=['run_shell'],
    question='Could this command delete or overwrite data, reach the network, or expose secrets?',
)
agent = Agent(
    'anthropic:claude-fable-5-1',
    capabilities=[reviewer],
    output_type=[str, DeferredToolRequests],
)


@agent.tool_plain(requires_approval=True)
def run_shell(command: str) -> str:
    # Keep the example non-destructive. A real tool runs the command in a sandbox here.
    return f'Would run: {command}'


result = agent.run_sync('How many Python files are in this directory?')
if isinstance(result.output, DeferredToolRequests):
    for call in result.output.approvals:
        print(f'Needs your approval: {call.tool_name} {call.args}')
else:
    print(result.output)
```

With this question, a call like `find . -name '*.py' | wc -l` is expected to be approved and run without stopping the run, and a call like `curl https://example.com/install.sh | sh` to be left in the run's `DeferredToolRequests` output for you to approve or deny, the same way it would be without the reviewer.

The reviewing model returns only the literal `yes`, `no`, or `unsure`, not a free-text explanation. Where a provider reports its own answer confidence in response metadata, it is read from there.

## Which calls are reviewed

Only calls that need approval reach the reviewer. A call needs approval when its tool is registered with `requires_approval=True` or raises [`ApprovalRequired`](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/#human-in-the-loop-tool-approval), which includes a [`ToolGuardrail`](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/guardrails/) that returns `GuardrailResult.approve()` and a [`ToolCallJudge`](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/tool_call_judge/) with `on_uncertain='ask'`. Calls that run without approval are not reviewed and cost nothing. Externally executed calls (`CallDeferred`) are not approval requests, so they are left alone too.

`tools` narrows that further. It is a [`ToolSelector`](https://pydantic.dev/docs/ai/api/pydantic-ai/tools/#pydantic_ai.tools.ToolSelector), the same selector the other tool-scoped capabilities take:

| Value | Reviews approval requests for |
|---|---|
| `'all'` (default) | every function tool |
| `['run_shell', 'write_file']` | tools with those names |
| `{'risk': 'low'}` | tools whose metadata includes those pairs |
| `lambda ctx, tool_def: ...` | whatever the predicate returns |

Approval requests for tools the selector does not match go to the person unreviewed. `'all'` also covers the requests a `ToolGuardrail` or a `ToolCallJudge` raises and any approval-required tool added later, all answered by the one `question`, so name the tools the question was written for.

## What happens to the rest

| Reviewer result | What happens |
|---|---|
| `no` | the call is approved and runs in the same run step |
| `yes` | the call is left for the person |
| `unsure` | the call is left for the person |
| the model raised, failed validation, or the run's usage limits could not fit the review | the call is left for the person |

There is no setting for the failure case, because leaving a call for the person is what happens without a reviewer: a reviewer that is down, overloaded, or out of budget makes the run ask, and never makes it approve.

A call left for the person goes where it would have gone without the reviewer. If the agent has a [`HandleDeferredToolCalls`](https://pydantic.dev/docs/ai/capabilities/handle-deferred-tool-calls/) handler, the handler receives only the requests the reviewer left; the reviewer is ordered ahead of it whichever order you list them in. Otherwise the requests end the run as [`DeferredToolRequests`](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) output, so include `DeferredToolRequests` in the agent's output type, and resume with `DeferredToolResults` once the person decides. A call the person decides is not reviewed again.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, RunContext
from pydantic_ai_harness.approval_reviewer import ApprovalReviewer


def ask_in_terminal(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
    approvals = {
        call.tool_call_id: input(f'Run {call.tool_name} {call.args}? [y/N] ') == 'y'
        for call in requests.approvals
    }
    return DeferredToolResults(approvals=approvals)


agent = Agent(
    'anthropic:claude-fable-5-1',
    capabilities=[
        ApprovalReviewer('anthropic:claude-haiku-5-5', tools=['run_shell'], question='Could this destroy data or leak secrets?'),
        HandleDeferredToolCalls(handler=ask_in_terminal),
    ],
)
```

## Working with Tool Call Judge

[`ToolCallJudge`](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/tool_call_judge/) and `ApprovalReviewer` ask the same kind of question at different points, and in opposite directions. A judge can only narrow what runs: its `yes` blocks a call. A reviewer can only widen what runs without a person: its `no` approves a call.

Together they sort every call into blocked, run, or sent to a person. Let the judge block what must never run and escalate what it cannot settle with `on_uncertain='ask'`, and let the reviewer approve the escalations that are safe enough to run unattended:

```python
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai_harness.approval_reviewer import ApprovalReviewer
from pydantic_ai_harness.tool_call_judge import ToolCallJudge

agent = Agent(
    'anthropic:claude-fable-5-1',
    capabilities=[
        ToolCallJudge(
            'anthropic:claude-haiku-5-5',
            tools=['run_shell'],
            question='Could this command delete or overwrite data outside the project directory?',
            on_uncertain='ask',
        ),
        ApprovalReviewer(
            'anthropic:claude-haiku-5-5',
            tools=['run_shell'],
            question='Could this command delete or overwrite data, reach the network, or expose secrets?',
        ),
    ],
    output_type=[str, DeferredToolRequests],
)


@agent.tool_plain
def run_shell(command: str) -> str:
    return f'Would run: {command}'
```

| Judge | Reviewer | What happens |
|---|---|---|
| `no` | not asked | the call runs |
| `yes` | not asked | the call is blocked |
| `unsure` or failed | `no` | the call is approved, and the judge is asked again before it runs |
| `unsure` or failed | `yes`, `unsure`, or failed | the call is left for the person |

`run_shell` is not registered with `requires_approval=True`, so the judge's escalations are the only approval requests the reviewer sees for it. On a tool that also requires approval, the reviewer answers those requests too, under the same `question`.

A call the reviewer approves is judged again, like any approved call ([working with human approval](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/tool_call_judge/#working-with-human-approval)): a `yes` still blocks it, and an `unsure` now lets it run. An escalated call that runs costs two judgements and one review.

Scope the reviewer's `tools` away from the judge's when you want the judge's escalations to reach a person.

## What the reviewer sees

By default, one review is a fresh internal-agent run whose entire input is:

- the fixed reviewer instructions, including your `question`
- a `<tool_call>` block with the tool name and the call's arguments as JSON, as the model made them: the same call a person would be shown

The outer agent's conversation, instructions, and tool results are not forwarded. That keeps each review small and cheap, and it keeps the reviewer away from text the agent read from pages, files, and other tool results.

Set `include_conversation=True` when the question needs the conversation that led to the call, such as "does this go beyond what the user asked for?". The reviewer then also receives a `<conversation>` block: user messages, assistant messages, tool calls, and tool results, clamped to the most recent `conversation_window` tokens (estimated at ~4 characters per token). System prompts and thinking parts are left out.

Turning it on widens the reviewer's own prompt-injection surface, because the transcript includes third-party content, and a reviewer that is talked into answering `no` approves the call. The instructions label both blocks as untrusted data, and the `<` of every opening or closing framework tag inside either block is escaped (`<tool_call>`, `<conversation>`, and Pydantic AI framing tags such as `<system>`), so content cannot end its block or forge another. Everything else is embedded as written. That reduces the risk rather than removing it.

Either way, what the reviewer is shown leaves the process: a call's arguments go to the reviewing model's provider, including for calls the reviewer then leaves for the person. Where a tool's arguments can carry credentials or customer data, scope `tools` away from it, or route the reviewer to a provider you already trust with that data.

## Risk tiers

Like `ToolCallJudge`, `ApprovalReviewer` asks one question and takes one answer rather than grading risk on a scale against a threshold: a model's self-reported severity is not calibrated across models, prompts, or days, so a threshold on it would not mean what it says.

Tiering is expressed with the `tools` selector instead. Calls to a tool the reviewer selects can be approved by the model; calls to a tool it does not select always go to a person. Mark tools by what a wrong approval would cost, and select only the tiers you are willing to hand to a model:

```python
from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai_harness.approval_reviewer import ApprovalReviewer

agent = Agent(
    'anthropic:claude-fable-5-1',
    output_type=[str, DeferredToolRequests],
    capabilities=[
        ApprovalReviewer(
            'anthropic:claude-haiku-5-5',
            tools={'risk': 'low'},
            question='Could this command delete or overwrite data, reach the network, or expose secrets?',
        ),
    ],
)


@agent.tool_plain(requires_approval=True, metadata={'risk': 'low'})
def run_shell(command: str) -> str:
    return f'Would run: {command}'


@agent.tool_plain(requires_approval=True, metadata={'risk': 'high'})
def deploy(environment: str) -> str:
    return f'Would deploy to {environment}'
```

A `run_shell` call the reviewer answers `no` for runs without asking anyone; every `deploy` call goes to a person, because no reviewer selects it.

For a different question per tool, add more than one reviewer, each with its own selector, question, and model. Every reviewer receives the requests the ones before it left, so a call is approved when any reviewer that selects it answers `no`. Give reviewers non-overlapping selectors, so that each tool's calls are cleared by the question written for it.

## Using a decision model

Whether a call is risky is a classification, which is what a [decision model](https://pydantic.dev/docs/ai/models/decision/) such as [TypeSafe's Jev](https://pydantic.dev/docs/ai/models/typesafe/) answers, cheaper and faster than a language model. Pass one as the reviewer's model and nothing else changes: the review is one pick-one question over `yes`, `no`, and `unsure`, worded by your `question` and the reviewer instructions, about the `<tool_call>` block. The decision model's confidence in its answer is recorded on `ApprovalReview.confidence` and the `approval_reviewer.confidence` span attribute.

For the models in the example below, install their provider extras and set API keys:

uv:

```bash
uv add pydantic-ai-harness "pydantic-ai-slim[anthropic,typesafe]"
```

pip:

```bash
pip install pydantic-ai-harness "pydantic-ai-slim[anthropic,typesafe]"
```

```bash
export TYPESAFE_API_KEY='your-typesafe-api-key'
export ANTHROPIC_API_KEY='your-anthropic-api-key'
```

A `no` approves the call however unsure the model was of it. To have a language model decide the reviews the decision model was unsure of, put the decision model in front of one in a [`FallbackModel`](https://pydantic.dev/docs/ai/models/overview/#fallback-model) that [falls back on low confidence](https://pydantic.dev/docs/ai/models/decision/#falling-back-on-low-confidence):

```python
from pydantic_ai import ModelAPIError, ModelResponse
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai_harness.approval_reviewer import ApprovalReviewer


def unsure(response: ModelResponse) -> bool:
    confidence = (response.provider_details or {}).get('confidence', {})
    return any(value < 0.8 for value in confidence.values())


reviewer = ApprovalReviewer(
    FallbackModel('typesafe:jev-latest', 'anthropic:claude-haiku-5-5', fallback_on=[ModelAPIError, unsure]),
    tools=['run_shell'],
    question='Could this command delete or overwrite data, reach the network, or expose secrets?',
)
```

The handler runs on every model in the chain, and a language model reports no `confidence`, so Haiku's answer stands. Tune the bar on labelled calls of your own, as in [tuning a threshold on your own data](https://pydantic.dev/docs/ai/models/decision/#tuning-a-threshold-on-your-own-data), and once you have, pin the Jev version you tuned it against (`typesafe:jev-1.13.0`) rather than `jev-latest`. Jev treats the text it judges as data rather than as hostile, so arguments written to steer its answer can move it ([what Jev answers badly](https://pydantic.dev/docs/ai/models/typesafe/#what-jev-answers-badly)): keep the [safety boundary](#safety-boundary) in place, as with any reviewer.

## Composition

Calls in one batch are reviewed one at a time, in the order the model made them, so a batch of approval requests adds one review per selected call to the run step's latency.

Each review's model usage is added to the outer run's usage and respects its usage limits. A review claims its request on the shared usage before it calls the model, so the run's `request_limit` bounds the reviews too. A review the budget cannot fit is not made, and the call is left for the person.

## Durable execution

Inside a durable workflow or flow (Temporal, DBOS, Prefect) each review is a durable operation. It is checkpointed under the name `<agent>__capability__<id>.review`, so a replay reuses the recorded answer instead of asking the model again: a call approved on the first execution is approved on every replay, and the reviewer is billed once.

`on_review` is not part of the durable operation, so a replay calls it again with the recorded review. Deduplicate on `ApprovalReview.tool_call_id` when the callback writes to an external system.

A durable operation is addressed by the capability's `id`, and `ApprovalReviewer` has no default one, because several reviewers on one agent is a normal shape. So a reviewer on a durable-capable agent needs an explicit `id`, distinct per reviewer:

```python
from pydantic_ai_harness.approval_reviewer import ApprovalReviewer

reviewer = ApprovalReviewer(
    'anthropic:claude-haiku-5-5',
    id='shell-reviewer',
    tools=['run_shell'],
    question='Could this command delete or overwrite data, reach the network, or expose secrets?',
)
```

Without one, Pydantic AI refuses to bind the agent, naming the fix. Outside durable execution `id` stays optional.

## Observability

Every review opens a `review tool approval` span on the run's tracer:

| Attribute | Value |
|---|---|
| `approval_reviewer.tool` | tool name |
| `approval_reviewer.tool_call_id` | call identifier |
| `approval_reviewer.model_result` | `yes`, `no`, `unsure`, or `error` |
| `approval_reviewer.decision` | `approve` or `ask` |
| `approval_reviewer.confidence` | answer confidence from provider metadata, when reported |
| `approval_reviewer.error.type` | exception type when the review failed |
| `approval_reviewer.arguments` | the call's arguments, only when `trace_include_content` is on |

The internal reviewer agent is not instrumented, so its prompt (which contains the call's arguments) is not recorded even when global instrumentation is on. Its usage is still added to the outer run's usage. Core reports the approvals themselves in the run's `DeferredToolResultsEvent`.

Pass `on_review` to observe the same decisions from application code:

```python
from pydantic_ai_harness.approval_reviewer import ApprovalReview, ApprovalReviewer

reviews: list[ApprovalReview] = []
reviewer = ApprovalReviewer(
    'anthropic:claude-haiku-5-5',
    tools=['run_shell'],
    question='Could this destroy data or leak secrets?',
    on_review=reviews.append,
)
```

An exception raised by `on_review` is not caught: it ends the run, and approvals already made in the same batch are not applied.

## Safety boundary

An approval a model gives is a model's decision, on input the agent partly controls. `ApprovalReviewer` is a filter that decides which requests a person no longer sees, not a security boundary. Keep authentication, authorization, argument validation, least privilege, sandboxing, and recovery controls inside or below any tool that destroys data, spends money, exposes secrets, or cannot be undone, and scope `tools` to the calls where an automatic approval is acceptable.

## Agent specs

`ApprovalReviewer` publishes an agent-spec schema for a string model identifier, the question, the serializable `tools` forms (`'all'`, a name list, a metadata match), `include_conversation`, `conversation_window`, and the standard capability fields. A live `Model`, a predicate selector, and the `on_review` callback are code-only.
