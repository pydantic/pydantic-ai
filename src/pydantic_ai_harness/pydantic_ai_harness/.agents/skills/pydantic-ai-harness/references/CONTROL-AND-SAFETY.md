# Control and Safety

Capabilities that constrain, screen, budget, or steer a run: repair malformed tool arguments,
guard prompts/tool calls/outputs, scan tool results for prompt injection, cap spend, ask the user
a question mid-run, re-inject reminders, interrupt repeated-action loops, and review the trajectory with a
judge model. All are
`AbstractCapability`s passed in `Agent(capabilities=[...])`. Only `PromptInjectionDefender` needs
an extra. For plain tool approval (`requires_approval=True`, `ApprovalRequired`,
`DeferredToolRequests`, `HandleDeferredToolCalls`), use the core mechanism documented in the
`building-pydantic-ai-agents` skill (`references/TOOLS-ADVANCED.md`); do not rebuild it here.

## Choose a capability

| I want to ... | Use |
|---|---|
| Stop tool calls failing on trailing commas, single quotes, truncated JSON | `RepairToolArguments()` |
| Refuse or redact a prompt before it reaches the model | `InputGuardrail(guard=...)` |
| Block, redact, or retry the final output | `OutputGuardrail(guard=...)` |
| Block/rewrite tool arguments, redact tool results, or hide tools | `ToolGuardrail(guard=..., result_guard=..., hidden=[...])` |
| Mask API keys, tokens, PII in text | `guardrails.detectors.redact_secrets` / `redact_personal_data` in a guard |
| Get human approval for some tool calls, decided per call | `ToolGuardrail` returning `GuardrailResult.approve()` |
| Get human approval for a whole tool | core `requires_approval=True` (core skill) |
| Detect indirect prompt injection in emails/web pages returned by tools | `PromptInjectionDefender` |
| Ask a second model whether each selected tool call should run | `ToolCallJudge(model, question=..., tools=...)` |
| Let a second model approve low-risk calls that need approval, and ask a person about the rest | `ApprovalReviewer(model, question=..., tools=...)` |
| Cap USD or tokens per day/month/run, per tenant, across workers | `SpendLimits(budgets=[Budget(...)])` |
| Cap tokens/cost for one run only | core `UsageLimits` (no harness capability needed) |
| Let the model ask the user multiple-choice questions and wait | `AskUser(answerer=...)` |
| Counter instruction fade in long runs, cache-safe | `SystemReminders` |
| Have a second model review the run and steer it when it drifts | `TrajectoryJudge` |
| Interrupt an agent that keeps repeating the same tool call, result, or response | `DetectLoops` |

## RepairToolArguments

Runs `json-repair` on raw tool-call argument strings in `before_tool_validate`, for every tool on the
agent. No extra (`json-repair` is a base dependency), no options.

```python
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments


def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(returns[0].content)])
    # Single quotes and a trailing comma: invalid JSON without the repair.
    return ModelResponse(parts=[ToolCallPart('greet', "{'name': 'Ada',}")])


agent = Agent(FunctionModel(model), capabilities=[RepairToolArguments()])


@agent.tool_plain
def greet(name: str) -> str:
    return f'Hello, {name}!'


print(agent.run_sync('Greet Ada').output)
#> Hello, Ada!
```

Gotchas: valid JSON and already-parsed dict args pass through untouched; repair is heuristic and
does not use the tool schema, so missing fields and wrong types still go through normal
validation/retry. `Coder` already includes it.

## Guardrails

`InputGuardrail`, `OutputGuardrail`, `ToolGuardrail`, `GuardrailResult`, `InputBlocked`,
`OutputBlocked`, `GuardrailError` are top-level (`from pydantic_ai_harness import ...`);
`ToolBlocked`, `ToolCallInfo`, `ToolResultInfo`, `detectors` live in `pydantic_ai_harness.guardrails`.

A guard is a sync or async callable taking the value, optionally preceded by `RunContext` (taken
when a second positional parameter is required, for `*args` with another parameter, or when the
second has a default and the first is annotated `RunContext`; a first parameter with no annotation,
`Any`, `object`, or an unresolvable type not naming `RunContext` still gets it, with a
`HarnessDeprecationWarning`). It
returns `bool` (`True` = allow) or a `GuardrailResult`: `allow()`, `block(message=None)`,
`replace(value)`, `retry(message)`, `approve()`. Any other return value (a `'safe'` label, `None`, a
`numpy.bool_`) is read for truthiness with a `HarnessDeprecationWarning` (a truthy label allows the
value); return `bool(...)` instead.

| Verdict | `InputGuardrail` | `OutputGuardrail` | `ToolGuardrail.guard` (args) | `ToolGuardrail.result_guard` |
|---|---|---|---|---|
| block | skip the model call; message becomes the response | raise `OutputBlocked` | skip tool; message is the tool result | message replaces the result |
| replace | rewrite the prompt (text prompts, sequential only) | substitute output | run with new args mapping | substitute result |
| retry | `UserError` | `ModelRetry` to the model (counts output retries) | `ModelRetry` | `ModelRetry` (tool already ran) |
| approve | `UserError` | `UserError` | defer via `ApprovalRequired` | not valid |

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import InputGuardrail, OutputGuardrail
from pydantic_ai_harness.guardrails.detectors import (
    blocked_keywords,
    for_text,
    redact_secrets,
)

agent = Agent(
    TestModel(custom_output_text='Use key sk-ant-abcdefghijklmnopqrstuvwx'),
    capabilities=[
        # A chain: runs in order; a `replace` feeds the next guard; block/retry stops it.
        InputGuardrail(guard=[redact_secrets, blocked_keywords(['internal-only'])]),
        OutputGuardrail(guard=for_text(redact_secrets)),
    ],
)

print(agent.run_sync('Summarise the internal-only roadmap').output)
#> Blocked term: 'internal-only'.
print(agent.run_sync('Which key do I use?').output)
#> Use key [redacted:anthropic_key]
```

`ToolGuardrail(guard=...)` receives a `ToolCallInfo(name, args, tool_call_id)`; `result_guard`
receives a `ToolResultInfo` (the same fields plus `result`), not the raw result, so wrap a text
detector with `for_tool_result_text`. `result_guard` also screens a `ModelRetry`/`ToolFailed`
raised while the tool runs (including inside `CodeMode`), and one raised by hooks of a capability
ordered inside it (another `position='innermost'` one), except from an error hook; `ToolGuardrail`
`retry` verdicts are never screened. `args` is a read-only `mappingproxy` copy: `json.dumps` rejects
it, so serialize or build a `replace` mapping from `dict(call.args)`.

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import GuardrailResult, ToolGuardrail
from pydantic_ai_harness.guardrails import ToolCallInfo


def no_prod(call: ToolCallInfo) -> GuardrailResult:
    if call.args.get('env') == 'prod':
        return GuardrailResult.block('Deploying to prod is not allowed.')
    return GuardrailResult.allow()


agent = Agent(
    TestModel(call_tools=['deploy']),
    capabilities=[ToolGuardrail(guard=no_prod, tools=['deploy'], hidden=['drop_db'])],
)


@agent.tool_plain
def deploy(env: str) -> str:
    return f'deployed to {env}'


@agent.tool_plain
def drop_db() -> str:
    return 'dropped'


print(agent.run_sync('deploy').output)
#> {"deploy":"deployed to a"}
```

Key parameters:
- `InputGuardrail(guard, parallel=False)`: `parallel=True` races the guard with the model call
  (no added latency on pass; tokens spent if it trips late; a request that raises or is retried
  first still waits for the verdict); `replace` is refused in parallel mode.
  `CodeMode` tool calls started by `eager` or `speculate` during the stream wait for its verdict.
- `OutputGuardrail(guard)`: guard receives the output object unchanged (a Pydantic model for
  structured output, not a string). Scan its JSON:
  `lambda out: redact_secrets(out.model_dump_json()).action == 'allow'` blocks on a match; to redact,
  return `GuardrailResult.replace(Model.model_validate_json(verdict.replacement))`.
- `ToolGuardrail(guard=None, result_guard=None, tools=None, hidden=())`: `tools` restricts both
  guards to those names; `hidden` drops tools from the model's tool list (no tokens; a call
  to one is rejected as an unknown tool). A `tools` name no step offered warns (`UserWarning`) after each successful
  run, including a legitimately absent tool (an integration whose `auth` function returned `None`).
  There is no switch: name only tools that are present, or omit `tools` and test `call.name` in the guard.
- `InputGuardrail.guard` / `OutputGuardrail.guard` accept a sequence (a list or tuple; sets,
  iterators, and empty sequences raise `UserError` at first use). `ToolGuardrail` guards are single
  callables; for several tool checks, stack several `ToolGuardrail` capabilities: argument guards run
  in list order, result guards in reverse.

Detectors (`pydantic_ai_harness.guardrails.detectors`), plain functions returning `GuardrailResult`
(`.action` is `'allow'`, `'block'`, or `'replace'`; `.replacement` holds the redacted text). To block
where a redactor would redact: `guard=lambda text: redact_secrets(text).action != 'replace'`.
- `redact_secrets` = `secret_data()`: vendor API keys (Anthropic, OpenAI, AWS access key id,
  GitHub, Slack, Stripe, Google), JWTs, private-key blocks. Placeholder `[redacted:{name}]`.
- `redact_personal_data` = `personal_data()`: `email`, `credit_card` (Luhn), `iban` (mod-97), `us_ssn`.
- Factories: `secret_data(only=[...], extra={'name': regex}, placeholder=...)`, same for
  `personal_data`; `blocked_keywords(keywords, case_sensitive=False, whole_words=False, message=None)`.
- Adapters: `for_text(detector, on_other='raise')` for outputs that may not be `str`;
  `for_tool_result_text(detector, on_other='raise')` for `result_guard` (scans string results and
  `ToolReturn` text; adjacent text parts are scanned joined, so a secret split across them is caught,
  and only a non-text part such as an image separates them, not a `CachePoint`).
- `for_tool_result_text` raises `UserError` on a non-text result (a structured return, `AskUser`'s
  dict answer), failing the run. Scope it with `tools=[...]`, or pass `on_other='allow'` to let
  non-text results (including `ToolReturn.content`) through unscanned.

Gotchas:
- Hard-fail instead of graceful: raise `InputBlocked`/`OutputBlocked`/`ToolBlocked(tool_name, reason)`
  (all subclass `GuardrailError`) from the guard; any exception propagates as-is.
- Input `replace` on a multimodal prompt raises `UserError` (it would drop attachments), also when
  the model was sent a text copy and the history's original is the multimodal prompt, and on a run
  resuming from a request with several prompt parts, which the guard screens combined as `ctx.prompt`.
- `InputGuardrail` screens and rewrites the user's prompt part only. Parts another capability adds to the
  request (`SystemReminders`, `Planning`, `Memory`) are left alone. If one sends the model a copy of the
  prompt part, a redaction also reaches the history's original; for a copy with changed text the guard
  is called again on the original's own text: a redaction from it is written there and a `block`
  blocks the run. A run resumed without a new prompt screens the most recent prompt in the history
  again.
- Redaction spans carry `guardrail.replacement` under `trace_include_content`, never the original value.
- `email` also matches `git@github.com`; on code agents use `personal_data(only=['us_ssn', 'credit_card', 'iban'])`.
- AWS *secret* keys are not in the defaults; add via `extra=`. Regex detectors do not catch prompt injection.
- `OutputGuardrail` sees only the final output: under `run_stream()`, chunks are already sent, and
  `retry` raises `UnexpectedModelBehavior`.
- `ToolGuardrail` never sees output tools, external/deferred tools, or provider builtin tools; only
  `hidden` applies to those.
- Approval: `approve()` needs `output_type=[..., DeferredToolRequests]` on the agent and a resume with
  `DeferredToolResults` (core flow). On resume, the guard runs again and `approve` is a no-op for cleared calls.
- Redaction/block spans carry content only when `trace_include_content` is on.

## PromptInjectionDefender

Classifies results of locally executed tools with StackOne `defender` after the tool returns.
Report-only by default; `block_high_risk=True` replaces a rejected result with a notice.

```bash
uv add "pydantic-ai-harness[prompt-injection-defender]"     # Python 3.11+
uv add "pydantic-ai-harness[prompt-injection-defender-ml]"  # adds ONNX classifier
```

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext
from stackone_defender import DefenseResult

from pydantic_ai_harness import PromptInjectionDefender


def log_detection(ctx: RunContext[None], call: ToolCallPart, verdict: DefenseResult) -> None:
    print(call.tool_name, verdict.risk_level, verdict.detections)


agent = Agent(
    'anthropic:claude-sonnet-5-5',
    capabilities=[
        PromptInjectionDefender(
            block_high_risk=True,
            tool_filter=['read_email', 'fetch_page'],
            on_detection=log_detection,
        )
    ],
)
```

Parameters: `defense` (positional; a configured `stackone_defender.PromptDefense`), then keyword-only
`block_high_risk=None` (library default: report only), `semantic_detection=False` (needs the `-ml`
extra; catches text under unrecognised fields), `tool_filter='all'` (names, `'all'`, or a
`ToolSelector`), `on_detection` (sync/async; raising fails the run), `blocked_message` (supports
`{tool_name}` and `{risk_level}` only).

Gotchas: `defense` plus `block_high_risk` or `semantic_detection` raises `UserError`; so does
`semantic_detection=True` without ONNX Runtime. Importing the module without the extra raises
`ImportError`. Not scanned: provider-native tools, externally supplied deferred results,
`ModelRetry` messages, `ToolReturn.metadata`, mapping keys, media. A withheld result carries
diagnostics in `ToolReturn.metadata['prompt_injection']` (not sent to the model). With a custom ML
`defense`, call `defense.warmup_tier2()` at startup.

## ToolCallJudge

Asks a second model one yes/no risk question after a selected tool call's arguments validate and
before its function runs. `yes` blocks the call, `no` allows it, and `unsure` follows the configured
uncertainty policy.

```python {test="skip"}
from pydantic_ai import Agent

from pydantic_ai_harness import ToolCallJudge

agent = Agent(
    'anthropic:claude-sonnet-5-5',
    capabilities=[
        ToolCallJudge(
            'anthropic:claude-haiku-5-5',
            question='Would this call destroy data, spend money, or expose secrets?',
            tools=['delete_file', 'run_shell'],
        )
    ],
)
```

Parameters: the judge `model` is positional; `question` is required; `tools='all'` accepts the
standard `ToolSelector` forms; `include_conversation=False`; `conversation_window=4_000`;
`on_uncertain='block' | 'allow' | 'ask'`; `denial_message`; `on_verdict` receives a
`ToolCallVerdict`. The verdict type is in `pydantic_ai_harness.tool_call_judge`.

Gotchas:

- The default sends only the tool name and validated arguments. `include_conversation=True` adds a
  recent transcript but widens the judge's prompt-injection surface.
- Selected calls and their arguments leave the process for the judge provider, including calls the
  judge blocks. Do not select tools whose arguments the provider must not receive.
- `on_uncertain='ask'` needs `DeferredToolRequests` in the outer agent's output type. A tool with
  `requires_approval=True` reaches the judge only after a person approves it; both gates must allow.
- Judge usage counts against the outer run's `usage` and `usage_limits`. Each selected call adds a
  model request, so scope `tools` to calls where the verdict can change what happens.
- Provider-native tools execute remotely and do not pass through this hook. This is a model-based
  filter, not a replacement for authorization, validation, least privilege, or recovery controls.
- Judgements are durable operations under Temporal, DBOS, and Prefect. Give every judge on a
  durable-capable agent a distinct explicit `id`; ordinary runs do not require one.

## ApprovalReviewer

Answers approval requests before a person sees them. For each call to a selected tool that needs
approval (`requires_approval=True` or a raised `ApprovalRequired`), a second model answers one yes/no
risk question: `no` approves the call and it runs in the same run step; `yes`, `unsure`, a model
failure, or a budget that cannot fit the review leaves the call for the person.

```python {test="skip"}
from pydantic_ai import Agent, DeferredToolRequests

from pydantic_ai_harness import ApprovalReviewer

agent = Agent(
    'anthropic:claude-sonnet-5-5',
    output_type=[str, DeferredToolRequests],
    capabilities=[
        ApprovalReviewer(
            'anthropic:claude-haiku-5-5',
            question='Could this call destroy data, spend money, or expose secrets?',
            tools=['run_shell'],
        )
    ],
)
```

Parameters: the reviewer `model` is positional; `question` is required; `tools='all'` accepts the
standard `ToolSelector` forms; `include_conversation=False`; `conversation_window=4_000`; `on_review`
receives an `ApprovalReview`. The review type is in `pydantic_ai_harness.approval_reviewer`.

Gotchas:

- It only approves. It never denies or runs a call, and calls that do not need approval are not
  reviewed. Add `ToolCallJudge` to block calls; the judge still runs on a call the reviewer approved.
- A decision model such as `'typesafe:jev-latest'` works as the reviewer model, and its confidence lands
  on `ApprovalReview.confidence`. A `no` still approves however unsure it was: to send unsure reviews to a
  language model, pass `FallbackModel('typesafe:jev-latest', <language model>, fallback_on=[ModelAPIError,
  unsure])`, where `unsure(response)` checks `response.provider_details['confidence']` against a bar.
- It is implemented with the `handle_deferred_tool_calls` hook and ordered before
  `HandleDeferredToolCalls`, so a human handler only receives what the reviewer left. Without a
  handler, the rest surface as `DeferredToolRequests` output.
- Reviews run one at a time per batch, count against the outer run's `usage` and `usage_limits`,
  and are durable operations under Temporal, DBOS, and Prefect (give the reviewer an explicit `id`).
- Reviewed arguments leave the process for the reviewer provider, including calls it leaves for the
  person. This is a model-based filter, not an authorization boundary.

## SpendLimits

Prices each model response via `ModelResponse.cost()` (genai-prices), adds it to every configured
budget window, and raises `SpendLimitExceeded` (a `UsageLimitExceeded`) before the next request once
a window is exhausted. State persists across runs on the capability's store.

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget, SpendLimitExceeded

limits = SpendLimits(budgets=[Budget(tokens=1, window='total')])
agent = Agent(TestModel(), capabilities=[limits])

agent.run_sync('first')  # starts under the ceiling, then crosses it
try:
    agent.run_sync('second')
except SpendLimitExceeded:
    print('budget exhausted')
    #> budget exhausted
```

```python {test="skip"}
from decimal import Decimal

from redis.asyncio import Redis

from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend import Budget, RedisSpendStore

limits = SpendLimits(
    budgets=[
        Budget(usd=Decimal(5), window='run'),
        Budget(usd=Decimal(100), window='day', warn_at=0.8),
        Budget(usd=Decimal(10), window='day', scope=lambda ctx: ctx.deps.tenant_id, name='tenant'),
        Budget(usd=Decimal(20), window='day', scope=lambda ctx: ctx.model.model_name, name='per-model'),
    ],
    store=RedisSpendStore(Redis.from_url('redis://localhost')),  # shared across workers
    on_unpriced='raise',
)
```

`Budget(usd=None, tokens=None, window='day', scope=None, warn_at=None, name='default', retain='window default')`;
windows: `'run' | 'conversation' | 'day' | 'month' | 'total'` (UTC). `'conversation'` keys on
`ctx.conversation_id` (`run(conversation_id=...)`, else inherited from `message_history`, else new), so
separate `agent.run` calls that pass the history share one bucket, kept 30 days by default. A budget
with no ceiling is a counter only. `scope` returns a partition key string (tenant, user, model). `SpendLimits(budgets=(),
store=InMemorySpendStore(), price=None, on_unpriced='zero', expose_tools=False, ...)`.

- Readings: `await limits.status(ctx_or_none, scope='acme')` returns `BudgetStatus` list (`spent`,
  `remaining_usd`, `warning`, `exhausted`); `await limits.exhausted(scope=...)` for an admission
  decision (raises if scoped budgets cannot be resolved; `status` leaves them out of its list).
- Per-response events: subscribe to `SpendRecordedEvent` with `@agent.on_event(...)` (`on_spend` is deprecated).
- `expose_tools=True` adds a `get_spend` tool for the model.
- `price=lambda response: Decimal(...) | None` prices models genai-prices does not know.

Gotchas:
- The guarantee is "no request **starts** after exhaustion", not "spend stays under the ceiling":
  the crossing request completes, and concurrent runs can overshoot.
- Unknown models count as $0 under `on_unpriced='zero'` (warns once with a `usd` budget); use
  `'raise'` or `price=` when callers choose the model. `TestModel` is unpriced: to test a `usd` budget
  offline, pass a fixed price such as `price=lambda response: Decimal('0.01')` (a `PriceFunc`,
  `ModelResponse -> Decimal | None`).
- Omitted usage increments `response.usage.unmeasured_requests`. When `price` is unset or returns `None`,
  any known cost subtotal still accrues, but the response remains unpriced and `on_unpriced` applies.
  Known token and cost totals are lower bounds.
- The built-in stores are `InMemorySpendStore` (default; per process and instance, lost on restart),
  `RedisSpendStore` (shared across workers; install `redis` yourself, `RedisClient` is a protocol), and
  `PostgresSpendStore(pool, table='spend')` (shared across workers; pass a caller-owned `asyncpg` pool,
  tables `{table}_counters` and `{table}_dedup` are created on first use). For another backend, implement
  `BatchSpendStore`:
  `get_many(keys) -> Mapping[str, Spent]` (unknown keys read as zero) and `add_many(entries:
  Sequence[SpendEntry]) -> Mapping[str, Spent]` (new totals; skip an entry whose `token` was already
  applied to its `key`, honour `ttl`), both async. `SpendStore` (`get`/`add`) is deprecated. Pass the
  same store object to two `SpendLimits` to share counters. `defer_loading=True` is refused.
- `SpendLimits` counts every billed response, including one a nested hook rejects, whatever the
  capability order; cached and `SkipModelRequest` responses aren't charged.
- Each response a `FallbackModel` rejected is charged as its own request, with its own
  `SpendRecordedEvent` (`failed_attempt=True`). `price` receives it as a `ModelResponse` with the
  attempt's model, provider and usage but no `parts` and no `provider_url`, so a `price` that reads
  the output or keys on the endpoint will misprice it. `Spent.requests` can exceed
  `RunUsage.requests`.
- Spec-loadable except callables (`store`, `price`, `scope`, `clock`).

## AskUser

Adds one tool, `ask_user_question`: the model sends 1-10 multiple-choice questions (unique
`header` of 25 chars or fewer, 2-6 options, optional `multi_select`); your `answerer` returns the picks. There is no
default answerer (it never reads stdin).

```python
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness import AskUser
from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserRequest, AskUserResponse


async def pick_first(request: AskUserRequest) -> AskUserResponse:
    answers = [AskUserAnswer(header=q.header, selected=(q.options[0].label,)) for q in request.questions]
    return AskUserResponse(answers=tuple(answers))


def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [p for p in messages[-1].parts if isinstance(p, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(f'User chose: {returns[0].content}')])
    question = {
        'header': 'Database',
        'question': 'Which database should I use?',
        'options': [{'label': 'Postgres'}, {'label': 'SQLite'}],
    }
    return ModelResponse(parts=[ToolCallPart('ask_user_question', {'questions': [question]})])


agent = Agent(FunctionModel(model), capabilities=[AskUser(answerer=pick_first)])
print(agent.run_sync('Set up storage').output)
#> User chose: {'Database': ['Postgres']}
```

- Answerer: any `async (AskUserRequest) -> AskUserResponse`. `request.id` matches replies in a UI.
- Decline: return `AskUserResponse(cancelled=True)`; the model gets a "user declined" tool result.
- Free text: `AskUserAnswer(header=..., custom_answer='...')` with `selected` empty.
- An answer that does not fit (unknown header/label, several picks on single-select, missing
  answer) raises `ValueError` and fails the run; call `check_response(request, response)` (from
  `pydantic_ai_harness.ask_user`) first.
- The run waits inside the tool call. `timeout=<seconds>` cancels a slow answerer; the model gets
  `TIMED_OUT` and `AskUserAnsweredEvent` fires with `timed_out=True`. A thread cannot be cancelled:
  wrap blocking work in `anyio.to_thread.run_sync(..., abandon_on_cancel=True)`, and read terminal
  input with an async prompt library, not `input()` in a thread.
- By default, parallel `ask_user_question` calls reach the answerer concurrently; serialize in the
  answerer if the UI shows one request at a time. Custom text has no length cap, so bound client input.
- `answerer=None` defers every call: the run ends with `DeferredToolRequests` (add it to
  `output_type`). Rebuild each request with `AskUserRequest.from_tool_call(call)`, then resume with
  `DeferredToolResults(calls={call.tool_call_id: ask_user_result(request, response)})`.
- Observe with `AskUserRequestedEvent` / `AskUserAnsweredEvent`. An answerer that raises still
  emits the answered event, with `failed=True`, before its error propagates. Two `AskUser` on one agent
  collide on the tool name. It has a spec name, but a spec cannot supply the `answerer` callable,
  so pass it with `capabilities=` on `Agent.from_spec`/`Agent.from_file`.

## SystemReminders

Enqueues reminders at the tail of each model request as a request-scoped `SystemPromptPart`
(`scope='request'`): the model sees each one only in the model request it's part of, and the next
request, even the next step of the same run, no longer shows it. It's kept in message history, and
it sits after the cached prefix, so it doesn't invalidate it. Turn on caching on the model
(e.g. `anthropic_cache`); its breakpoint lands before the reminder. `cache_ttl` is deprecated and
has no effect. With `interval>1` the model sees a reminder only on the requests where it fires, so
text it must follow at every step belongs in the agent's instructions, not in a `Reminder`.

```python
from pydantic_ai import Agent

from pydantic_ai_harness import SystemReminders
from pydantic_ai_harness.system_reminders import GoalReanchor, Reminder

agent = Agent(
    'test',
    capabilities=[
        SystemReminders(
            reminders=[
                Reminder('Stay focused on the original request.', interval=5, tag=None),
                Reminder('Run the tests before finishing.', interval=10, max_fires=2, tag=None),
            ],
            dynamic_reminders=[
                GoalReanchor(),
                lambda ctx: 'Wrap up soon.' if ctx.run_step > 20 else None,
            ],
        )
    ],
)
```

- `Reminder(content, interval=1, first_after=None, trigger=None, max_fires=None, tag=None)`:
  requests are counted from 1 per run, so `interval=4` fires on requests 4, 8, 12; `first_after=N`
  moves the first fire to request N, then every `interval` after (`first_after=1, interval=4` fires
  on 1, 5, 9). `trigger(ctx) -> bool` adds a condition. Pass `tag=None`: left unset it warns that the
  default used to be `'system-reminder'`. A reminder needs no tag, since it's already sent as a
  system prompt; `tag='name'` wraps the text in `<name>`.
- `dynamic_reminders`: callables `(ctx) -> str | GeneratedReminder | None`, sync or async, evaluated
  on **every** request. A `str` is sent as written (no tag); a `GeneratedReminder` is quoted as data. Reminders are system prompts, so return only text you wrote and
  quote user input or tool output as data.
- `GoalReanchor()` points the model back at the run's first user prompt without repeating it (at the
  compaction summary once native compaction took the prompt); free. `LLMReminder` returns a
  `GeneratedReminder`, not a `str`, so it stays data: a fixed note in system voice and the generated
  text on the user channel (`<generated_reminder model="...">`), kept in history together as one record.
  A plain `str` a dynamic reminder returns is operator text in system voice. Framing tags in quoted
  text are escaped.
- `LLMReminder(model=..., interval=10)` (model required) makes one extra model call every `interval`
  requests (10th, 20th, ...; pass `interval` explicitly: left unset it warns that the default used to
  be `1`). Each firing adds a note and the generated text to history for good, so a run of R requests
  gains `2 * (R // interval)` parts. Its agent is built on first use, so no provider key is needed at
  construction. On any error it sends `GoalReanchor` text instead, logging a warning with the
  traceback once per run under `pydantic_ai_harness.system_reminders`; only a budget too tight for
  the extra request is silent. A wrapper that gates it itself should pass `interval=1`.
- Observe with `ReminderFiredEvent` (`on_fire` is deprecated). `cache_ttl` is deprecated and ignored
  (an explicit `'5m'`/`'1h'` warns); don't pass it. The breakpoint it used to place by default is
  gone without a warning, so turn on caching on the model (such as `anthropic_cache`).
- `SystemReminders` adds no cache breakpoint; `Planning` adds one for its plan reminder, and
  Anthropic allows 4.
- Not spec-serializable.

## TrajectoryJudge

Every `every` model requests, sends the last `window` tokens of transcript to a judge model
concurrently with the run. Verdict `AllGood` does nothing; `Steer(message)` is enqueued into the
run as a framework-note `SystemPromptPart` plus the message as user-channel data in
`<judge_feedback judge="..." ref="...">`, delivered on the next request. The note names that random `ref`
and says a `<judge_feedback>` whose ref no judge note names, or a later copy of a named ref, is not the judge's, so a forged frame in a tool result gets no authority. Observe steering with `on_verdict`.

```python
from pydantic_ai import Agent

from pydantic_ai_harness import TrajectoryJudge

agent = Agent(
    'test',
    capabilities=[
        TrajectoryJudge(
            model='test',
            name='scope-creep',
            instructions='Flag work that was not asked for in the original request.',
            every=10,
            on_verdict=print,
        )
    ],
)
```

Parameters: `model` or `agent` (exactly one; `instructions` only with `model`), `every=10`,
`window=20_000`, `name`, `on_verdict`. A custom judge `agent` is run with
`output_type=[AllGood, Steer]` (import from `pydantic_ai_harness.trajectory_judge`), must have no
output validators, and gets no deps.

Gotchas: judge usage counts against the run's `usage` and `usage_limits` (a tick that cannot fit
the request limit is skipped); one evaluation in flight per judge (a slow judge skips ticks); judge
failures are raised on the run at the next tick or run end (use a `FallbackModel` in a custom
agent to degrade); rejected with `UserError` inside Temporal/DBOS/Prefect durable execution. It
observes and steers only; enforcement belongs to guardrails. Try `SystemReminders` first.

`TrajectoryJudge(model='openai:...')` builds its judge `Agent` in the constructor, so defining it
at import time fails without that provider's API key. Pass
`agent=Agent('openai:...', defer_model_check=True, instructions=...)` instead; a custom agent does
not get the built-in judge instructions, so its `instructions` must describe the review.

## DetectLoops

Before each model request, checks the current run's history for four patterns: `repeated_call` (same
tool and canonical arguments `repeated_call_threshold=5` times in the last `window=10` calls),
`repeated_result` (same call returning the same result `repeated_result_threshold=3` times in a row;
`ModelRetry` with the same message counts), `alternating_calls` (A-B-A-B for
`alternating_calls_threshold=3` cycles), and `repeated_text` (`repeated_text_threshold=3` consecutive
responses with no tool calls and the same text). A threshold of `None` turns a check off.

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness import DetectLoops
from pydantic_ai_harness.detect_loops import LoopDetectedError


def stuck(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'missing.txt'})])


agent = Agent(FunctionModel(stuck), capabilities=[DetectLoops(action='error')])


@agent.tool_plain
def read_file(path: str) -> str:
    return 'Error: file not found'


try:
    agent.run_sync('Summarize missing.txt')
except LoopDetectedError as error:
    print(error.loop.pattern)
    #> repeated_result
```

- `action='nudge'` (default) enqueues a `SystemPromptPart` (conversation scope, so it keeps
  applying) into the current request; find it by `metadata['origin'] == 'pydantic-ai-harness.detect_loops.nudge.v1'`, never by
  text. `'error'` raises `LoopDetectedError` (an `AgentRunError`, `.loop` is the `DetectedLoop`). A
  sync or async `(ctx, loop)` callable decides itself; type `ctx` as `RunContext[object]` on an agent
  without `deps_type`.
- Calls, results, and text are recorded as they arrive (the capability runs outermost), so
  compaction can't make calls look alike.
  One loop per request; its responses are then not counted again. Only the current run counts, so
  `message_history` from earlier runs starts at zero. Hooks run in workflow code under Temporal,
  DBOS, and Prefect, so replay reaches the same decisions; keep a handler free of I/O.
- Agents that poll on purpose trip `repeated_result` after 3 identical statuses: raise or disable it.
- Emits a `loop detected` span (`detect_loops.pattern`, `.action`, `.count`, `.window`,
  `.tool_names`; `.fingerprints` only with `trace_include_content`); no arguments or results.

## See also

- https://pydantic.dev/docs/ai/harness/repair-tool-arguments/
- https://pydantic.dev/docs/ai/harness/guardrails/
- https://pydantic.dev/docs/ai/harness/prompt-injection-defender/
- https://pydantic.dev/docs/ai/harness/tool-call-judge/
- https://pydantic.dev/docs/ai/harness/approval-reviewer/
- https://pydantic.dev/docs/ai/harness/spend/
- https://pydantic.dev/docs/ai/harness/ask-user/
- https://pydantic.dev/docs/ai/harness/system-reminders/
- https://pydantic.dev/docs/ai/harness/trajectory-judge/
- https://pydantic.dev/docs/ai/harness/detect-loops/
- https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/ (core tool approval and deferred tools)
