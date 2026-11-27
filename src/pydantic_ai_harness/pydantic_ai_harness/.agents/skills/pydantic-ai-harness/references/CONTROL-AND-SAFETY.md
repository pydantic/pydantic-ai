# Control and Safety

Capabilities that constrain, screen, budget, or steer a run: repair malformed tool arguments,
guard prompts/tool calls/outputs, scan tool results for prompt injection, cap spend, ask the user
a question mid-run, re-inject reminders, interrupt repeated-action loops, review the trajectory with a
judge model, and have the agent revise its answer before the run ends. All are
`AbstractCapability`s passed in `Agent(capabilities=[...])`. Only `RepairToolArguments` (`[repair-tool-arguments]`) and
`PromptInjectionDefender` need an extra. For plain tool approval (`requires_approval=True`, `ApprovalRequired`,
`DeferredToolRequests`, `HandleDeferredToolCalls`), use the core mechanism documented in the
`building-pydantic-ai-agents` skill (`references/TOOLS-ADVANCED.md`); do not rebuild it here.

## Choose a capability

| I want to ... | Use |
|---|---|
| Stop tool calls failing on trailing commas, single quotes, truncated JSON | `RepairToolArguments()` |
| Refuse or redact a prompt before it reaches the model | `InputGuardrail(guard=...)` |
| Block, redact, or retry the final output | `OutputGuardrail(guard=...)` |
| Keep an unattended run going until a completion check you define accepts the output | `Goal(goal=..., verify=...)` |
| Screen each streamed text, thinking or tool-call part before the caller sees it | `StreamGuardrail(guard=...)` |
| Block/rewrite tool arguments, redact tool results, or hide tools | `ToolGuardrail(guard=..., result_guard=..., hidden=[...])` |
| Mask API keys, tokens, PII in text | `guardrails.detectors.redact_secrets` / `redact_personal_data` in a guard |
| Get human approval for some tool calls, decided per call | `ToolGuardrail` returning `GuardrailResult.approve()` |
| Get human approval for a whole tool | core `requires_approval=True` (core skill) |
| Detect indirect prompt injection in emails/web pages returned by tools | `PromptInjectionDefender` |
| Keep injected instructions, exfiltration commands, hidden text out of memory and skills | `Memory(scanner=ArtifactScanner())`, `Skills(..., scanner=ArtifactScanner())` |
| Ask a second model whether each selected tool call should run | `ToolCallJudge(model, question=..., tools=...)` |
| Let a second model approve low-risk calls that need approval, and ask a person about the rest | `ApprovalReviewer(model, question=..., tools=...)` |
| Ask a person before file writes and commands, by permission mode (full, auto, read-only) | `Permissions(mode)` with core `HandleDeferredToolCalls(approval_ask(approver))` |
| Allow, ask about, or deny tool calls and shell commands by rule | `Permissions(rules=[PermissionRule(...)])` |
| Cap USD or tokens per day/month/run, per tenant, across workers | `SpendLimits(budgets=[Budget(...)])` |
| Cap tokens/cost for one run only | core `UsageLimits` (no harness capability needed) |
| Tell the model how much of its usage limits and spend budgets is left, every request | `ReportRemainingBudget(spend_limits=...)` |
| Let the model ask the user multiple-choice questions and wait | `AskUser(answerer=...)` |
| Counter instruction fade in long runs, cache-safe | `SystemReminders` |
| Have a second model review the run and steer it when it drifts | `TrajectoryJudge` |
| Interrupt an agent that keeps repeating the same tool call, result, or response, or repeats itself within a streaming response | `DetectLoops` |
| Cap calls per tool or per run, or refuse exact repeats, and let the run continue | `ToolBudgets(per_tool=..., total=..., refuse_repeats=...)` |
| Have the agent review and revise its answer before the run ends | `RefineOutput` |
| Send an exception a tool raises to the model instead of ending the run | `ReportToolErrors(format_error=...)` |
| Send a coding agent back to run its checks before it finishes after an edit | `RequireVerification` |
| End the run with a final answer instead of `UsageLimitExceeded` at a usage limit | `WrapUpAtLimits` |

## RepairToolArguments

Runs `json-repair` on raw tool-call argument strings in `before_tool_validate`, for every tool on the
agent. Needs the `repair-tool-arguments` extra, which installs `json-repair`
(`uv add "pydantic-ai-harness[repair-tool-arguments]"`; `[coder]` includes it); no options.
A repair is written back to the `ToolCallPart` in history (original args in
`provider_details['original_args']`), so later requests replay the repaired arguments.

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

`InputGuardrail`, `OutputGuardrail`, `StreamGuardrail`, `ToolGuardrail`, `GuardrailResult`, `InputBlocked`,
`OutputBlocked`, `GuardrailError` are top-level (`from pydantic_ai_harness import ...`);
`ToolBlocked`, `ToolCallInfo`, `ToolResultInfo`, `StreamGuardrailPart`, `detectors` live in `pydantic_ai_harness.guardrails`.

A guard is a sync or async callable taking the value, optionally preceded by `RunContext` (taken
when a second positional parameter is required, for `*args` with another parameter, or when the
second has a default and the first is annotated `RunContext`; a first parameter with no annotation,
`Any`, `object`, or an unresolvable type not naming `RunContext` still gets it, with a
`HarnessDeprecationWarning`). It
returns `bool` (`True` = allow) or a `GuardrailResult`: `allow()`, `block(message=None)`,
`replace(value)`, `retry(message)`, `approve()`. Any other return value (a `'safe'` label, `None`, a
`numpy.bool_`) is read for truthiness with a `HarnessDeprecationWarning` (a truthy label allows the
value); return `bool(...)` instead.

| Verdict | `InputGuardrail` | `OutputGuardrail` | `StreamGuardrail` | `ToolGuardrail.guard` (args) | `ToolGuardrail.result_guard` |
|---|---|---|---|---|---|
| block | skip the model call; message becomes the response | raise `OutputBlocked` | raise `OutputBlocked`, part never released | skip tool; message is the tool result | message replaces the result |
| replace | rewrite the prompt (text prompts, sequential only) | substitute output | stream a same-type text or thinking part instead (events only); `UserError` for a tool call | run with new args mapping | substitute result |
| retry | `UserError` | `ModelRetry` to the model (counts output retries) | `UserError` | `ModelRetry` | `ModelRetry` (tool already ran) |
| approve | `UserError` | `UserError` | `UserError` | defer via `ApprovalRequired` | not valid |

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
  `lambda out: redact_secrets(pydantic_core.to_json(out).decode()).action == 'allow'` blocks on a
  match (and also works on the JSON data a Temporal worker passes); to redact,
  return `GuardrailResult.replace(Model.model_validate_json(verdict.replacement))`.
- `StreamGuardrail(guard)`: holds each streamed `TextPart`, `ThinkingPart` or `ToolCallPart` until it
  is complete and passes the whole part to the guard before its events reach `run_stream_events()`,
  a run's `event_stream_handler` argument or a UI adapter; `ProcessEventStream` and `on_event`
  see raw events. Adds the part's generation time to its first chunk; a tool call is held until
  the model response ends.
  `run_stream()`'s `stream_text()`/`stream_output()` are not screened (a `block` still raises), and
  the run's history and output are not changed; pair with `OutputGuardrail`/`ToolGuardrail` for
  those. Makes `run()` stream every request; raises `UserError` under durable execution and in a
  realtime session.
- `ToolGuardrail(guard=None, result_guard=None, tools=None, hidden=())`: `tools` restricts both
  guards to those names; `hidden` drops tools from the model's tool list (no tokens; a call
  to one is rejected as an unknown tool). A `tools` name no step offered warns (`UserWarning`) after each successful
  run, including a legitimately absent tool (an integration whose `auth` function returned `None`).
  There is no switch: name only tools that are present, or omit `tools` and test `call.name` in the guard.
- `InputGuardrail.guard` / `OutputGuardrail.guard` accept a sequence (a list or tuple; sets,
  iterators, and empty sequences raise `UserError` at first use). `ToolGuardrail` guards are single
  callables; for several tool checks, stack several `ToolGuardrail` capabilities: argument guards run
  in list order, result guards in reverse.
- Guard verdicts are durable operations under Temporal, DBOS, and Prefect. Give every guardrail on a
  durable-capable agent a distinct explicit `id`; ordinary runs do not require one. Under Temporal
  a structured output or tool result reaches the guard as JSON data, its `RunContext` has no
  `messages` or `prompt`, and an exception it raises other than `InputBlocked`, `OutputBlocked`, `ToolBlocked`,
  `ApprovalRequired` or `ModelRetry` fails (and retries) the activity instead of reaching the run.

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
`{tool_name}` and `{risk_level}` only), `id` (none by default; required on durable agents).

Gotchas: `defense` plus `block_high_risk` or `semantic_detection` raises `UserError`; so does
`semantic_detection=True` without ONNX Runtime. Importing the module without the extra raises
`ImportError`. Not scanned: provider-native tools, externally supplied deferred results,
`ModelRetry` messages, `ToolReturn.metadata`, mapping keys, media. A withheld result carries
diagnostics in `ToolReturn.metadata['prompt_injection']` (not sent to the model). Every flagged verdict
emits a zero-duration `prompt_injection flagged` span (`prompt_injection.action`, `.risk_level`,
`.detections`; sanitized field names only with content tracing). With a custom ML
`defense`, call `defense.warmup_tier2()` at startup (on the worker, under durable execution).
Under Temporal, DBOS, and Prefect, semantic warmup and classification with `semantic_detection=True`
or a supplied `defense` are durable operations. Give every defender on a durable-capable agent a
distinct explicit `id`, or the agent build raises `UserError`. Ordinary runs do not require one.

## ArtifactScanner

Pattern scanner for agent-written artifacts that later sessions load: memory files and skills. Not a
capability; pass it to `Memory(scanner=...)` and `Skills(scanner=...)`, or call `scan()` directly.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace

from pydantic_ai_harness import ArtifactScanner, Memory, Skills

scanner = ArtifactScanner()  # threats=[...] narrows; extra_patterns={'name': regex} adds 'custom' ones
agent = Agent(
    'anthropic:claude-sonnet-5-5',
    capabilities=[LocalWorkspace('.'), Memory(scanner=scanner), Skills('.agents/skills', scanner=scanner)],
)
result = scanner.scan('Ignore previous instructions', source='notes.md')
result.flagged, result.summary()  # True, 'injection (ignore_instructions)'
```

Threats: `injection`, `role_override`, `exfiltration`, `html_comment`, `invisible_unicode` (plus
`custom` for `extra_patterns`). With a scanner: `write_memory` refuses flagged text with a
`ToolFailed`-style result (nothing stored); a flagged `MEMORY.md` is replaced by a notice in the
snapshot; `read_memory` refuses a flagged file; flagged `search_memory` snippets become a notice; a
flagged skill is skipped with a `UserWarning` and `SkippedSkill(reason='flagged', scan=...)`.
Each decision emits an `artifact_scan flagged` span (source and excerpts only with `trace_include_content`).

Gotchas: regex only, so paraphrases slip through and quoted attacks false-positive. `html_comment`
flags linter directives in skills; narrow `threats` there. Memory files read via `FileSystem` and
bundled skill files read other than through `Skills(read_files=True)` are not scanned. Subclass and override `scan` to plug in another detector.

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

## Permissions

Holds the `FileSystem` writes and `Shell` commands a permission mode asks about by raising
`ApprovalRequired`: `'full'` (default) asks about nothing, `'auto'` asks about commands matching
`ask_commands` (`DEFAULT_ASK_COMMANDS`: `rm -rf`, `git push`, `sudo`, network clients, installs), and
`'read-only'` asks about every write and command. Other tools are not checked unless a rule matches them.

`rules=[PermissionRule(...)]` decide before the mode: each rule is `'allow'`, `'ask'`, or `'deny'` for calls
matching its `tools` selector, `args` globs, and shell `command` words. `'deny'` beats `'ask'` beats `'allow'`,
whatever the order. A compound command runs by rule only if an allow rule matches every simple command in it, while
an ask or deny rule matching any one decides it. An allow skips `'read-only'`'s blanket ask but not `ask_commands`.
A denied call returns a `ToolReturnPart` with `outcome='denied'`, even when it was approved.

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.capabilities import HandleDeferredToolCalls, LocalWorkspace

from pydantic_ai_harness import Coder
from pydantic_ai_harness.permissions import (
    PermissionDecision,
    PermissionRequest,
    Permissions,
    approval_ask,
)


async def approve(request: PermissionRequest) -> PermissionDecision:
    return PermissionDecision(approved=input(f'{request.subject}? [y/N] ') == 'y')


agent = Agent(
    'anthropic:claude-sonnet-5-5',
    capabilities=[
        LocalWorkspace('.'),
        Coder(),
        Permissions('auto'),
        HandleDeferredToolCalls(approval_ask(approve)),
    ],
)
```

Gotchas:

- Without a handler a held call ends the run as `DeferredToolRequests` output, like `requires_approval=True`.
- `approval_ask` takes a resume token, which durable runs on engines other than AWS Lambda (`AWSLambdaDurability`)
  refuse; there, pass the two-parameter handler `lambda ctx, requests: ask_permissions(requests, approve)`. On
  Lambda, `approval_ask` answers in place in a recorded step: keep the approver quick and set `timeout=...`.
- A plain two-parameter handler runs before a token handler such as `approval_ask`, whatever the list order: a
  second handler for other approvals must skip the calls `permission_requests(requests)` returns.
- `mode` may be a function of the run context, read on every call. `PermissionDecision(remember=True)`
  skips asking about the same tool and command or path for the rest of the conversation, via the caller-ownable
  `remembered` set; inside durable execution it is kept in the message history instead.
- Delegates are only checked when `Permissions` reaches them: pass it in `SubAgents(shared_capabilities=[...])`.
- `ask_commands` patterns and `rules` are a convenience, not a sandbox: use a sandbox workspace when the agent
  must not reach something. An allow rule trusts everything after its words (`find` allows `find . -delete`).
- Commands with `$`, backticks, quoted operators, brace expansion, globbed command names, subshells,
  here-documents, or line breaks, and every command on a `'windows'` workspace, cannot be split: no `command` rule
  allows them. Their quote-stripped text is still checked against ask and deny rules, and when the shell expands
  part of them (`$`, backticks, braces, line continuations, globbed command names) they ask if an ask or deny
  command rule applies.

## SpendLimits

Prices each model response via `ModelResponse.cost()` (genai-prices), adds it to every configured
budget window, and raises `SpendLimitExceeded` (a `UsageLimitExceeded`) before the next request once
a window is exhausted. State persists across runs on the capability's store.

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai_harness import SpendLimits
from pydantic_ai_harness.spend_limits import Budget, SpendLimitExceeded

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
from pydantic_ai_harness.spend_limits import Budget, RedisSpendStore

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
with no ceiling is a counter only. `scope` returns a partition key string (tenant, user, model). `usd` and
`tokens` also take a callable of the run context, so each partition gets its own ceiling
(`usd=lambda ctx: ctx.deps.allowance_usd`; `None` means no ceiling); `status()` without `ctx` omits such a
budget, and a spec cannot express one. `SpendLimits(budgets=(),
store=InMemorySpendStore(), price=None, on_unpriced='zero', expose_tools=False, ...)`.

- Readings: `await limits.status(ctx_or_none, scope='acme')` returns `BudgetStatus` list (`spent`,
  `usd_limit`/`token_limit` holding the resolved ceilings, `remaining_usd`, `warning`, `exhausted`;
  read those rather than `status.budget.usd`, which holds the callable itself when the ceiling is
  given as a callable); `await limits.exhausted(scope=...)` for an admission decision (raises if a
  scoped, run-windowed, or callable-ceiling budget cannot be resolved; `status` leaves them out of
  its list).
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
- Each response a `FallbackModel` or the `ModelFallback` capability rejected is charged as its own request,
  with its own `SpendRecordedEvent` (`failed_attempt=True`). `Spent.requests` can exceed `RunUsage.requests`.
- For a `FallbackModel` rejection, `price` receives a `ModelResponse` rebuilt from the attempt's model,
  provider and usage, with no `parts` and no `provider_url`, so a `price` that reads the output or keys
  on the endpoint will misprice it. A `ModelFallback` capability rejection reaches `price` as the real response.
- Spec-loadable except callables (`store`, `price`, `scope`, `clock`, callable `usd`/`tokens`).

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
- Two required positional parameters receive `(ctx, request)` (`ContextAnswerer[DepsT]`), including lambdas.
  If both call shapes bind, a first parameter annotated `RunContext[DepsT]` selects the context form; otherwise request-only.
  Only the first annotation is inspected; unresolved `RunContext` references are recognized.
  Read the asking run's dependencies and messages through `ctx`, not the request.
  A delegated question receives the child's context. For durable field access, follow
  https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#agent-run-context-and-dependencies.
- Type a stored callback as `AnyAnswerer[DepsT]` when either signature is possible.
- Decline: return `AskUserResponse(cancelled=True)`; the model gets a "user declined" tool result.
- Free text: `AskUserAnswer(header=..., custom_answer='...')` with `selected` empty.
- Answer notes: `AskUserAnswer.note` accompanies selected labels or a custom answer with text and attachments.
  Notes reach the user channel in `answer_note` frames and remain user turns during compaction.
  See [Adding notes to an answer](https://pydantic.dev/docs/ai/harness/ask-user/#adding-notes-to-an-answer)
  for supported content, validation and provenance; annotate tool results with `AskUserResult` from `pydantic_ai_harness.ask_user`.
- An answer that does not fit (unknown header/label, several picks on single-select, missing
  answer) raises `ValueError` and fails the run; call `check_response(request, response)` (from
  `pydantic_ai_harness.ask_user`) first.
- Each question is a deferred call marked `kind='question'` (`requests.kind(id) == 'question'`); `AskUser`
  answers its own in `handle_deferred_tool_calls`, after the step's other tool calls finish, one question
  at a time, through core `wait_for_external_event` like a token-taking `HandleDeferredToolCalls` handler: in process by default, journaled
  and suspended on AWS Lambda durable functions and on Temporal (`TemporalDurability`: the answerer runs in
  a local activity whose result the history records, the workflow waits on a condition, and the token is
  resolved with `await pydantic_ai.durable_exec.temporal.resolve_external_event(client, token, payload)`,
  which signals the workflow; the tool runs in an activity, which can't emit events yet (#7971), so
  `AskUserRequestedEvent` isn't emitted there; import the agent's module under
  `workflow.unsafe.imports_passed_through()`, or the answerer is the sandbox's copy); on DBOS/Prefect the engine refuses it with `UserError`
  (an unrecorded wait could diverge from the journal on replay). Works under `run_stream()` too: the
  tool body defers, so streaming doesn't stop on the call.
- Three required positional parameters receive `(ctx, request, token)` (`TokenAnswerer[DepsT]`): return the
  response, or hand `token` on and return `None`; the answer then arrives when the token is resolved
  with the JSON of an `AskUserResponse` (`resolve_external_event` in process, the engine's API durably).
  In a durable run the answerer runs in a durable unit, so it must return quickly.
- `timeout=<seconds>` bounds the wait (and, in process, cancels a slow async answerer); the model gets
  `TIMED_OUT` and `AskUserAnsweredEvent` fires with `timed_out=True`, as it does when the token is failed.
  A thread cannot be cancelled: wrap blocking work in `anyio.to_thread.run_sync(..., abandon_on_cancel=True)`,
  and read terminal input with an async prompt library, not `input()` in a thread.
- Custom text has no length cap, so bound client input.
- With a `HandleDeferredToolCalls` on the same agent, `AskUser` answers its questions first; with
  `answerer=None`, the handler gets the question and answers it with `ask_user_result` in `calls`.
- `answerer=None` defers every call: the run ends with `DeferredToolRequests` (add it to
  `output_type`). Rebuild each request with `AskUserRequest.from_tool_call(call)`, then resume with
  `DeferredToolResults(calls={call.tool_call_id: ask_user_result(request, response)})`.
- Observe with `AskUserRequestedEvent` / `AskUserAnsweredEvent`. An answerer that raises still
  emits the answered event, with `failed=True`, before its error propagates. Two `AskUser` on one agent
  collide on the tool name. A spec can set `answerer` only to `null`, which defers every call as above;
  pass an `AskUser` with an answerer callable with `capabilities=` on `Agent.from_spec`/`Agent.from_file`.

## Deny Unanswered

`DenyUnanswered` ends the wait in an unattended run: a held approval is denied with
`message` (`ToolDenied`) and a deferred `AskUser` question is answered with
`question_message` (nobody is available to answer, not "the user declined"),
so the run finishes instead of pausing on `DeferredToolRequests`. Any other deferred call
stays deferred for its own handler.

Pairs with `AskUser` for unattended runs, where deferred calls exist for a caller who
answers them in a second run and nobody ever does. `message` and `question_message` change
the two texts.

`SubAgents` delegations and `DynamicWorkflow` child runs carry the parent run's
`DenyUnanswered` into the child, so a delegate's question or approval request is denied
inside the child too and the delegation finishes. This per-capability forwarding is interim:
a general hook for passing capabilities on to child runs is being designed in #13135. A
delegation a durable engine runs as a tool activity (a named sub-agent under
`TemporalDurability`) and a remote sub-agent run by a `SubAgentWorker` get no forwarded
instance: give that child, or the worker's agent, its own `DenyUnanswered`.

## SystemReminders

By default, enqueues reminders at the tail of each model request as a request-scoped `SystemPromptPart`
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

- `Reminder(content, interval=1, first_after=None, trigger=None, max_fires=None, tag=None, scope='request')`:
  requests are counted from 1 per run, so `interval=4` fires on requests 4, 8, 12; `first_after=N`
  moves the first fire to request N, then every `interval` after (`first_after=1, interval=4` fires
  on 1, 5, 9). `trigger(ctx) -> bool` adds a condition. Pass `tag=None`: left unset it warns that the
  default used to be `'system-reminder'`. A reminder needs no tag, since it's already sent as a
  system prompt; `tag='name'` wraps the text in `<name>`. `scope='conversation'` records each firing
  as a lasting `SystemPromptPart` re-sent on every later request (one more part per firing, so bound it
  with `interval`/`max_fires`); use it for guidance that must outlast its step, instructions for text
  that holds on every step, and the default `'request'` for per-step nudges or changing values.
  `skip_unchanged=True` (requires `scope='conversation'`) skips a firing while a copy of the same text
  is still in the history the model sees, across runs too, and records it again once compaction drops it.
  List any `ProcessHistory`/compaction that drops messages before `SystemReminders`, or a trimmed
  copy still suppresses the firing.
- `dynamic_reminders`: callables `(ctx) -> str | GeneratedReminder | None`, sync or async, evaluated
  on **every** request. A `str` is sent as written (no tag); a `GeneratedReminder` is quoted as data. Reminders are system prompts, so return only text you wrote and
  quote user input or tool output as data: a `str` that looks like a framework frame raises `UserError` and fails the run.
- `GoalReanchor()` points the model back at the run's first user prompt without repeating it (at the
  compaction summary once native compaction took the prompt); free. `LLMReminder` returns a
  `GeneratedReminder`, not a `str`, so it stays data: a `GeneratedReminderPart` (from
  `pydantic_ai_harness.system_reminders`), the generated text on the user channel
  (`<generated_reminder model="...">`) behind a fixed note in the application's voice that scopes it to
  the user's task (no actions, such as tool calls, the user didn't ask for) and names the reminder model
  as its author, kept in history as one record. Text in it that looks like a framework
  frame gets a further note naming its author. A plain `str` a dynamic reminder returns is operator
  text in system voice; static `Reminder` content that looks like a framework frame, such as `<system>`,
  raises `UserError` when the `Reminder` is created. Framing tags in the transcript `LLMReminder`
  quotes to its model are escaped.
- `LLMReminder(model=..., interval=10)` (model required) makes one extra model call every `interval`
  requests (10th, 20th, ...; pass `interval` explicitly: left unset it warns that the default used to
  be `1`). Each firing adds one `GeneratedReminderPart` to history for good, so a run of R requests
  gains `R // interval` parts. Its agent is built on first use, so no provider key is needed at
  construction. On any error it sends `GoalReanchor` text instead, logging a warning with the
  traceback once per run under `pydantic_ai_harness.system_reminders`; only a budget too tight for
  the extra request is silent. A wrapper that gates it itself should pass `interval=1`.
- `LLMReminder` vs `TrajectoryJudge`: `LLMReminder` adds a model-written goal refresher on every
  firing, blocks the request it precedes, and falls back to `GoalReanchor` text on error;
  `TrajectoryJudge` runs alongside the agent (outside durable execution), steers only when the
  run drifts, and raises an evaluation error on the run. Use `LLMReminder` for a regular goal
  refresher, the judge for course correction, and `GoalReanchor` for free re-anchoring.
- Observe with `ReminderFiredEvent` (`on_fire` is deprecated). `cache_ttl` is deprecated and ignored
  (an explicit `'5m'`/`'1h'` warns); don't pass it. The breakpoint it used to place by default is
  gone without a warning, so turn on caching on the model (such as `anthropic_cache`).
- Neither `SystemReminders` nor `Planning` adds a cache breakpoint; the model's caching settings
  place them.
- Not spec-serializable.

## TrajectoryJudge

Every `every` model requests, sends the last `window` tokens of transcript to a judge model
concurrently with the run. Verdict `AllGood` does nothing; `Steer(message)` is enqueued into the
run as a framework-note `SystemPromptPart` plus the message as user-channel data in
`<judge_feedback judge="..." ref="...">`, delivered on the next request. The feedback carries its own note
telling the model not to call a tool it asks for unless the user asked for that. The framework note names that random `ref`
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
failures are raised on the run at the next tick or run end (give a custom agent the `ModelFallback`
capability to degrade); rejected with `UserError` inside Temporal/DBOS/Prefect durable execution. It
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

- `action='nudge'` (default) enqueues a `LoopNudgePart` (an `InstructionCapabilityPart` from
  `pydantic_ai_harness.detect_loops`, conversation scope, so it keeps applying) into the current request; find it
  by `isinstance(part, LoopNudgePart)`, never by text. Its `pattern` and `tool_names` fields never reach the model. `'error'` raises `LoopDetectedError` (an `AgentRunError`, `.loop` is the `DetectedLoop`). A
  sync or async `(ctx, loop)` callable decides itself; type `ctx` as `RunContext[object]` on an agent
  without `deps_type`.
- Calls, results, and text are recorded as they arrive (the capability runs outermost), so
  compaction can't make calls look alike.
  One loop per request; its responses are then not counted again. Only the current run counts, so
  `message_history` from earlier runs starts at zero. Hooks run in workflow code under Temporal,
  DBOS, and Prefect, so replay reaches the same decisions; keep a handler free of I/O.
- Agents that poll on purpose trip `repeated_result` after 3 identical statuses: raise or disable it.
- Also checks each response on its own, as it streams: `stream_detector` (default
  `RepetitionDetector()`: a passage of `min_period=20` to `max_period=2000` chars with
  `min_distinct_chars=8`, repeated `min_repeats=8` times covering `min_chars=2000`, so grids, table
  rules and a few dozen identical rows don't count) runs every `stream_check_every=256` chars and at
  each part's end over `stream_channels=('thinking', 'text', 'tool_args')`, giving
  `repeated_thinking`, `repeated_stream_text`, `repeated_tool_args`; `repeated_parallel_call` is the
  same call `repeated_parallel_call_threshold=5` times in one response, reported before any runs. A
  response that isn't streamed gets the same checks on arrival; `DetectLoops` never makes a run
  stream.
- Within a response, `'nudge'` also stops the stream and retries via `ModelRetry`, with the nudge
  request-scoped (counts against output retries; a stopped stream is dropped from history and its
  tokens aren't in run usage). A handler that returns lets the response continue; it can raise `ModelRetry` to retry. Under
  `run_stream()`, a loop found once output (text or an output tool call) has started streaming
  can't be retried and ends the run with `UnexpectedModelBehavior`. `UsageLimits` don't see
  stopped streams.
- `LoopJudge(model=...)` asks a small, fast "System 1" model (not a reasoning model) on heuristic
  suspicion, plus every `every` chars if set; failure or `timeout=5.0` falls back to the heuristic.
  It's never asked in durable workflow code (Temporal, DBOS, Prefect), where only the heuristic runs
  on the whole response. A custom `LoopDetector` runs in workflow code too, so keep it deterministic
  and free of I/O. Subclass `LoopDetector` (`async detect(ctx, sample: StreamSample) ->
  Repetition | None`) for your own methodology.
- Emits a `loop detected` span (`detect_loops.pattern`, `.action`, `.count`, `.window`,
  `.tool_names`; `.fingerprints` and `.passage` only with `trace_include_content`); no
  arguments or results otherwise. A failed judge emits `loop judge failed`
  (`error.type`). A `model=` judge isn't instrumented; its usage is added to the run's.

## ToolBudgets

In `wrap_tool_execute`, refuses a function tool call that is over its `per_tool` budget, over the run's
`total`, or (for tools in `refuse_repeats`) an exact repeat of a call that already ran in this run.
The refused call doesn't run: `SkipToolExecution` answers it with a `ToolReturn` naming the budget
("3 of 3 `web_search` calls used; no more are allowed in this run."), and the run continues.

```python
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness import ToolBudgets


def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) < 5:
        return ModelResponse(parts=[ToolCallPart('web_search', {'query': 'pydantic'})])
    return ModelResponse(parts=[TextPart('done')])


budgets = ToolBudgets(per_tool={'web_search': 3}, total=20, refuse_repeats={'web_search'})
agent = Agent(FunctionModel(model), capabilities=[budgets])


@agent.tool_plain
def web_search(query: str) -> str:
    return f'results for {query}'


messages = agent.run_sync('Research Pydantic').all_messages()
parts = [part for message in messages for part in message.parts]
print([part.metadata for part in parts if isinstance(part, ToolReturnPart) and part.metadata])
#> [{'origin': 'pydantic-ai-harness.tool_budgets.refusal.v1', 'reason': 'repeat'}]
```

- Find refusals by `metadata['origin'] == 'pydantic-ai-harness.tool_budgets.refusal.v1'`, never by text;
  `metadata['reason']` is `'per_tool'`, `'total'` or `'repeat'` (`ToolBudgetReason`).
  `UsageLimits.tool_calls_limit` stays the hard ceiling and counts refused calls.
- A call counts when execution starts (success, `ModelRetry`, failure or `CallDeferred`); calls another
  capability skips or holds for approval, and refused calls, don't. Repeats match tool name plus sorted-key
  JSON arguments; a failed call isn't remembered. Parallel calls can't overshoot: they are checked against
  calls still in flight, so one refused because of a call that is then skipped stays refused.
- Counts live on the per-run instance from `for_run`, so compaction doesn't reset them; hooks run in
  workflow code under Temporal, DBOS, and Prefect, so replay reaches the same refusals. Each new run
  starts at zero, including one resumed after an approval round outside the run, so with approval
  rounds the budget is per round, not per task.
- Emits a `tool call refused` span (`tool_budgets.reason`, `tool_budgets.tool_name`).

## ReportToolErrors

Wraps the agent's toolset and turns an exception a function tool raises into `ToolFailed`, as if the
tool raised it, so the model gets a failed tool result (`outcome='failed'`) and the run continues.

```python
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness import ReportToolErrors


def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if len(messages) == 1:
        return ModelResponse(parts=[ToolCallPart('divide', {'a': 1, 'b': 0})])
    return ModelResponse(parts=[TextPart("Can't divide by zero.")])


agent = Agent(FunctionModel(model), capabilities=[ReportToolErrors()])


@agent.tool_plain
def divide(a: int, b: int) -> float:
    return a / b


result = agent.run_sync('What is 1 / 0?')
parts = [part for message in result.all_messages() for part in message.parts]
print([part.content for part in parts if isinstance(part, ToolReturnPart)])
#> ['`divide` raised ZeroDivisionError']
```

- The default text leaves out the exception's message (it can carry paths, hosts, connection strings).
- `format_error(ctx, info) -> str` replaces the text, for example to include `info.error`; `info` is a
  `ToolErrorInfo` (`name`, `args`, `tool_call_id`, `error`) from `pydantic_ai_harness.report_tool_errors`;
  `exceptions=(TimeoutError, ...)` reports only those types (default `(Exception,)`), the rest propagate.
- No limit of its own and the tool stays available; `UsageLimits.request_limit` stops a model that keeps
  failing (`tool_calls_limit` counts only successful calls; `ToolBudgets` caps one tool).
- Under Temporal, a failed tool activity reaches the workflow as `ActivityError` after its retry policy gives
  up; `ReportToolErrors` leaves it alone, so the run fails as without it.
- `ModelRetry`, `ToolFailed`, `ApprovalRequired`, `CallDeferred`, `SkipToolExecution`, `RunCancelled` pass
  through; `UsageLimitExceeded` and `UserError` still end the run, and so does a tool that exhausts its
  `retries` (`Coder` reports those instead). Output tools and argument validation aren't affected.
- A `ToolGuardrail` `result_guard` screens the reported text. `on_tool_execute_error` hooks aren't called
  for a reported exception, in any order; narrow `exceptions` to leave a type to such a hook. Under `CodeMode`, a
  snippet's nested tool call raises the `ToolFailed`, not the original exception. With several,
  the last listed formats the error. Emits no spans.

## ReportRemainingBudget

In `before_model_request`, enqueues a request-scoped `RemainingBudgetPart` (`InstructionCapabilityPart`) with what is left of the run's
`UsageLimits` (read from `ctx.usage_limits` and `ctx.usage`) and, given `spend_limits=`, of each
`SpendLimits` budget with a ceiling (read with `status(ctx)`). Disclosure only; enforcement stays with
`UsageLimits` and `SpendLimits`.

```python
from decimal import Decimal

from pydantic_ai import Agent

from pydantic_ai_harness import ReportRemainingBudget, SpendLimits
from pydantic_ai_harness.spend_limits import Budget

spend = SpendLimits(budgets=[Budget(usd=Decimal(10), window='day')])
agent = Agent('test', capabilities=[spend, ReportRemainingBudget(spend_limits=spend)])
```

```text
Budget remaining:
- this run: 8 requests including this one, about 38k total tokens, $1.42
- today: $12.50
```

- The run line lists requests (including the one being sent), tool calls, input/output/total tokens and
  `cost_limit`, for each limit set; `per_request_input_tokens_limit` isn't reported. Spend lines are labelled
  by window (`spend limit for this run`, `this conversation`, `today`, `this month`, `in total`) plus `name` unless
  `'default'`; counter-only budgets are left out. Tokens of 1,000 or more and USD are rounded down (`about 38k`).
- Pass the same `SpendLimits` instance the agent carries; its store is read on every request.
- A run without `usage_limits=` still reports core's default `request_limit=50`; nothing is sent only with
  `UsageLimits(request_limit=None)`, no other limit and no `spend_limits`. No spans: the status is part of
  the recorded model request.

## RefineOutput

When a run would end with an output, adds a system prompt telling the model to review its last answer
and give the complete answer again, revised. The last answer is the run's output. `rounds=1` sets how
many times per run; `instructions` replaces `DEFAULT_REFINE_OUTPUT_INSTRUCTIONS`.

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_harness import RefineOutput


def answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[TextPart('draft' if len(messages) == 1 else 'revised')])


agent = Agent(FunctionModel(answer), capabilities=[RefineOutput()])
print(agent.run_sync('Explain the change').output)
#> revised
```

- Each round costs at least one more model request and counts toward `UsageLimits`. Output validators
  and output functions run on every answer, so keep them free of side effects.
- The instruction is a `ReviewRoundPart` (an `InstructionCapabilityPart` from `pydantic_ai_harness.refine_output`;
  conversation scope, so it stays in the history); find it by `isinstance(part, ReviewRoundPart)`, never by text. Its
  `round` field never reaches the model. Each run starts with no
  rounds used. The count is per-run state, so compaction that drops the instruction does not restart it.
- Not reviewed: a run that returns `DeferredToolRequests` or `PendingModelResponse`, an output no model
  request produced (an `InputGuardrail` refusal; detected as a response with no `model_name`), and `run_stream()`, which ends at the first output it
  streams. Stream a refined run with `run(event_stream_handler=...)` or `iter()`; the stream carries
  every answer, the first one included.
- Several on one agent each count their own rounds and add their instructions to a round. Emits no spans.

## Goal

States the run's goal in the instructions and, when you pass an async `verify(ctx, output)`, calls it
on every final output. Return `None` to accept or a nonempty string saying
what remains; a rejection raises `ModelRetry` with the goal and the gap, so the model keeps working
in the same conversation. No per-run state, not serializable to an agent spec.

```python
from pydantic_ai import Agent, RunContext

from pydantic_ai_harness import Goal


async def verify(ctx: RunContext[None], output: object) -> str | None:
    return None if output == 'done' else 'Finish the migration before answering.'


agent = Agent('test', retries={'output': 3}, capabilities=[Goal(goal='Migrate the config.', verify=verify)])
```

- Rejections share the `retries={'output': n}` budget with other output validation, and exhaustion
  raises `UnexpectedModelBehavior`; `UsageLimits` still end the run. For a final tool-less answer at the
  limit add `WrapUpAtLimits`: its answer still goes through `verify`, and a rejection there raises
  `UsageLimitExceeded` since no request is left for the retry.
- Verify real evidence (files, tests, external state), not the model's claim. An LLM judge agent
  inside `verify` works: pass `usage=ctx.usage` to its `run()` and give it evidence, not just the answer.
- Output functions run before the check and can run again after a rejection, so keep them idempotent.
- Without `verify`, `Goal` only adds the goal instructions. `headless=False` drops the
  nobody-is-available text for interactive runs; a `verify` is still enforced. To let an interactive
  run stop and ask, set `headless=False` and leave `verify` unset.
- Pick `OutputGuardrail` instead for block/redact/retry policies that don't depend on a goal.
- Default `id='goal'`: two that disagree under one `id` raise, so give each extra `Goal` its own `id`.
  `verify` runs as the durable operation `verify` under Temporal, DBOS, and Prefect: a replay reuses
  its result; on Temporal a structured output arrives as JSON data (validate it, don't `isinstance`),
  `ctx.messages` is unavailable, and a verifier exception retries the activity instead of propagating.
- Emits a `goal.verify` span (`goal.met`; `goal.description`/`goal.gap` with `trace_include_content`).

## RequireVerification

When a run would end after an edit (a `FileSystem` `write_file`/`edit_file` call, matched by tool kind, or a
shell command matching `edit_commands`) and the latest run of every check since then (a shell command
matching `verification_commands`) did not pass, adds an instruction telling the model to run its checks or
explain the blocker. Up to `max_attempts=2` times per run; after that the run ends and emits a
`CompletionUnverifiedEvent` (`attempts`, `failed_commands`), dispatched immediately so `Hooks.on.event`
listeners get it before the run returns; at most once per run.

```python
from pydantic_ai import Agent

from pydantic_ai_harness import FileSystem, RequireVerification, Shell

agent = Agent('test', capabilities=[FileSystem(), Shell(), RequireVerification()])
```

- Any edit makes every earlier check stale. `write_file`/`edit_file` edits to `exempt_paths` (`*.md`, `*.mdx`,
  `*.rst`) don't count; shell edit commands always count, since the file they touch isn't known.
- Commands come from tools whose metadata sets `code_arg_language='shell'` and `code_arg_name`, as
  `Shell`'s do. A check is an `&&`-joined step of the last `;`/newline statement that starts with a
  `verification_commands` match (`cd pkg; pytest` counts, `pytest; echo done` does not). `|`, `||`,
  background `&`, command substitution, and `exit`/`exec`/`return` disqualify it; `2>&1` and quoted
  text are ignored. Pass/fail is read from `Shell`'s `[exit code: N]`/timeout/file-limit trailer, the `shell`
  tool's `exit_code`, or ACP's `[exited with code N]`/`[terminated by signal S]` (other formats pass); a background or still-running command never counts and clears that
  command's earlier result. A check that an edit in a parallel call of the same response overtook
  records nothing. Results are keyed by command text, not working directory; probes (`--version`, `--help`,
  `--collect-only`) are not checks; results are read inside `ToolOutputLimits`. `edit_commands` match outside quotes (defaults include formatters, `--fix`, `sed -i`,
  `git checkout`/`stash`/..., `mv`/`cp`/`rm`, `sh -c`/`python -c`/`node -e`, and `>` except `->`, `>&`, `/dev/null`).
- The redirect is a `VerificationRequiredPart` (an `InstructionCapabilityPart` from
  `pydantic_ai_harness.require_verification`, with `attempt`); find it by `isinstance`, never by text.
- State is per run (`for_run`); a run resumed after a deferred-tool pause recovers earlier `write_file`/`edit_file`
  and shell edits of the turn from history (not check results). An answer to a redirect with no tool call ends the redirects
  and emits the event. Hooks run in workflow code under Temporal, DBOS, and Prefect. Skipped like
  `RefineOutput`: deferred or pending output, output from no model request, and `run_stream()`.
- Emits `verification required` (`require_verification.attempt`) and `completion unverified`
  (`require_verification.attempts`) spans.

## WrapUpAtLimits

When the run's `UsageLimits` leave room for one more model request but not another, makes that request
the last: `tool_choice='none'` with the tool definitions kept (cached tools and instructions match), plus a
request-scoped `WrapUpPart` (an `InstructionCapabilityPart` from `pydantic_ai_harness.wrap_up_at_limits`)
with `prompt` (default `DEFAULT_WRAP_UP_PROMPT`). The answer is the
run's output instead of `UsageLimitExceeded`.

```python
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness import WrapUpAtLimits


def research(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if info.model_settings and info.model_settings.get('tool_choice') == 'none':
        return ModelResponse(parts=[TextPart('partial answer')])
    return ModelResponse(parts=[ToolCallPart('look_up', {'topic': 'x'})])


agent = Agent(FunctionModel(research), capabilities=[WrapUpAtLimits()])


@agent.tool_plain
def look_up(topic: str) -> str:
    return topic


print(agent.run_sync('Research', usage_limits=UsageLimits(request_limit=2)).output)
#> partial answer
```

- Triggers on `request_limit` (the last allowed request), `tool_calls_limit` (fewer calls left than the
  largest batch so far), and input/output/total token limits (projected from the run's responses).
  `cost_limit` and `per_request_input_tokens_limit` are not projected. A larger batch that goes over
  `tool_calls_limit` still raises. With no `usage_limits`, the default `request_limit=50` applies.
- `wrapped_up_by(result.new_messages())` returns the `LimitName` that cut the run short, or `None`; it reads
  the `WrapUpPart` on the final request, found by type.
- With tool output, `tool_choice='none'` sends only the output tools; Anthropic caches messages per
  `tool_choice`.
- List it after capabilities that spend from the run's usage before a request, so it sees what they spent.
- For a sub-agent, give it its own `SubAgent.usage_limits`; sharing the parent's budget spends the parent's
  last request.
- Compaction still runs on the final request; a summarizing strategy has no request left there, so wrap
  it in `FallbackCompaction(fallback_on=(..., UsageLimitExceeded))`. Stateless and durable-safe. Span: `run wrapped up`
  (`wrap_up_at_limits.limit`).

## See also

- https://pydantic.dev/docs/ai/harness/repair-tool-arguments/
- https://pydantic.dev/docs/ai/harness/guardrails/
- https://pydantic.dev/docs/ai/harness/goal/
- https://pydantic.dev/docs/ai/harness/prompt-injection-defender/
- https://pydantic.dev/docs/ai/harness/tool-call-judge/
- https://pydantic.dev/docs/ai/harness/approval-reviewer/
- https://pydantic.dev/docs/ai/harness/spend/
- https://pydantic.dev/docs/ai/harness/ask-user/
- https://pydantic.dev/docs/ai/harness/system-reminders/
- https://pydantic.dev/docs/ai/harness/trajectory-judge/
- https://pydantic.dev/docs/ai/harness/detect-loops/
- https://pydantic.dev/docs/ai/harness/refine-output/
- https://pydantic.dev/docs/ai/harness/report-tool-errors/
- https://pydantic.dev/docs/ai/harness/require-verification/
- https://pydantic.dev/docs/ai/harness/wrap-up-at-limits/
- https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/ (core tool approval and deferred tools)
