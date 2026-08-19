# Input, Output & Tool Guardrails

Validate the user prompt before it reaches the model, the tool calls the model makes along the way, and the output before it reaches the caller.

> [!NOTE]
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/guardrails/)

## The problem

Agents take unstructured input from users and return unstructured output to callers. Without a validation layer, a prompt injection attempt, PII-laden message, or off-topic question goes to the model as-is, and any output the model produces is returned verbatim. The framework does not reason about "this is unsafe to send" or "this is unsafe to show".

Tool calls are the third edge, and the one an input guard cannot reach: `InputGuardrail` evaluates the first model request only, so the arguments a tool receives and the content a tool returns both pass unchecked. Tool results are where untrusted content enters an agent loop -- a fetched page, a file's contents, an MCP server's response.

## The solution

Three capabilities -- `InputGuardrail`, `OutputGuardrail`, and `ToolGuardrail` -- each backed by a `guard` callable you supply. The guard inspects a value and returns one of five outcomes. For the run's two edges:

| Outcome | `InputGuardrail` | `OutputGuardrail` |
|---|---|---|
| **allow** | send the prompt to the model | return the output to the caller |
| **block** | skip the model call; a refusal message becomes the response (`SkipModelRequest`) | raise `OutputBlocked` |
| **replace** | rewrite the prompt sent to the model (redaction) | substitute a sanitized output |
| **retry** | -- (not valid for input) | send the output back to the model to try again (`ModelRetry`) |
| **approve** | -- (not valid for input) | -- (not valid for output) |

`ToolGuardrail` uses the same outcomes on both sides of a tool call; see [Tool calls](#tool-calls). A guard that raises an exception instead propagates it as a hard failure. The asymmetry between input `block` and output `block` is intentional: blocking the input spends no tokens, so a graceful refusal is almost always right; blocking the output means the model already produced something you do not want exposed, so raising forces the caller to decide what to do next.

## Usage

A guard returns a bare `bool` (`True` = allow, `False` = block) for the simple case, or a `GuardrailResult` for the richer outcomes.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import GuardrailResult, InputGuardrail, OutputGuardrail


def no_secrets(prompt: str) -> bool:
    return 'api_key' not in prompt.lower()


def no_pii(output: object) -> GuardrailResult:
    if 'SSN' in str(output):
        return GuardrailResult.block('The response contained personal data.')
    return GuardrailResult.allow()


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[
        InputGuardrail(guard=no_secrets),
        OutputGuardrail(guard=no_pii),
    ],
)
```

`OutputGuardrail` receives the output unchanged -- no automatic stringification. For a string output the guard reads it directly; for a typed (Pydantic model) output the guard gets the model instance, so pick the serialization that fits the check (read a field, or call `output.model_dump_json()` for JSON text). This avoids the trap of `str(MyModel(...))` producing a `MyModel(field=...)` repr that hides field contents from regex-based checks.

Guards may also be async -- return an awaitable `bool`/`GuardrailResult`, e.g. to call a moderation API.

## Several guards at once

`InputGuardrail.guard` and `OutputGuardrail.guard` take one callable or a
sequence of them; `ToolGuardrail`'s `guard` and `result_guard` take a single
callable each. In a chain the guards run in order, and what happens next depends
on the verdict:

| Verdict | Effect on the chain |
|---|---|
| `allow` | move on to the next guard |
| `replace` | the rest of the chain inspects the substituted value |
| `block` / `retry` | the chain ends there, since neither leaves a value to judge |

`replace` threading forward is what makes order meaningful: put a redactor
first and everything after it sees the cleaned text.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import InputGuardrail
from pydantic_ai_harness.guardrails.detectors import blocked_keywords, redact_secrets

agent = Agent(
    'openai:gpt-5.4',
    capabilities=[InputGuardrail(guard=[redact_secrets, blocked_keywords(['internal-only'])])],
)
```

One `InputGuardrail` holding two checks is not the same as two `InputGuardrail`
capabilities: the chain is one place in the capability list, one ordering
decision, and one set of spans, and only the chain threads a redaction into the
check that follows it.

A redaction reaches the run's message history only if the chain finishes: a
later `block` ends it before `InputGuardrail` writes the cleaned prompt back, so
the original stays in the history. Put the redactor last when that matters, at
the cost of the checks before it seeing the original text.

An empty sequence is refused -- when the guardrail first runs, not when it is
constructed. A guardrail that inspects nothing reads as configured and behaves
as absent, which is worth an error rather than a quiet pass. A set and a
one-shot iterator are refused the same way: a set has no order for the chain to
run in, and an iterator is spent after the first request, since the chain is
rebuilt per request.

## Ready-made detectors

`pydantic_ai_harness.guardrails.detectors` holds checks you would otherwise
write again: they are plain functions returning a `GuardrailResult`, so they drop
into a chain beside your own.

```python
from pydantic_ai_harness.guardrails import detectors

detectors.redact_secrets  # rewrites vendor API keys, tokens, and whole private-key blocks out of text
detectors.redact_personal_data  # rewrites emails, card numbers, IBANs, US SSNs
detectors.blocked_keywords(['internal-only'])  # refuses text containing any of them
```

The two redactors rewrite rather than refuse, which is the useful default:
an agent that quoted a key back has still done the work, and blocking the answer
loses it while leaving the key in the message history either way.

Each is `secret_data()` and `personal_data()` called with the defaults. Use those factories directly when you need to narrow or extend them:

```python
from pydantic_ai_harness.guardrails.detectors import secret_data

secret_data(only=['aws_access_key', 'private_key'])
secret_data(extra={'internal_ticket': r'INT-\d{4}'})
secret_data(placeholder='***')  # the default is `[redacted:{name}]`, which says what it removed
```

Detectors read text, so they suit a prompt directly. An agent output may be a
model instance, and substituting a scrubbed string for one would change its
type, so `for_text` makes you say what should happen:

```python
from pydantic_ai_harness import OutputGuardrail
from pydantic_ai_harness.guardrails.detectors import for_text, redact_secrets

OutputGuardrail(guard=for_text(redact_secrets))  # raises on a non-string output
OutputGuardrail(guard=for_text(redact_secrets, on_other='allow'))  # skips it deliberately
```

A tool result guard receives `ToolResultInfo`, not just a value. Use
`for_tool_result_text` to adapt a detector for a bare string result or the text
channels of a `ToolReturn`:

```python
from pydantic_ai_harness import ToolGuardrail
from pydantic_ai_harness.guardrails.detectors import for_tool_result_text, redact_secrets

ToolGuardrail(result_guard=for_tool_result_text(redact_secrets))
```

The adapter checks a string `return_value` and all text in `content`, which core
sends as a separate user-prompt part. Text parts the model reads as one span are
sanitized as one string, so `content=[a, b]` gets the same treatment as
`content=a + b` and a secret split across the two is still caught. A part
carrying no model content does not end a span: a `CachePoint` between two texts
is dropped by providers without caching, so the text either side of it is joined,
while an image or document between them is not, since joining across real content
would invent adjacency.

The span keeps its parts, and each part its own metadata, when sanitizing them one
by one already produces what sanitizing the whole span produces. Parts that run
into one another are the case where the two differ: a key at the end of one part
and a word at the start of the next are one token to the model, so that span
collapses into a single part carrying the whole-span result. A `CachePoint` in a
collapsed span is kept, not dropped. One before the first text or after the last
keeps its side; one between two merged texts has lost the split it marked and
moves ahead of the merged text, narrowing the cached prefix rather than widening
it. When it redacts either channel, it preserves the `ToolReturn` metadata, kind,
and non-text content.

As with `for_text`, a non-text result raises by default. `on_other='allow'`
skips the whole result rather than part of it: neither `return_value` nor
`content` is scanned. A `ToolReturn` whose `return_value` is structured counts
as a non-text result, so sensitive text in its `content` is not scanned under
that option. The opt-out is the caller taking responsibility for the result;
scanning `content` there would return a redaction from the branch documented as
allowing the result, and it would still be uneven, since a structured result
that is not a `ToolReturn` has no separate text channel to scan. When a result
may carry sensitive `content`, leave `on_other` at its default and apply the
detector to a field of the output instead.

A detector on the input side needs a text prompt. A multimodal prompt reaches
the guard rendered as text, so a detector that matches one returns `replace`,
which `InputGuardrail` refuses rather than dropping the attached parts (see
"Redaction (`replace`)" below). Guard the output instead when prompts can carry
attachments.

**Where a shape is not enough.** `email` matches anything shaped like an address, because that is all an address is:

```python
from pydantic_ai_harness.guardrails.detectors import redact_personal_data

redact_personal_data('git clone git@github.com:pydantic/pydantic-ai.git')
# replaces `git@github.com`, leaving a command that no longer runs
```

An input guard rewrites the prompt in place, so the model receives the broken version. On an agent that handles code or paths, use `personal_data(only=['us_ssn', 'credit_card', 'iban'])`, or put the detector on the output rather than the prompt. `credit_card` has it less badly. It covers the 13 to 19 digits ISO/IEC 7812 allows, in any grouping, and only where the leading digit is 2 to 6, which is what a payment card starts with and a millisecond timestamp does not. Every match is then checked against the Luhn algorithm, which discards most of the runs that are left. Not all of them -- roughly one run of four consecutive years in ten satisfies the checksum by chance, so a prompt listing years can still lose one.

`iban` has the same problem and the same answer: a country code plus two digits is a shape ordinary text hits constantly, so spaces are allowed only where the printed form puts them and every match is checked against the ISO 7064 mod-97 digit an IBAN carries.

An AWS *secret* access key is deliberately absent from the defaults. It is forty characters of base64 with no distinguishing prefix, so nothing in the value marks it as a key and a pattern for the shape alone would take ordinary base64 with it. Matching one means anchoring on the name written beside it -- `aws_secret_access_key = ...` -- which is what [`pydantic-ai-shields`](https://github.com/vstorm-co/pydantic-ai-shields) does, and which finds the key only where it is written as an assignment. That narrower pattern is not shipped here; pass it through `extra=` if that is how keys reach your agent.

`only=` selects patterns; it does not reorder them. The application order is part of each mapping's contract -- `iban` runs before `credit_card` so a spaced account number is not labelled a card -- and it holds that order whatever order you list `only` in.

A private key is matched whether its line breaks are real newlines or the escaped `\n` a JSON service-account file or a `.env` line carries, which is how one usually reaches a chat window. Terminated or not, it is one `private_key` pattern rather than two names, so `only=['private_key']` cannot select the complete block and leave a key pasted without its `END` marker unredacted.

**What these do not do.** A regex finds a credential because credentials have a
shape. It does not find a prompt injection, which is ordinary language, and it
does not understand context, so a redactor will sometimes take a string that
only looks like a key. They are one cheap layer, not the answer.

## `GuardrailResult`

Construct a `GuardrailResult` with its classmethods, not the raw fields:

```python
from pydantic_ai_harness import GuardrailResult

GuardrailResult.allow()                 # let the value through
GuardrailResult.block('reason')         # refuse; `reason` is optional (a default is used otherwise)
GuardrailResult.replace(cleaned_value)  # substitute a sanitized value and continue
GuardrailResult.retry('instruction')    # ask the model to redo the output or the tool call
GuardrailResult.approve()               # ToolGuardrail arguments only: defer the call for human approval
```

The block/retry message is produced at the moment the guard decides, so it can carry the guard's own reasoning rather than a string frozen at construction time.

## Redaction (`replace`)

Return `GuardrailResult.replace(value)` to sanitize rather than refuse. `InputGuardrail` rewrites the prompt sent to the model; `OutputGuardrail` substitutes the output returned to the caller.

```python
def scrub_emails(text: str) -> GuardrailResult:
    cleaned = EMAIL_RE.sub('[email]', text)
    return GuardrailResult.replace(cleaned) if cleaned != text else GuardrailResult.allow()


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[
        InputGuardrail(guard=scrub_emails),   # strip PII before it reaches the model
        OutputGuardrail(guard=scrub_emails),  # strip PII before it reaches the caller
    ],
)
```

Input redaction requires sequential mode -- it is incompatible with `parallel=True`, since a parallel guard runs alongside a model call that has already started with the original prompt. It also requires a text prompt. A guard sees a multimodal prompt (`agent.run_sync(['describe this', BinaryContent(...)])`) rendered as text, and one string written back over a prompt built from several parts would drop the attached images, documents and audio, so `replace` raises `UserError` there instead. Return `allow` or `block` for those prompts, or guard the output.

## Retry (`retry`)

`OutputGuardrail` can send a bad output back to the model instead of blocking it. Return `GuardrailResult.retry(instruction)` -- the instruction is the retry prompt the model sees. This reuses pydantic-ai's normal retry machinery and counts against the run's output-retry budget.

```python
def must_cite_sources(output: object) -> GuardrailResult:
    if not has_citations(output):
        return GuardrailResult.retry('Include at least one source citation.')
    return GuardrailResult.allow()


OutputGuardrail(guard=must_cite_sources)
```

## Tool calls

`ToolGuardrail` inspects both sides of a tool call: `guard` sees the validated arguments before the tool runs, `result_guard` sees what it returned before the model does.

```python
from pathlib import Path

import httpx
from pydantic_ai import Agent
from pydantic_ai_harness import GuardrailResult, ToolGuardrail
from pydantic_ai_harness.guardrails import ToolCallInfo, ToolResultInfo

WORKSPACE = Path('/workspace')


def stay_in_the_workspace(call: ToolCallInfo) -> GuardrailResult:
    if call.name == 'write_file':
        # `resolve()` before the containment check: a prefix test on the raw
        # string accepts `/workspace/../etc/passwd`.
        target = Path(str(call.args['path'])).resolve()
        if not target.is_relative_to(WORKSPACE):
            return GuardrailResult.block(f'{target} is outside the workspace.')
    return GuardrailResult.allow()


def scrub_secrets(info: ToolResultInfo) -> GuardrailResult:
    # Text only. `str()` on a structured result or a `ToolReturn` yields a repr, and
    # replacing it with that string would change the result's type -- but only on the
    # calls where the pattern happened to match.
    if not isinstance(info.result, str):
        return GuardrailResult.allow()
    cleaned = SECRET_RE.sub('[redacted]', info.result)
    return GuardrailResult.replace(cleaned) if cleaned != info.result else GuardrailResult.allow()


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[ToolGuardrail(guard=stay_in_the_workspace, result_guard=scrub_secrets)],
)


@agent.tool_plain
def write_file(path: str, content: str) -> str:
    Path(path).write_text(content)
    return f'wrote {path}'


@agent.tool_plain
def fetch_page(url: str) -> str:
    return httpx.get(url).text
```

The outcomes map onto Pydantic AI control flow rather than a parallel mechanism:

| Outcome | `guard` (arguments) | `result_guard` (result) |
|---|---|---|
| **allow** | run the tool | return the result unchanged (the guard is handed the object the tool produced, so read it and use `replace` rather than mutating it) |
| **block** | skip execution; the refusal message becomes the tool result (`SkipToolExecution`) | the refusal message replaces the result |
| **replace** | run the tool with substituted arguments (a mapping); the call recorded in the message history keeps the model's original arguments. The replacement is trusted to match the tool's signature: keys the tool does not accept reach it as keyword arguments and raise a bare `TypeError` that names neither the tool nor the guard | substitute a sanitized result |
| **retry** | ask the model to redo the call (`ModelRetry`) | ask the model to redo the call (`ModelRetry`); the tool has already run once, so its side effects have happened and the retry runs it again |
| **approve** | defer the call for human approval (`ApprovalRequired`) | -- (the tool has already run) |

`block` is graceful on both stages: the agent sees the refusal text where it expected a tool result, so it can explain the refusal or try another approach. To fail the run instead, raise `ToolBlocked` from the guard.

### Human in the loop

Pydantic AI already owns the approval round trip: a call raising `ApprovalRequired` is held back, the run finishes with a `DeferredToolRequests` output, and you resume it with the human's answers. `ToolGuardrail` plugs into that rather than inventing a second mechanism, which means approvals a guard asks for and tools marked `requires_approval=True` arrive in the same place.

```python
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai_harness import GuardrailResult, ToolGuardrail
from pydantic_ai_harness.guardrails import ToolCallInfo


def confirm_production(call: ToolCallInfo) -> GuardrailResult:
    if call.args.get('env') == 'prod':
        return GuardrailResult.approve()
    return GuardrailResult.allow()


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[ToolGuardrail(guard=confirm_production)],
    output_type=[str, DeferredToolRequests],
)


@agent.tool_plain
def deploy(env: str) -> str:
    return f'deployed to {env}'


deferred = await agent.run('deploy the new build')
if isinstance(deferred.output, DeferredToolRequests):
    approvals = {
        call.tool_call_id: True if operator_says_yes(call) else ToolDenied('not on a Friday')
        for call in deferred.output.approvals
    }
    final = await agent.run(
        message_history=deferred.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals=approvals),
    )
```

A denial reaches the model as the tool's result, so the agent can explain itself or try something else. On the resumed run the guard is evaluated again, and `approve` becomes a no-op for a call the human already cleared -- every other verdict still applies, so a policy that has since changed its mind can still block an approved call.

Two shapes of approval, and which to reach for:

| | Deferred (`GuardrailResult.approve()`) | In-process (an async guard) |
|---|---|---|
| The run | ends, then resumes from `message_history` | stays open; the tool call awaits |
| Fits | HTTP APIs, queues, durable execution, anything that cannot hold a process open | CLIs, TUIs, desktop apps, a websocket to an operator |
| Human answer | `DeferredToolResults` (`True`, `ToolDenied`, or `ToolApproved(override_args=...)`) | whatever the guard returns |

The in-process shape needs nothing extra -- a guard may be async, so it can await the human directly:

```python
async def ask_the_operator(call: ToolCallInfo) -> GuardrailResult:
    if await operator_approves(call.name, call.args):
        return GuardrailResult.allow()
    return GuardrailResult.block('The operator declined this action.')


ToolGuardrail(guard=ask_the_operator)
```

Pydantic AI also offers approval without a guard at all: `requires_approval=True` on a tool, or `ApprovalRequiredToolset` for a synchronous predicate over a whole toolset. Reach for `ToolGuardrail` when the decision is async, needs `deps`, or should sit alongside the other verdicts.

Two fields narrow what a guard sees:

```python
ToolGuardrail(
    guard=stay_in_the_workspace,
    tools=['write_file', 'run_shell'],  # guard only these; None (default) guards every tool
    hidden=['delete_everything'],       # withhold these from the model entirely
)
```

`hidden` is not a blocklist with a nicer name. A hidden tool is dropped from the definitions sent to the model, so it costs no tokens and the model never attempts it; a blocked tool stays visible and the model learns it was refused. Hiding takes a static list of names -- for policy that depends on `deps` or on the arguments, use `guard`.

Configured `hidden` names are checked when a run completes successfully. Configured
`tools` names are checked then only when `guard` or `result_guard` is set. A dynamic
toolset may omit a tool on one step and offer it later, so a name that never appears
is the only typo signal. That warning catches `tools=['send_monye']`, which otherwise
leaves the intended tool unguarded, and a misspelled `hidden` name, which otherwise
stays visible to the model.

### What a tool guard does not see

Three kinds of call never reach the execution hooks, so neither `guard` nor `result_guard` is consulted for them:

- **Output tools**, which produce the agent's structured output. Screen that with `OutputGuardrail`.
- **External and deferred tools**, which the run hands back to your application in `DeferredToolRequests` instead of executing. Pydantic AI rejects them before any execution hook runs, so a guard cannot vet the arguments -- your application is the thing that executes them, and the check belongs there. `hidden` *does* cover them, since it works on the tool definitions.
- **Provider-side builtin tools** such as web search, which run inside the provider and come back as builtin call/return parts rather than tool executions.

A `ToolGuardrail` is a control over tools this run executes. For the ones it does not, `hidden` is the lever that still applies.

## Streaming

`OutputGuardrail` inspects the **final** output only -- during `run_stream()` partial chunks reach the caller before the guard runs, so a `block` or `replace` verdict cannot un-send content already streamed. Use `run()` / `run_sync()` when the output must be screened before any of it is exposed. `GuardrailResult.retry()` is **not** supported under `run_stream()` -- pydantic-ai does not retry output during streaming, and a `retry` verdict there surfaces as `UnexpectedModelBehavior`. `InputGuardrail` (including `parallel=True`) works the same in streamed and non-streamed runs.

## Tracing

`replace` and `block` are recorded as spans on the active OpenTelemetry tracer, so a redaction or refusal shows up in Logfire traces (`guardrail redacted input`, `guardrail blocked output`, etc.) with `guardrail.*` attributes. Content attributes -- the original/replacement values for a redaction and the refusal `message` for a block -- are attached **only** when `RunContext.trace_include_content` is enabled, since these can quote the very content the guard exists to keep out of traces. `retry` needs no special tracing: the retried model request appears in the trace on its own.

Tool spans add a `guardrail.tool` attribute naming the tool. `approve` records `guardrail deferred tool args`, which always carries `guardrail.tool_call_id` so the span can be correlated with the `DeferredToolRequests` the application answers, and carries `guardrail.arguments` under the same `trace_include_content` rule as the other content attributes. A deferred call never executes, so there is no `execute_tool` span recording what was asked for.

`OutputGuardrail` declares `position='outermost', wrapped_by=[Instrumentation]` so its block/redact spans are always captured by an enclosing `Instrumentation` span regardless of how the user orders capabilities. `InputGuardrail` declares `position='innermost'` so any capability that morphs messages (a prompt rewriter, a context manager) runs first and the guard sees the final prompt the model will receive. `ToolGuardrail` is also `position='innermost'`, which puts it last among argument hooks (it sees the arguments every other capability has finished modifying) and first among result hooks (it sees the raw tool result, before a capability such as `ToolOutputLimits` truncates or offloads it).

## Parallel input guards

A slow guard (an LLM classifier, a network call) run sequentially adds its latency to every turn. Set `parallel=True` to run the guard concurrently with the model call instead, overlapping the two so the guard adds no latency on the pass path. The model call is cancelled the moment the guard reports a violation.

```python
InputGuardrail(guard=slow_async_classifier, parallel=True)
```

Parallel mode trades tokens for latency: sequential mode never calls the model when the guard blocks, but parallel mode has already started the model call -- if the guard trips only after the model has responded, those tokens were spent. For fast local checks (regex, keyword lookup) sequential is the better default. `replace` is not available under `parallel=True` (see [Redaction](#redaction-replace)).

## Accessing run context

A guard may take a `RunContext` as its first parameter when it needs run state -- `deps` for tenant- or role-aware policy, message history for conversation-aware checks. The parameter is detected from the signature, so prompt-only guards need not declare it:

```python
from pydantic_ai import RunContext
from pydantic_ai_harness import InputGuardrail


def tenant_policy(ctx: RunContext[MyDeps], prompt: str) -> bool:
    return ctx.deps.tier == 'pro' or 'advanced-feature' not in prompt


InputGuardrail(guard=tenant_policy)
```

## Hard-fail path

`block` is the graceful path. To make the caller see an exception instead, raise from the guard:

```python
from pydantic_ai_harness import InputBlocked


def strict_guard(prompt: str) -> bool:
    if contains_credentials(prompt):
        raise InputBlocked('credentials detected')
    return True
```

Any exception raised by the guard propagates as-is -- use `InputBlocked` / `OutputBlocked` / `ToolBlocked` from this module, or your own exception types. `ToolBlocked` carries the `tool_name` and an optional `reason`.

## API

```python {test="skip"}
@dataclass
class GuardrailResult:
    action: Literal['allow', 'block', 'replace', 'retry', 'approve']
    message: str | None = None
    replacement: object = ...  # an unset sentinel, so `replace(None)` is valid
    # classmethods: allow(), block(message=None), replace(value), retry(message), approve()


InputGuardrail(
    guard: InputGuardrailFunc | Sequence[InputGuardrailFunc],  # several run in order
    parallel: bool = False,
)

OutputGuardrail(
    guard: OutputGuardrailFunc | Sequence[OutputGuardrailFunc],
)

ToolGuardrail(
    guard: Callable[..., bool | GuardrailResult | Awaitable[bool | GuardrailResult]] | None = None,
    result_guard: Callable[..., bool | GuardrailResult | Awaitable[bool | GuardrailResult]] | None = None,
    tools: Sequence[str] | None = None,
    hidden: Sequence[str] = (),
)


@dataclass(frozen=True)
class ToolCallInfo:
    name: str
    args: Mapping[str, Any]
    tool_call_id: str


@dataclass(frozen=True)
class ToolResultInfo(ToolCallInfo):
    result: object
```

The guard callable takes the inspected value -- the prompt for `InputGuardrail`, the output for `OutputGuardrail`, a `ToolCallInfo` or `ToolResultInfo` for `ToolGuardrail` -- optionally preceded by a `RunContext`.

## Relationship to `pydantic-ai-shields`

[`pydantic-ai-shields`](https://github.com/vstorm-co/pydantic-ai-shields) ships each detector as its own capability. The equivalents here are functions instead, which is what lets several run as one chain, share a redaction, and sit beside a guard you wrote.

Two things it has that this deliberately does not. Its `PromptInjection` matches phrases like "ignore previous instructions": injection is ordinary language, so a pattern list catches the examples and misses the attack while flagging a pasted log, and a check that reads as protection without being it is worse than none. Its `NoRefusals` blocks the model from declining, which is a decision about what an agent may say rather than a guardrail on data, and not one to make a default.
