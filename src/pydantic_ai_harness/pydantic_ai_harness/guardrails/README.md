# Input, Output & Tool Guardrails

Guardrails run a callable you supply at three points of a run: the user prompt before the model sees it, each tool call and its result, and the final output before the caller sees it. Use them to redact secrets or PII, refuse off-topic prompts cheaply, vet or approve tool calls, or require citations in an answer. The callable allows, blocks, redacts, retries, or (for tool calls) defers for approval.

> [!NOTE]
> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](https://pydantic.dev/docs/ai/harness/#version-policy).

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/guardrails/)

## How it works

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

`pydantic_ai_harness` exports the three capabilities, `GuardrailResult`, the `InputGuardrailFunc` and `OutputGuardrailFunc` aliases, and the `GuardrailError`, `InputBlocked`, and `OutputBlocked` exceptions. `ToolCallInfo`, `ToolResultInfo`, `ToolBlocked`, the tool guard aliases, and `detectors` import from `pydantic_ai_harness.guardrails`:

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

A guard returns a bare `bool` (`True` = allow, `False` = block) for the simple case, or a `GuardrailResult` for the richer outcomes. Any other return value is read for truthiness, as in earlier releases, and emits a `HarnessDeprecationWarning`: a classifier label such as `'unsafe'` is truthy and lets the value through, and a future release will raise `UserError` instead. Return `bool(...)`, for example around a comparison on a NumPy or tensor score. Guards may also be async -- return an awaitable `bool`/`GuardrailResult`, for example to call a moderation API.

`OutputGuardrail` receives the output unchanged -- no automatic stringification. For a string output the guard reads it directly; for a typed (Pydantic model) output the guard gets the model instance, so pick the serialization that fits the check (read a field, or call `output.model_dump_json()` for JSON text). This avoids the trap of `str(MyModel(...))` producing a `MyModel(field=...)` repr that hides field contents from regex-based checks.

`InputGuardrail` checks the run's prompt. A run without a prompt of its own, such as `agent.run(None)` continuing a conversation, checks the latest user prompt in its history instead, and skips the background-task reports `SubAgents` delivers there, which it recognizes by their metadata. History passed in `message_history` is trusted as the caller's own, metadata included. Load history from an untrusted client with a [UI adapter](https://pydantic.dev/docs/ai/integrations/ui/overview/), which never puts client metadata on message content. `sanitize_messages` will be a safe path too once [#10943](https://github.com/pydantic/pydantic-ai/pull/10943) lands, which makes it drop `TextContent.metadata` from user prompts; until then it keeps that metadata, so don't rely on it alone for this.

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

An empty sequence, a set, or an iterator raises `UserError` the first time the
guardrail runs, not at construction: `InputGuardrail` on the first model request,
`OutputGuardrail` on the final output. A set has no order for the chain to run in,
and an iterator would be spent after one run.

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
sends as a separate user-prompt part. Adjacent text parts are scanned joined, so a
secret split across two parts is still caught. A `CachePoint` between two text parts
does not separate them; an image or document does. If sanitizing the joined text gives
a different result from sanitizing each part, the span becomes one part carrying the
first part's metadata, and a `CachePoint` between merged parts moves ahead of it;
otherwise each part keeps its own metadata. `ToolReturn` metadata, kind, and non-text
parts are kept.

As with `for_text`, a non-text result raises by default. `on_other='allow'`
skips the whole result, including the `content` of a `ToolReturn` whose
`return_value` is not a string. When a skipped result may carry sensitive `content`,
leave `on_other` at its default and apply the detector to a field of the output instead.

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

An input guard rewrites the prompt in place, so the model receives the broken version. On an agent that handles code or paths, use `personal_data(only=['us_ssn', 'credit_card', 'iban'])`, or put the detector on the output rather than the prompt.

`credit_card` matches 13 to 19 digits, in any grouping, starting with 2 to 6 and passing Luhn. Some digit runs still pass by chance, so lists of numbers can lose a value.

`iban` matches a country code and two check digits, allows spaces only where the printed form puts them, and checks every match against the ISO 7064 mod-97 check digits.

An AWS *secret* access key is not in the defaults. It is forty characters of base64 with no prefix, so a pattern for the shape alone would also match ordinary base64. Matching one needs an anchor on the name beside it (`aws_secret_access_key = ...`). Pass such a pattern through `extra=` if that is how keys reach your agent.

`only=` selects patterns; it does not reorder them. Patterns apply in their declared order whatever order `only` lists them in: `iban` runs before `credit_card` so a spaced account number is not labelled a card.

A private key matches with real newlines or the escaped `\n` of a JSON or `.env` file. A complete block and one pasted without its `END` marker share the one name `private_key`, so `only=['private_key']` covers both.

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
GuardrailResult.replace('cleaned')      # substitute a sanitized value and continue
GuardrailResult.retry('instruction')    # ask the model to redo the output or the tool call
GuardrailResult.approve()               # ToolGuardrail arguments only: defer the call for human approval
```

The block/retry message is produced at the moment the guard decides, so it can carry the guard's own reasoning rather than a string frozen at construction time.

## Redaction (`replace`)

Return `GuardrailResult.replace(value)` to sanitize rather than refuse. `InputGuardrail` rewrites the prompt sent to the model; `OutputGuardrail` substitutes the output returned to the caller. The replacement is not validated against the agent's `output_type`, so for a structured output return an instance of the same type, such as `output.model_copy(update=...)`.

```python
import re

from pydantic_ai import Agent
from pydantic_ai_harness import GuardrailResult, InputGuardrail, OutputGuardrail
from pydantic_ai_harness.guardrails.detectors import for_text

EMAIL_RE = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')


def scrub_emails(text: str) -> GuardrailResult:
    cleaned = EMAIL_RE.sub('[email]', text)
    return GuardrailResult.replace(cleaned) if cleaned != text else GuardrailResult.allow()


agent = Agent(
    'openai:gpt-5.4',
    capabilities=[
        InputGuardrail(guard=scrub_emails),   # strip PII before it reaches the model
        OutputGuardrail(guard=for_text(scrub_emails)),  # strip PII before it reaches the caller
    ],
)
```

Input redaction requires sequential mode -- it is incompatible with `parallel=True`, since a parallel guard runs alongside a model call that has already started with the original prompt. It also requires a text prompt. A guard sees a multimodal prompt (`agent.run_sync(['describe this', BinaryContent(...)])`) rendered as text, and one string written back over a prompt built from several parts would drop the attached images, documents and audio, so `replace` raises `UserError` there instead, including when the model was sent a text copy and the history's original is the multimodal one. The same holds for a run resuming from a request with several prompt parts: the guard screens them together, as `ctx.prompt` reports them, and one replacement cannot be written back over them. Return `allow` or `block` for those prompts, or guard the output.

## Retry (`retry`)

`OutputGuardrail` can send a bad output back to the model instead of blocking it. Return `GuardrailResult.retry(instruction)` -- the instruction is the retry prompt the model sees. This reuses pydantic-ai's normal retry machinery and counts against the run's output-retry budget.

```python
from pydantic_ai_harness import GuardrailResult, OutputGuardrail


def has_citations(output: object) -> bool:
    return 'Source:' in str(output)


def must_cite_sources(output: object) -> GuardrailResult:
    if not has_citations(output):
        return GuardrailResult.retry('Include at least one source citation.')
    return GuardrailResult.allow()


OutputGuardrail(guard=must_cite_sources)
```

## Accessing run context

A guard may take a `RunContext` as its first parameter when it needs run state -- `deps` for tenant- or role-aware policy, message history for conversation-aware checks. The parameter is detected from the signature. A guard with a second positional parameter takes `RunContext` first when that parameter is required. So does a `*args` parameter alongside another parameter. When the second parameter has a default, the first parameter's annotation decides: `RunContext` (also in a union or `Annotated`) means the run context, and any other type means the guarded value. A first parameter with no annotation, `Any`, `object`, or a type that cannot be resolved still gets `RunContext`, as in earlier releases, with a `HarnessDeprecationWarning`, unless the unresolved annotation names `RunContext`, which needs no warning. Keyword-only parameters are left to the guard, so prompt-only guards need not declare anything:

```python
from dataclasses import dataclass

from pydantic_ai import RunContext
from pydantic_ai_harness import InputGuardrail


@dataclass
class MyDeps:
    tier: str


def tenant_policy(ctx: RunContext[MyDeps], prompt: str) -> bool:
    return ctx.deps.tier == 'pro' or 'advanced-feature' not in prompt


InputGuardrail(guard=tenant_policy)
```

## Parallel input guards

A slow guard (an LLM classifier, a network call) run sequentially adds its latency to every turn. Set `parallel=True` to run the guard concurrently with the model call instead, overlapping the two so the guard adds no latency on the pass path. The model call is cancelled the moment the guard reports a violation.

```python
from pydantic_ai_harness import InputGuardrail


async def slow_async_classifier(prompt: str) -> bool:
    return 'internal-only' not in prompt  # stands in for an LLM or moderation API call


InputGuardrail(guard=slow_async_classifier, parallel=True)
```

Parallel mode trades tokens for latency: sequential mode never calls the model when the guard blocks, but parallel mode has already started the model call -- if the guard trips only after the model has responded, those tokens were spent. For fast local checks (regex, keyword lookup) sequential is the better default. `replace` is not available under `parallel=True` (see [Redaction](#redaction-replace)).

Normal tool dispatch starts after the model request, which the guard holds until it decides. That includes a request that raises before the guard decides, such as a `ModelRetry` from an `after_model_request` hook or a model error raised before the response streams: the exception waits for the verdict, and a block takes its place. In a streamed run, a block that lands after the whole response has streamed, as after a `ModelRetry` from `after_model_request`, is raised as `SkipModelRequest` (see [Streaming](#streaming)). `CodeMode` with `eager=True` or `speculate` starts nested tool calls while the model is still streaming, so each of those calls waits for the verdict before it runs: a blocked prompt runs none of them, and an allowed one releases them as soon as the guard decides.

## Hard-fail path

`block` is the graceful path. To make the caller see an exception instead, raise from the guard:

```python
from pydantic_ai_harness import InputBlocked
from pydantic_ai_harness.guardrails.detectors import redact_secrets


def contains_credentials(prompt: str) -> bool:
    return redact_secrets(prompt).action != 'allow'


def strict_guard(prompt: str) -> bool:
    if contains_credentials(prompt):
        raise InputBlocked('credentials detected')
    return True
```

Any exception raised by the guard propagates as-is -- use `InputBlocked` / `OutputBlocked` / `ToolBlocked` from this module, or your own exception types. `ToolBlocked` carries the `tool_name` and an optional `reason`.

## Tool calls

`InputGuardrail` screens only the user prompt, on the first model request of a run, so the arguments a tool receives and the content it returns pass unchecked unless a `ToolGuardrail` screens them. `ToolGuardrail` inspects both sides of a tool call: `guard` sees the validated arguments before the tool runs, `result_guard` sees what it returned before the model does. `result_guard` also screens the message of a `ModelRetry` or `ToolFailed` raised while the tool runs, including inside `CodeMode`, because that text reaches the model too. Hooks of a capability ordered inside this one (another `position='innermost'` capability) run within the same screening window, so their raw retries are screened as well, except one from an error hook, which this guard's error hook closes the window on; a capability ordered outside passes through. A screened failure stays a failure. A `ToolGuardrail` `retry` verdict is the guard's own instruction and is never screened.

```python
import re
from pathlib import Path

import httpx
from pydantic_ai import Agent
from pydantic_ai_harness import GuardrailResult, ToolGuardrail
from pydantic_ai_harness.guardrails import ToolCallInfo, ToolResultInfo

WORKSPACE = Path('/workspace')
SECRET_RE = re.compile(r'\bsk-[A-Za-z0-9_-]{20,}')


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

Both guards receive `args` as a read-only `MappingProxyType` over a deep copy made for each guard call, so large arguments are copied every time. To run several tool checks, stack several `ToolGuardrail` capabilities: they share the innermost tier in list order, so argument guards run first to last and result guards last to first.

### Human in the loop

Pydantic AI already owns the approval round trip: a call raising `ApprovalRequired` is held back, the run finishes with a `DeferredToolRequests` output, and you resume it with the human's answers. `ToolGuardrail` plugs into that rather than inventing a second mechanism, which means approvals a guard asks for and tools marked `requires_approval=True` arrive in the same place.

```python
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness import GuardrailResult, ToolGuardrail
from pydantic_ai_harness.guardrails import ToolCallInfo


def operator_says_yes(call: ToolCallPart) -> bool:
    return input(f'Run {call.tool_name}({call.args})? [y/N] ') == 'y'


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
    results = DeferredToolResults(
        approvals={
            call.tool_call_id: True if operator_says_yes(call) else ToolDenied('not on a Friday')
            for call in deferred.output.approvals
        }
    )
    final = await agent.run(message_history=deferred.all_messages(), deferred_tool_results=results)
```

A denial reaches the model as the tool's result, so the agent can explain itself or try something else. On the resumed run the guard is evaluated again, and `approve` becomes a no-op for a call the human already cleared -- every other verdict still applies, so a policy that has since changed its mind can still block an approved call. An `InputGuardrail` on the same agent also runs again: the resumed run carries no new prompt, so its first request screens the most recent prompt in the history.

Two shapes of approval, and which to reach for:

| | Deferred (`GuardrailResult.approve()`) | In-process (an async guard) |
|---|---|---|
| The run | ends, then resumes from `message_history` | stays open; the tool call awaits |
| Fits | HTTP APIs, queues, durable execution, anything that cannot hold a process open | CLIs, TUIs, desktop apps, a websocket to an operator |
| Human answer | `DeferredToolResults` (`True`, `ToolDenied`, or `ToolApproved(override_args=...)`) | whatever the guard returns |

The in-process shape needs nothing extra -- a guard may be async, so it can await the human directly:

```python
import asyncio
from collections.abc import Mapping
from typing import Any

from pydantic_ai_harness import GuardrailResult, ToolGuardrail
from pydantic_ai_harness.guardrails import ToolCallInfo


async def operator_approves(name: str, args: Mapping[str, Any]) -> bool:
    answer = await asyncio.to_thread(input, f'Allow {name}({dict(args)})? [y/N] ')
    return answer == 'y'


async def ask_the_operator(call: ToolCallInfo) -> GuardrailResult:
    if await operator_approves(call.name, call.args):
        return GuardrailResult.allow()
    return GuardrailResult.block('The operator declined this action.')


ToolGuardrail(guard=ask_the_operator)
```

Pydantic AI also offers approval without a guard at all: `requires_approval=True` on a tool, or `ApprovalRequiredToolset` for a synchronous predicate over a whole toolset. Reach for `ToolGuardrail` when the decision is async, needs `deps`, or should sit alongside the other verdicts.

Two fields narrow what a guard sees:

```python
from pathlib import Path

from pydantic_ai_harness import ToolGuardrail
from pydantic_ai_harness.guardrails import ToolCallInfo


def stay_in_the_workspace(call: ToolCallInfo) -> bool:
    return Path(str(call.args.get('path', '/workspace'))).resolve().is_relative_to('/workspace')


ToolGuardrail(
    guard=stay_in_the_workspace,
    tools=['write_file', 'run_shell'],  # guard only these; None (default) guards every tool
    hidden=['delete_everything'],       # withhold these from the model entirely
)
```

A hidden tool is dropped from the definitions sent to the model, so it costs no tokens and the model is not offered it (a call to it is rejected as an unknown tool); a blocked tool stays visible and the model learns it was refused. Hiding takes a static list of names -- for policy that depends on `deps` or on the arguments, use `guard`.

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

`OutputGuardrail` inspects the final output only -- during `run_stream()` partial chunks reach the caller before the guard runs, so a `block` or `replace` verdict cannot un-send content already streamed. Use `run()` / `run_sync()` when the output must be screened before any of it is exposed. `GuardrailResult.retry()` is not supported under `run_stream()` -- pydantic-ai does not retry output during streaming, and a `retry` verdict there surfaces as `UnexpectedModelBehavior`. `InputGuardrail` works the same in streamed and non-streamed runs, with one caveat for `parallel=True`: the model streams while the guard runs, so chunks sent before the guard blocks still reach the caller, before the block message replaces them. If the guard blocks only after the whole response has streamed, the run raises `SkipModelRequest` instead.

## Tracing

`replace` and `block` are recorded as spans on the active OpenTelemetry tracer, so a redaction or refusal shows up in Logfire traces (`guardrail redacted input`, `guardrail blocked output`, and so on) with `guardrail.*` attributes. Content attributes -- the replacement value for a redaction and the refusal `message` for a block -- are attached only when `RunContext.trace_include_content` is enabled, since these can quote the content the guard exists to keep out of traces. The value a redaction removed is never recorded, so content capture does not put a redacted secret into the trace. `retry` records no guardrail span: the retry request it causes appears in the trace as a normal model request.

Tool spans add a `guardrail.tool` attribute naming the tool. `approve` records `guardrail deferred tool args`, which always carries `guardrail.tool_call_id` so the span can be correlated with the `DeferredToolRequests` the application answers, and carries `guardrail.arguments` under the same `trace_include_content` rule as the other content attributes. A deferred call never executes, so there is no `execute_tool` span recording what was asked for.

`OutputGuardrail` declares `position='outermost', wrapped_by=[Instrumentation]`, so an enclosing `Instrumentation` span captures its block and redact spans whatever order the capabilities are listed in. `InputGuardrail` declares `position='innermost'`, so any capability that rewrites the user prompt (a prompt rewriter, a history processor) runs first and the guard sees the rewritten text. A `UserPromptPart` another capability adds to the request only, such as a `SystemReminders` reminder or a `Planning` plan, is not the user's prompt: the guard neither screens nor rewrites it. When an outer capability sends the model a copy of the prompt part, a redaction also reaches the original in the run's message history, which later steps replay. Every copy with the same text gets the same replacement. For a copy with changed text the guard is called again on the original's own text, so a guard with side effects or a per-call cost runs twice for that request; only a redaction from that call is written to the history, `allow` leaves the original as it was, and `block` blocks the run before anything is rewritten. A request rebuilt with a new prompt part is not tied to the history, so only the part sent to the model is rewritten. `ToolGuardrail` is also `position='innermost'`, which puts it last among argument hooks (it sees the arguments every other capability has finished modifying) and first among result hooks (it sees the raw tool result, before a capability such as `ToolOutputLimits` truncates or offloads it).

## API

```python {test="skip"}
InputGuardrailFunc = (
    Callable[[str], bool | GuardrailResult | Awaitable[bool | GuardrailResult]]
    | Callable[[RunContext[AgentDepsT], str], bool | GuardrailResult | Awaitable[bool | GuardrailResult]]
)
OutputGuardrailFunc = (
    Callable[[object], bool | GuardrailResult | Awaitable[bool | GuardrailResult]]
    | Callable[[RunContext[AgentDepsT], object], bool | GuardrailResult | Awaitable[bool | GuardrailResult]]
)
ToolGuardrailFunc = (
    Callable[[ToolCallInfo], bool | GuardrailResult | Awaitable[bool | GuardrailResult]]
    | Callable[[RunContext[AgentDepsT], ToolCallInfo], bool | GuardrailResult | Awaitable[bool | GuardrailResult]]
)
ToolResultGuardrailFunc = (
    Callable[[ToolResultInfo], bool | GuardrailResult | Awaitable[bool | GuardrailResult]]
    | Callable[[RunContext[AgentDepsT], ToolResultInfo], bool | GuardrailResult | Awaitable[bool | GuardrailResult]]
)


@dataclass(frozen=True, kw_only=True)
class GuardrailResult:
    action: Literal['allow', 'block', 'replace', 'retry', 'approve']
    message: str | None = None
    replacement: object = ...  # unset by default, so `replace(None)` is valid
    # classmethods: allow(), block(message=None), replace(value), retry(message), approve()


@dataclass
class InputGuardrail(AbstractCapability[AgentDepsT]):
    guard: InputGuardrailFunc[AgentDepsT] | Sequence[InputGuardrailFunc[AgentDepsT]]
    parallel: bool = False


@dataclass
class OutputGuardrail(AbstractCapability[AgentDepsT]):
    guard: OutputGuardrailFunc[AgentDepsT] | Sequence[OutputGuardrailFunc[AgentDepsT]]


@dataclass
class ToolGuardrail(AbstractCapability[AgentDepsT]):
    guard: ToolGuardrailFunc[AgentDepsT] | None = None
    result_guard: ToolResultGuardrailFunc[AgentDepsT] | None = None
    tools: Sequence[str] | None = None
    hidden: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True, kw_only=True)
class ToolCallInfo:
    name: str
    args: Mapping[str, Any]
    tool_call_id: str


@dataclass(frozen=True, kw_only=True)
class ToolResultInfo(ToolCallInfo):
    result: object
```

The guard callable takes the inspected value (the prompt for `InputGuardrail`, the output for `OutputGuardrail`, a `ToolCallInfo` or `ToolResultInfo` for `ToolGuardrail`), optionally preceded by a `RunContext`. `InputGuardrailFunc`, `OutputGuardrailFunc`, `ToolGuardrailFunc`, and `ToolResultGuardrailFunc` are the exported signature aliases; `GuardrailError` is the base for `InputBlocked`, `OutputBlocked`, and `ToolBlocked`.

## Relationship to `pydantic-ai-shields`

[`pydantic-ai-shields`](https://github.com/vstorm-co/pydantic-ai-shields), a community package by [Vstorm](https://vstorm.co/), ships each detector as its own capability. The equivalents here are functions instead, which is what lets several run as one chain, share a redaction, and sit beside a guard you wrote.

Two things it has that this deliberately does not. Its `PromptInjection` matches phrases like "ignore previous instructions": injection is ordinary language, so a pattern list catches the examples and misses the attack while flagging a pasted log, and a check that reads as protection without being it is worse than none. Its `NoRefusals` blocks the model from declining, which is a decision about what an agent may say rather than a guardrail on data, and not one to make a default.
