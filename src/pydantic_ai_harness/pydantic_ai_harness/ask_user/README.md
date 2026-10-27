# Ask User

Let the model ask the user multiple-choice questions mid-run and wait for the answers.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/ask_user/)

## The problem

An agent given an ambiguous task either guesses or buries a question in its output and stops.
Guessing wastes a run; a question in prose is only found by a person reading the whole answer.
The model needs a way to ask a small, structured question and get the answer back as data.

## The solution

`AskUser` exposes one tool, `ask_user_question`. The model passes one to ten questions, each
with a short `header`, the question text, two to six options (a `label` and an optional
`description`), and `multi_select` when several answers are allowed. The capability validates
the call, hands it to your `answerer`, and returns the picked labels keyed by header. If the
user declines, the model is told so and the run continues.

The capability owns the schema and validation. It never prints, reads stdin, or imports a
terminal library: the `answerer` is whatever puts the questions in front of a person, and you
supply it. There is no default, because a capability that reads stdin is unusable from a server.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import AskUser
from pydantic_ai_harness.ask_user import AskUserAnswer, AskUserRequest, AskUserResponse


async def pick_first(request: AskUserRequest) -> AskUserResponse:
    answers = [AskUserAnswer(header=q.header, selected=(q.options[0].label,)) for q in request.questions]
    return AskUserResponse(answers=tuple(answers))


agent = Agent('anthropic:claude-fable-5-1', capabilities=[AskUser(answerer=pick_first)])
```

`pick_first` stands in for a real UI. [CLAI](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_clai2/)
ships a terminal menu built on the same protocol; a web form would be another.

## Writing an answerer

An `Answerer` is one async callable: it takes an `AskUserRequest` and returns an
`AskUserResponse`. A plain `async def` qualifies; a class with `async def __call__` does too.

- `AskUserRequest.questions` holds the validated `Question` objects; `AskUserRequest.id`
  distinguishes concurrent or repeated calls so a UI can match its reply to the request.
- A completed `AskUserResponse` carries one `AskUserAnswer` per question, keyed by `header`,
  with the picked option labels in `selected`: exactly one unless the question is
  `multi_select`, at least one either way, and no label twice.
- An answerer may instead return `AskUserAnswer(header=q.header, custom_answer='My own answer')`.
  Leave `selected` empty. Custom text must be nonblank and contain no control characters other
  than newlines and no format characters such as zero-width spaces, joiners and bidirectional
  overrides, so run `check_response` on text a user typed. Its length is not limited, so cap free text that
  arrives from a client before returning it. The model receives it as a one-item list under the
  question's header.
  Whether to offer text entry is up to the answerer; the model cannot request it.
- When the user declines, return `AskUserResponse(cancelled=True)` with no answers. The tool
  result tells the model the user declined; nothing is raised into the run.
- A response that does not fit the request (an unknown header, a label the question did not
  offer, several labels on a single-select question, a missing answer) is a bug in the answerer
  and raises `ValueError`, which fails the run. `check_response` is exported so an answerer can
  validate before returning.
- By default, two `ask_user_question` calls in one model response reach the answerer
  concurrently, so serialize in the answerer if your UI shows one request at a time. Other tool
  calls from that response may run while the user decides: an answer shapes the model's next
  response, not calls it already made. To run a response's calls one at a time, wrap the run in
  `with Agent.parallel_tool_call_execution_mode('sequential'):`.
- The model writes the questions, and a prompt injection can make it ask for a password or a
  token. Present the questions as coming from the model, not as a system or sign-in prompt.

## Bounding the wait

The run waits inside the tool call for the answerer to return. Set `timeout`, in seconds, to
bound that wait. When it runs out, the answerer is cancelled, `AskUserAnsweredEvent` fires with
`timed_out=True` and a cancelled response, and the model is told the user did not answer in time
(the `TIMED_OUT` result) and continues.

The timeout cancels the answerer, so the answerer must be cancellable. A thread cannot be
cancelled: run blocking work with `anyio.to_thread.run_sync(..., abandon_on_cancel=True)`, or the
run waits for the thread and then uses its late answer. An abandoned thread keeps running, so a
terminal answerer should read input with an async prompt library rather than `input()` in a
thread, which would take the next line the user types.

```python {test="skip"}
AskUser(answerer=pick_first, timeout=300)
```

## Pausing the run until the user answers

A web server or a durable worker often cannot hold a run open while a person decides. Pass
`answerer=None` and every `ask_user_question` call is [deferred](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/) instead:
the run ends with `DeferredToolRequests` output, so your `output_type` must include it, and you
answer in a later run, from the same process or another one.

- `AskUserRequest.from_tool_call(call)` rebuilds the validated request from a pending call. Its
  `id` is the call's `tool_call_id`, the same request `AskUserRequestedEvent` carried before the
  run paused.
- `ask_user_result(request, response)` checks the response like an answerer's (raising
  `ValueError` if it does not fit) and renders the tool result the model would have got inline.
  Pass it in `DeferredToolResults.calls` under the call's ID.
- No `AskUserAnsweredEvent` fires for a deferred call: the answer comes from you, in the next run.

```python
from pydantic_ai import Agent, AgentRunResult, DeferredToolRequests, DeferredToolResults, ModelMessage
from pydantic_ai_harness import AskUser
from pydantic_ai_harness.ask_user import TOOL_NAME, AskUserRequest, AskUserResponse, ask_user_result

agent = Agent(
    'anthropic:claude-fable-5-1',
    output_type=[str, DeferredToolRequests],
    capabilities=[AskUser(answerer=None)],
)


def pending_questions(requests: DeferredToolRequests) -> dict[str, AskUserRequest]:
    """What the paused run is waiting on, keyed by tool call ID: show these to the user."""
    return {
        call.tool_call_id: AskUserRequest.from_tool_call(call)
        for call in requests.calls
        if call.tool_name == TOOL_NAME
    }


async def resume(
    messages: list[ModelMessage],
    questions: dict[str, AskUserRequest],
    responses: dict[str, AskUserResponse],
) -> AgentRunResult[str | DeferredToolRequests]:
    """Answer the paused run's questions and carry on."""
    calls = {call_id: ask_user_result(request, responses[call_id]) for call_id, request in questions.items()}
    return await agent.run(message_history=messages, deferred_tool_results=DeferredToolResults(calls=calls))
```

Keep the paused run's `all_messages()` and its pending questions wherever you keep sessions,
then call `resume` when the answers arrive. `timeout` needs an answerer to bound, so combining it
with `answerer=None` raises `UserError`.

## Watching without answering

Two `CapabilityEvent`s let anything else in the run observe the exchange:

| Event | When | Fields |
| --- | --- | --- |
| `AskUserRequestedEvent` | before the answerer is called, or before a deferred call pauses the run | `request` |
| `AskUserAnsweredEvent` | after it returns, times out, or raises, before the response is checked or the model sees the result | `request_id`, `response`, `timed_out`, `failed` |

Both dispatch immediately, so a listener that shows a "waiting for you" state sees the wait
start and end in step with the run. When the answerer raises, the answered event carries
`failed=True` and a cancelled response, then the error propagates from the tool call: a
`ModelRetry` sends the model a retry prompt, and any other error fails the run. A run
cancelled while the answerer waits emits no answered event, so also clear waiting state when the
run ends. Subscribe with `@on_event` on a capability or through the run's event stream:

```python
from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, on_event

from pydantic_ai_harness.ask_user import AskUserAnsweredEvent, AskUserRequestedEvent


class WaitIndicator(AbstractCapability[None]):
    @on_event(AskUserRequestedEvent)
    async def waiting(self, ctx: RunContext[None], event: AskUserRequestedEvent) -> None:
        print(f'waiting on {len(event.request.questions)} question(s)')

    @on_event(AskUserAnsweredEvent)
    async def done(self, ctx: RunContext[None], event: AskUserAnsweredEvent) -> None:
        if event.failed:
            print('failed')
        elif event.timed_out:
            print('timed out')
        else:
            print('declined' if event.response.cancelled else 'answered')
```

## What the model sees

The tool schema mirrors [Code Puppy](https://github.com/mpfaffenberger/code_puppy)'s
`ask_user_question`, so prompts written for it carry over.

| Field | Limit |
| --- | --- |
| `questions` | 1 to 10 per call; headers unique, ignoring case and compatibility forms such as full-width letters |
| `header` | 1 to 25 characters, one line |
| `question` | 1 to 500 characters, may span lines |
| `options` | 2 to 6 per question; labels unique, ignoring case and compatibility forms such as full-width letters |
| `label` | 1 to 50 characters, one line |
| `description` | optional, at most 200 characters, may span lines |

These strings are drawn on terminals, where an escape sequence in a prompt-injected call is an
attack, so control characters and format characters such as zero-width spaces, joiners and
bidirectional overrides are rejected. Spaces of any width are allowed, and uniqueness treats
most of them as an ordinary space. A call outside these limits, or with a field the schema
does not have, goes back to the model as a validation retry and never reaches the answerer.
Once validated the questions are frozen: what the answerer sees is what the model asked.

The result is a JSON object mapping each header to the list of picked labels (or one custom answer), or the sentence
`The user declined to answer. Continue without the answer, or ask differently if it is essential.`
When `timeout` runs out, it is the `TIMED_OUT` sentence instead.

The capability adds one instruction: ask when the task is ambiguous and the answer is not in
the workspace, offer concrete options, batch related questions, and make a stated choice if
the user declines.

## One per agent

`AskUser` declares the default `id` `'ask_user'`. The `id` keys the capability's instructions, so they
can be addressed, and lets `defer_loading=True` work without one. Two `AskUser`s on one agent that
share it are one answerer stated twice: they become one when they agree, and the agent raises a
`UserError` when it's built if they differ. To keep two, give each its own `id` and wrap them
in [PrefixTools](https://pydantic.dev/docs/ai/capabilities/prefix-tools/), since both register `ask_user_question`.
To serve several UIs, pass one answerer that dispatches to them.

Earlier releases declared no default `id`, so two always failed on the tool name when the run
started. Two identical ones now become one, and two that differ fail when the agent is built.

## Tracing

`AskUser` emits no spans. Core's tool-call span already covers the wait, and the two events
above carry what was asked and answered. That span records the tool's arguments and result,
which here are the questions and the user's answers, custom text included. Set
[`include_content=False`](https://pydantic.dev/docs/ai/integrations/logfire/#excluding-prompts-and-completions) to leave them out.

## Specs

`Agent.from_spec` cannot construct `AskUser`: the answerer is a live object a spec has no way
to name.

## API

`AskUser`, `Answerer`, `AskUserRequest`, `AskUserResponse`, `AskUserAnswer`, `Question`,
`QuestionOption`, `AskUserRequestedEvent`, `AskUserAnsweredEvent`, `check_response`, `ask_user_result`,
`TOOL_NAME`, `DECLINED`, `TIMED_OUT`, and `MAX_QUESTIONS` are exported from `pydantic_ai_harness.ask_user`,
along with `AskUserToolset`, the toolset `AskUser` registers, and `ASK_USER_EVENTS`, the namespace
of both events. See the [docs page](https://pydantic.dev/docs/ai/harness/ask-user/) for
signatures and defaults.
