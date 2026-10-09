# Live Agent Control tests

Eighteen claims about [Agent Control](../../../../docs/harness/agent-control.md), checked against a running
Logfire platform with real model requests. Nothing here is mocked: the tests that need a published
config put one there through the platform's own variables API, every test resolves through the real
Logfire SDK and sends a real request to a real provider, and the evidence comes back out of what
ran -- the request the model was handed, the spans the process exported, and, for the hint span, the
platform's own copy read back with the query API.

```bash
make integration-logfire-platform
```

## Read this first: it publishes and deletes

**This suite creates, publishes over and deletes variables on the project it is pointed at, before
and after every test.** A conformance suite for a feature whose entire surface is stored state has
to own that state, so each test publishes what it is about to check and the variable is removed
either side of it.

Three things keep that from reaching anything you care about:

- **The names are generated per run**: they all begin `agent__harness_agent_control_live_<32 hex>`,
  from an agent name made at import time, and the two hint-span read-back tests add a suffix of
  their own so the span they query for is one no other test emitted. So the only config the suite
  can delete is one this run created, and no pre-existing config can be in its way. A run killed
  outright (rather than failed, where the fixtures still tear down) can leave one of those behind;
  they are safe to delete.
- It does nothing at all unless `LOGFIRE_PLATFORM_ALLOW_WRITES=1` is set. Unconfigured, every test
  skips with the reason.
- It reads its own `LOGFIRE_PLATFORM_*` variables, never the SDK's `LOGFIRE_API_KEY`. A credential
  that happens to be in your shell for a real project cannot become the one this publishes with.

Running it between UI edits on some other agent is therefore safe. It still sends spans and makes
real model requests against whatever you point it at, which is the part to think about before
pointing it at a project that matters.

## What to point it at

| Variable | What it is |
|---|---|
| `LOGFIRE_PLATFORM_ALLOW_WRITES` | Set to `1` to allow the creating, publishing and deleting above. Without it the suite skips |
| `LOGFIRE_PLATFORM_API_KEY` | An API key with `project:read_variables` **and** `project:write_variables`. The span write token cannot serve the variables API |
| `LOGFIRE_PLATFORM_TEST_URL` | The platform origin: UI, OTLP and API. Defaults to `http://localhost:3000` |
| `LOGFIRE_PLATFORM_WRITE_TOKEN` | A span write token. Without it spans stay in the process, every local check still runs, and the two read-back tests skip |
| `LOGFIRE_PLATFORM_READ_TOKEN` | A read token for the query API, which is how a test proves the hint span *arrived* rather than that it was emitted |
| `LOGFIRE_PLATFORM_REQUIRE_LIVE` | Set to `1` to make an unusable platform fail instead of skip, for running this as a gate in a pipeline of your own |
| `ANTHROPIC_API_KEY` | The code-side model (`anthropic:claude-haiku-4-5`). Without it the whole suite skips |
| `OPENAI_API_KEY` | The published model (`openai:gpt-5.4-nano`), a different provider on purpose. Without it the two model tests skip |

Twenty-nine tests and around forty real model requests -- several need a second round trip, and
a few publish and run again -- so a few minutes end to end.

## Getting a platform up

The stack is the `pydantic/platform` Docker Compose setup, which serves the UI, the OTLP endpoint
and the API from one origin (`http://localhost:3000` by default, which is why that is the default
here). Follow that repository's own instructions to bring it up; the credentials it seeds for local
development are what the variables above want.

Two traps, both of which cost real time:

**`show tables` never lists materialized views.** If the Agents page reports
`table '__ff_default__agent_runs_gen_ai_v1' not found`, do not reach for `show tables`: it lists
only base views, by design, on healthy stacks too. Ask the CRUD service instead, which answers with
the views that exist (28 by default):

```bash
curl "http://localhost:8010/mv?organization_id=<org>"
```

**Retag a stale `fusionfire-dev` image before the first `compose-up` after a data wipe, never
after.** A stale image is the usual reason those materialized views are missing in the first place,
and the first startup after a wipe is what creates them. Retagging it to a current published image
afterwards leaves you with a fresh database and no views.

## What it proves

One or more tests per claim, named after the claim rather than the mechanism. The module docstring
in [`test_live_agent_control.py`](test_live_agent_control.py) lists all eighteen; the shape of the
evidence is what matters here:

- **The agent as written stays the agent as written.** Nothing published, an unparsable value, and
  a platform on a closed port all leave the prompt, the model and the tools exactly as the code has
  them, and the run completes.
- **One block at a time.** Every override is checked against the code's own block text, so "this
  block changed and nothing else did" is asserted against the five other blocks rather than
  asserted in the abstract.
- **Both ends of a tool rename.** The model is offered the managed name (read off the request), the
  model calls it (read off the message history), the code's function runs under its code-side
  `ctx.tool_name` (read off the tool body), and the answer carries the real tool result.
- **The hint span, twice.** Once off the local OpenTelemetry pipeline, with every attribute the
  contract promises, and once read back out of the platform with the query API. A span that was
  emitted is not the same claim as a span that arrived.
- **A prompt containing `auth` survives scrubbing.** Logfire's scrubbing is on by default and
  matches substrings, and the hint's attributes are exempt from it. Without the exemption the
  `toolset:orders` block ("Order tools are authoritative ...") reaches the editor as
  `[Scrubbed due to 'auth']` and `baseline_sha256` stops verifying. This suite runs with scrubbing
  at its default, so it is the regression test for that.

The agent under test lives in [`_agent.py`](_agent.py): six addressable prompt blocks written in
five different ways, four tools in two toolsets, real `deps`, and code-side settings.
[`examples/agent_control.py`](../../examples/agent_control.py) is the readable version of the same
agent, and runs with no Logfire at all.

## Why CI does not run it

The three suites beside this one need a single container each, which a GitHub Actions service
container provides. This needs a whole Logfire platform plus two provider credentials, so it is a
suite you run by hand and read the output of. That is also why it is written to skip rather than
fail when it is not pointed anywhere: an unconfigured checkout must not go red for it.

Because it runs in one process rather than one process per scenario, it clears the two
once-per-process guards between tests (the hint span's and the drop warning's), exactly as
`tests/logfire_variables/conftest.py` does. It also inherits `filterwarnings = error` from the root
pytest config, so a test that expects a warning has to record it rather than let it through.

## What it deliberately leaves out

- **Whether Pydantic AI allows two toolsets to advertise one tool name.** That is a fact about core,
  it needs no platform, and it belongs in `tests/logfire_variables`. It is the reason the agent here
  gives every tool a distinct name and still qualifies each published override by `toolset`.
- **`seed`, `presence_penalty` and `frequency_penalty`.** All three are canonical contract keys that
  pydantic-ai's Anthropic model has no field for, so they are accepted by the run and never reach
  the provider. The contract has an `ApplyIssue` reason for exactly that
  (`'dropped-by-provider'`) and the adapter emits none, so Logfire can show a setting the agent does
  not send. Asserting the current behavior would pin the gap open; it is written down here instead.
