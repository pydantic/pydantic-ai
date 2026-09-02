# Testing Guidelines

## Testing philosophy

VCR + public-API tests are the default. We test through the public API the way a user would (`Agent(...)`, `agent.run(...)`) against real provider responses recorded as cassettes — provider APIs are the ultimate judge of whether the code is correct when run as intended, and that user-facing correctness is what we care about, not behavior in isolated units.

Unit tests still earn their place — for internal behavior that is definitory and worth pinning against drift. That includes behavior you can't reach or reliably trigger through the public API (pre-request guards, defensive branches no real model produces), but also behavior a VCR test wouldn't actually protect: our cassette matchers aren't always sensitive to the request body, so a changed internal payload can still match an existing recording and pass green — a unit test asserting the internal shape directly is what catches that regression. Each unit test should still say why it isn't (or can't be) a VCR test.

Recording cassettes needs provider API keys and isn't trivial, so contributors routinely under-test the real behavior — writing the VCR test a contributor couldn't is core maintainer work.

Recorded cassettes also double as a suite-wide prompt-cache prefix regression net. The cache-prefix invariant checks that consecutive requests extend the same serialized provider-cache prefix; tests that deliberately move it through compaction, dynamic tool disclosure, or history rewriting must use `@pytest.mark.moves_cache_prefix(reason=...)` on the test.

## Test File Structure

```python
from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.models import Model
# ... other imports

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]


# fixtures/helpers immediately before their test
@pytest.fixture
def my_helper():
    ...


@pytest.mark.parametrize('model', ['openai', 'anthropic', 'google'], indirect=True)
@pytest.mark.parametrize('stream', [False, True])
async def test_feature(model: Model, stream: bool):
    ...
```

## Parametrization with Expectations

For cartesian product tests, use a dict to map parameter combinations to expected results:

```python
from vcr.cassette import Cassette

from pydantic_ai.models import Model

# expectation can be a dataclass, for more complex cases
EXPECTATIONS: dict[tuple[str, bool], str] = {
    ('openai', False): 'expected output for openai non-streaming',
    ('openai', True): 'expected output for openai streaming',
    ('anthropic', False): 'expected output for anthropic non-streaming',
    ('anthropic', True): 'expected output for anthropic streaming',
}


@pytest.mark.parametrize('model', ['openai', 'anthropic'], indirect=True)
@pytest.mark.parametrize('stream', [False, True])
async def test_feature(model: Model, stream: bool, request: pytest.FixtureRequest, vcr: Cassette):
    """What the test is asserting.

    Use the `request` fixture to access test parameter values.

    Use the `vcr` to make assertions about the HTTP requests if needed.
    To assert what the code actually sent — headers included — take the `request_capture` fixture
    instead. It reads the wire rather than the recording, so it is preferred whenever a transport
    can be injected.
    """
    model_name = request.node.callspec.params['model']
    expected = EXPECTATIONS[(model_name, stream)]

    agent = Agent(model)
    if stream:
        async with agent.run_stream('hello') as result:
            output = await result.get_output()
    else:
        result = await agent.run('hello')
        output = result.output

    assert output == expected
```

Use the `EXPECTATIONS` dict only for a pure cartesian output lookup keyed by `(model, stream)`. For feature-centric files where cases are heterogeneous — different inputs, expectations, and xfails per case, all run through one minimal comprehensive test — use a `@dataclass Case` with sensible defaults plus per-case overrides:

```python
from dataclasses import dataclass, field

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage


@dataclass(frozen=True)
class Case:
    id: str
    model: str
    prompt: str = 'hello'
    instructions: str | None = None
    expected_messages: list[ModelMessage] = field(default_factory=list[ModelMessage])
    marks: tuple[pytest.MarkDecorator, ...] = ()


CASES = [
    Case(
        id='openai',
        model='openai:gpt-5',
        expected_messages=snapshot([...]),
    ),
    Case(
        id='anthropic',
        model='anthropic:claude-sonnet-4-5',
        instructions='be terse',
        expected_messages=snapshot([...]),
        marks=(pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed'),),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(c, id=c.id, marks=c.marks) for c in CASES])
async def test_feature(case: Case):
    agent = Agent(case.model, instructions=case.instructions)
    result = await agent.run(case.prompt)
    assert result.all_messages() == case.expected_messages
```

Each case carries its own snapshot (not the central test body), so a reviewer can read the cases top to bottom and check that every expectation is realistic.

## VCR Workflow

Record cassettes with `--record-mode=rewrite`, verify playback without the flag, and review diffs.
For detailed workflows see `.claude/skills/testing-skill/SKILL.md`.

### Realtime WebSocket cassettes

Realtime cassettes live in `tests/realtime/cassettes/<module>/<test>.yaml`. Record them with
`uv run --env-file .env pytest --record-mode=rewrite <test>`; normal test runs replay them offline.
Unlike HTTP VCR cassettes, they contain raw WebSocket frames, with secrets scrubbed and audio payloads
truncated. The testing skill's `parse_cassette.py` is HTTP-only and does not apply to these cassettes.

### Asserting what goes out on the wire

Four mechanisms, and they are not interchangeable — pick by what the test's claim is about.

**An httpx request hook — the boundary tap, and the first reach.** Request the `request_capture` fixture from `tests/conftest.py`, route the provider through `request_capture.client`, and assert on `request_capture.body(path_suffix)`; see `test_anthropic_code_execution_tool_container_reuse` in `tests/models/test_anthropic.py`. For an Anthropic model the `anthropic_model` factory does the routing, so `anthropic_model('claude-sonnet-4-5', capture=True)` is the whole setup. The fixture's client is an `httpx2.AsyncClient` carrying an `event_hooks={'request': [...]}` recorder, and event hooks run inside `AsyncClient.send`, above the transport VCR patches, so the hook fires on replay too and observes the request the live code actually built — a field the code stops sending fails the assertion instead of hiding behind a frozen recording. It costs the cassette nothing: matching is untouched, so fields that legitimately vary stay free to vary, and it sees what a cassette will not, headers included (the cassette serializer strips `anthropic-*`, so the wire is the only place a test can assert beta gating). Snapshot a projection of the body rather than the body itself — `content_blocks`, `message_shape` and `cache_breakpoints`, in `tests/models/conftest.py`, pin the fields a claim rests on without churning on unrelated conversation-shape changes. The fixture is reachable from anywhere under `tests/`; those three projections are not. A test outside `tests/models/` imports them explicitly (`from tests.models.conftest import message_shape`) rather than re-implementing the comprehension inline. The shared fixture works only when the provider accepts `httpx2.AsyncClient` as its injected client. Check the provider signature before using it. `GroqProvider` requires legacy `httpx.AsyncClient`, and `HuggingFaceProvider` rejects `http_client` in favor of `hf_client`; capture those SDK boundaries with provider-specific clients. A provider on another transport taps its own send-time event for the same reason — see `_capture_bedrock_request_headers` in `tests/models/test_bedrock.py`, a botocore `before-send` handler.

**Snapshot the request body — the review surface.** `single_request_body` in `tests/cassette_utils.py` decodes the request the *cassette* holds, so `assert single_request_body(vcr) == snapshot({...})` puts the whole outbound payload in the test body instead of leaving it buried in a long YAML. What it does not do is catch drift: the recording is frozen, so it keeps passing after the live code stops sending a field, and a recording can already disagree with what the code sends today. Its second job is at re-record time, where a changed payload surfaces as a snapshot diff. Where a hook is attached, snapshot what the hook captured instead — same review surface, and it is the request that is actually going out.

**A per-test matcher — the gate for when you cannot tap the boundary.** Making a field part of the cassette match also fails when the live code drifts: a request that no longer carries the recorded value cannot match its recording, so replay errors instead of quietly passing. Reach for it when the test doesn't build its own model (a shared or `indirect` fixture supplies it), when one guard should cover a whole family of tests, or when the provider isn't on httpx. Register the matcher via the `pytest_recording_configure(config, vcr)` hook and opt the test in with `@pytest.mark.vcr(additional_matchers=['<name>'])`, which adds to the defaults rather than replacing them; see `function_calling_mode` in `tests/models/google/conftest.py`. Match on the fields the test's claim rests on, not on every field you snapshot — matching a field that legitimately varies makes the cassette brittle and buys no information, which is the cost a hook doesn't have.

**Capture the render — for claims no single exchange holds.** When the SDK client is mocked outright so nothing goes over the wire, or the claim is that two renderings are byte-identical rather than that one recorded request looked a certain way, capture what the adapter produces and assert against that; see `test_tool_availability_delta_and_the_tools_cache_section` in `tests/test_tool_availability_portability.py`, and `rendered_requests` in `tests/models/test_anthropic_mid_conversation_system.py` for the per-request form. Where the requests do go out over httpx, the boundary tap gets the same evidence without reaching into adapter internals.

Default for a test that asserts an outbound field: take the capture fixture and assert on what it recorded, snapshotting a projection when the whole payload is worth reviewing. Fall back to snapshotting the cassette body and matching on the field where no hook can be attached. Don't add field-by-field asserts alongside a snapshot — the snapshot already pins them.

## Key Fixtures

### From `conftest.py`

#### Model requests
- `allow_model_requests` - bypasses the default `ALLOW_MODEL_REQUESTS = False`

#### Blocking-call detection
- `blockbuster` (autouse) raises `BlockingError` when a scanned library frame blocks inside the event loop.
- Scanned packages are `pydantic_ai`, `pydantic_graph`, `pydantic_evals`, and `clai`.
- Test-only and third-party stacks are ignored. User callbacks remain covered while a scanned library frame is below them.
- Fix blocking calls by offloading with `anyio.to_thread.run_sync`.
- Add legitimate blocking calls to `BLOCKBUSTER_EXEMPTIONS` in `tests/conftest.py` with a reason.
- Override `blockbuster_excluded_modules` when one integration module intentionally provides synchronous APIs.
- Override `blockbuster_enabled` at the narrowest test or integration-module boundary that intentionally performs blocking work.
- Use the narrowest module or test boundary. Do not replace the autouse `blockbuster` fixture.
- Each pytest worker configures one detector per exclusion set and activates it separately for each test.
- CI enables BlockBuster in Python 3.13 slim, evals, and standard jobs, plus unique compatibility or live-provider jobs.
- The all-extras job disables BlockBuster because its stack-inspection overhead exceeds the seven-minute CI budget. Rebenchmark after [BlockBuster #61](https://github.com/cbornet/blockbuster/pull/61) is released in a compatible version, and re-enable it if the job stays within seven minutes.
- Other Python-version lanes disable BlockBuster because they repeat the same dependency set.
- Lowest-version lanes disable BlockBuster to keep jobs within the seven-minute CI budget. This leaves dependency-version-specific blocking behavior outside CI detection.
- Local targeted tests enable BlockBuster by default.

#### The `model` fixture (use with `indirect=True`)

The `model` fixture takes a string param (e.g. `'openai'`, `'anthropic'`, `'google'`) and returns a configured `Model` instance, using session-scoped API key fixtures that default to `'mock-api-key'` (real keys loaded from env when recording).
See `tests/conftest.py` for the full list of supported param values.

```python
@pytest.mark.parametrize('model', ['openai', 'anthropic'], indirect=True)
async def test_something(model: Model):
    ...
```

#### Environment management
- `env` - `TestEnv` instance for temporary env var changes
  ```python
  def test_missing_key(env: TestEnv):
      env.remove('OPENAI_API_KEY')
      with pytest.raises(UserError):
          ...
  ```

#### Binary content (session-scoped)
- `assets_path` - `Path` to `tests/assets/`
- `image_content` - `BinaryImage` (kiwi.jpg)
- `audio_content` - `BinaryContent` (marcelo.mp3)
- `video_content` - `BinaryContent` (small_video.mp4)
- `document_content` - `BinaryContent` (dummy.pdf)
- `text_document_content` - `BinaryContent` (dummy.txt)

#### SSRF protection for URL downloads
- `disable_ssrf_protection_for_vcr` - required for VCR tests that download URL content (`ImageUrl`, `AudioUrl`, `DocumentUrl`, `VideoUrl` with `force_download=True`)
- An autouse guard raises a `RuntimeError` if a VCR test triggers SSRF validation without this fixture

#### Asserting the outbound request
- `request_capture` - a `RequestCapture` whose `client` records every outbound request; read bodies with `.body(path_suffix)` / `.bodies(path_suffix)` and headers off `.headers`. See "Asserting what goes out on the wire" for when to reach for it

### From `models/conftest.py`

Available to every test under `tests/models/`. See "Asserting what goes out on the wire" for when to reach for them.

- `anthropic_model` - factory for `AnthropicModel`; `capture=True` routes it through `request_capture.client`
  ```python
  async def test_something(
      allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
  ):
      agent = Agent(anthropic_model('claude-sonnet-4-5', capture=True))
      await agent.run('hello')
      assert message_shape(request_capture.body('/v1/messages')) == snapshot([...])
  ```
- `content_blocks(body, block_type)`, `message_shape(body)`, `cache_breakpoints(body)` - plain functions, not fixtures; projections of a captured body to snapshot instead of the whole payload

## Assertion Helpers

### From `conftest.py`
- `IsNow(tz=timezone.utc)` - datetime within 10 seconds of now
- `IsStr()` - any string, supports `regex=r'...'`
- `IsDatetime()` - any datetime
- `IsBytes()` - any bytes
- `IsInt()` - any int
- `IsFloat()` - any float
- `IsList()` - any list
- `IsInstance(SomeClass)` - instance of class

### Additional helpers
- `IsSameStr()` - asserts same string value across multiple uses in one assertion
  ```python
  assert events == [
      {'id': (msg_id := IsSameStr())},
      {'id': msg_id},  # must match first
  ]
  ```

## Best Practices

- Test through public APIs, not private methods (prefixed with `_`) or helpers — validates actual user-facing behavior and prevents brittle tests tied to implementation details
- Prefer feature-centric parametrized test files (e.g. `test_multimodal_tool_returns.py`) over appending to monolithic `test_<provider>.py` files — the legacy per-provider files are large and hard for agents to navigate; new features should get their own test file with a `Case` class and parametrized providers
- Use `snapshot()` for complex structured outputs (objects, message sequences, API responses, nested dicts) — catches unexpected changes more reliably than field-by-field assertions; use `IsStr` and similar matchers for variable values
- Assert the core aspect of the change being introduced — use whatever means necessary: patching clients to inspect request payloads, tapping into pydantic-ai internals, snapshot comparisons. Snapshots are valuable for catching structural drift in objects and message arrays, but only use `result.all_messages()` or output assertions when the structure demonstrates behavior you care about keeping consistent
- Pin recorded facts in the test body, never in the cassette alone — cassettes are long and humans skim them, so a load-bearing fact that only lives in the recording is effectively unreviewed. If a recording demonstrates something worth keeping (a resolved model alias, the actual returned format, an echoed setting), assert it in the test and explain it in the docstring
- Test both positive and negative cases for optional capabilities (model features, server features, streaming) — ensures features work when supported AND fail gracefully when absent
- When a change branches on a model-profile flag (a `profile.get(...)` read), at least one test must pin a model on each side of that flag so both the flag-on and flag-off paths run; if only one side is reachable, say why in the test. This is stricter than the capability bullet above and mechanical: it fires on any profile-flag branch, including a flag that is ambient to the feature under test, and one side alone does not satisfy it. Coverage cannot catch the gap, since the branch line reads at 100% while the on/off combination goes unvisited
- Ensure test assertions match test names and docstrings — tests without proper assertions or that verify opposite behavior create false positives
- Test MCP against real `tests.mcp_server` instance, not mocks — extend test server with helper tools to expose runtime context (instructions, client info, session state)
- Remove stale test docstrings, comments, and historical provider bug notes when behavior changes
- Prefer `instructions=` over `system_prompt=` when the test doesn't specifically need the system-prompt code path — `instructions=` is the canonical entry point for non-system-prompt-specific behavior (cacheable prefix, persona priming, format guidance), and reserving `system_prompt=` for tests that exercise the system-prompt machinery keeps intent legible
- Never reference line numbers in test docstrings or comments (`lines 872-873`, `L42`, `line 100`) — they go stale on the next edit to the referenced file. Describe the condition or behavior instead
- When testing prompt caching, assert prefix stability, not just a cache hit — `cache_read_tokens > 0` only proves that some prefix was reused, not that the prefix you intended stayed stable as history grows; pin it by asserting the cacheable region (serialized leading blocks up to the breakpoint) is byte-identical across consecutive requests and/or that the cache read covers the full prior prefix, since a per-request injection or serialization that moves with history length silently busts the cache with no error
- When you deprecate a public symbol (add `@deprecated`), add a test asserting the warning fires — a `pytest.warns(PydanticAIDeprecationWarning, match=...)` block whose `match` pins both the symbol name and the migration guidance — so the deprecation is protected against accidental removal or a message rewrite, and any executable docstring/doc example that constructs the symbol is marked `{test="skip"}` (it would otherwise fail under `filterwarnings=["error"]`)
- Don't blanket-suppress `reportDeprecated` at file scope — no whole-file `# pyright: ignore` (or `# pyright: basic`) at the top of a file to kill every deprecation static-warning at once, since that also silences the next *unintended* use of a deprecated symbol in the file. Put `# pyright: ignore[reportDeprecated]` on each intentional call instead. (The runtime side is different and conventional: a module-level `warnings.filterwarnings('ignore', message=...)` scoped to one deprecation message, in a file that pervasively constructs the deprecated symbol it tests — see `tests/durable_exec/temporal/test_agent.py` / `tests/durable_exec/test_prefect.py` — is fine, as long as a dedicated `pytest.warns` test still proves the deprecation fires.)
- Gate on a missing optional dependency with `try_import()`, the default. It has two halves and they live in different files: each **test module** wraps its optional imports in `with try_import() as imports_successful:` and applies `pytestmark = pytest.mark.skipif(not imports_successful(), reason='<dep> not installed')` (see `tests/models/anthropic/test_native_tool_filtering.py`); a **`conftest.py`** wraps its own optional imports the same way but carries no skip of its own (see `tests/models/anthropic/conftest.py`) — a conftest's `pytestmark` is not inherited by the modules it serves, so the mark there would gate nothing while `try_import` still keeps the file importable. Reach for `pytest.skip(..., allow_module_level=True)` only when the module cannot import `tests/conftest.py` at all — the Temporal suite can't, because that would pull `vcr` into a module the workflow sandbox re-imports. In that case the gate goes in the test modules **and any non-test helper they import** (`tests/durable_exec/temporal/_shared.py`), never in the `conftest.py`, which instead stays importable without the dependency by deferring its fixtures' imports into the fixture bodies. Reason: pytest loads the conftest of every command-line argument's directory before collection, and `_importconftest` catches only `Exception` while `Skipped` is a `BaseException`, so a gate there escapes as a traceback with exit 1 the moment someone names that directory (or a node id inside it) instead of a parent. See `tests/durable_exec/temporal/conftest.py` and the guard in `tests/durable_exec/test_temporal_skip_gating.py`
- Scrub `COVERAGE_*` from the environment of any test that spawns a Python subprocess (`env={k: v for k, v in os.environ.items() if not k.startswith('COVERAGE_')}`) — `[tool.coverage.run] patch = ["subprocess"]` otherwise starts coverage in the child, which costs a full instrumented interpreter start-up per spawn and can mark `pragma: no cover` import guards as covered, failing the `strict-no-cover` audit. See `tests/test_public_interface_contracts.py` and `tests/models/test_openai.py`

## Directory Structure

```
tests/
├── conftest.py              # shared fixtures
├── json_body_serializer.py  # custom VCR serializer
├── assets/                  # binary test files
├── cassettes/               # VCR recordings for root tests
├── models/
│   ├── conftest.py          # model-specific fixtures (if needed)
│   ├── cassettes/           # VCR recordings per test file
│   │   ├── test_openai/
│   │   ├── test_anthropic/
│   │   └── ...
│   └── test_*.py
├── providers/
│   └── test_*.py            # provider initialization tests (unit)
└── test_*.py                # feature tests (prefer VCR + parametrize)
```
