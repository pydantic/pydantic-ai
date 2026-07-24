# Testing Guidelines

## Testing philosophy

VCR + public-API tests are the default. We test through the public API the way a user would (`Agent(...)`, `agent.run(...)`) against real provider responses recorded as cassettes — provider APIs are the ultimate judge of whether the code is correct when run as intended, and that user-facing correctness is what we care about, not behavior in isolated units.

Unit tests still earn their place — for internal behavior that is definitory and worth pinning against drift. That includes behavior you can't reach or reliably trigger through the public API (pre-request guards, defensive branches no real model produces), but also behavior a VCR test wouldn't actually protect: our cassette matchers aren't always sensitive to the request body, so a changed internal payload can still match an existing recording and pass green — a unit test asserting the internal shape directly is what catches that regression. Each unit test should still say why it isn't (or can't be) a VCR test.

Recording cassettes needs provider API keys and isn't trivial, so contributors routinely under-test the real behavior — writing the VCR test a contributor couldn't is core maintainer work.

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
    Another creative way of, for instance, asserting headers, is to use a patched httpx client fixture.
    This spares us the overhead of parsing cassette fields, so it is to be preferred whenever optimal.
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

## Key Fixtures

### From `conftest.py`

#### Model requests
- `allow_model_requests` - bypasses the default `ALLOW_MODEL_REQUESTS = False`

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
- Don't blanket-suppress `reportDeprecated` at file scope — no whole-file `# pyright: ignore` (or `# pyright: basic`) at the top of a file to kill every deprecation static-warning at once, since that also silences the next *unintended* use of a deprecated symbol in the file. Put `# pyright: ignore[reportDeprecated]` on each intentional call instead. (The runtime side is different and conventional: a module-level `warnings.filterwarnings('ignore', message=...)` scoped to one deprecation message, in a file that pervasively constructs the deprecated symbol it tests — see `test_temporal.py` / `test_prefect.py` — is fine, as long as a dedicated `pytest.warns` test still proves the deprecation fires.)

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
