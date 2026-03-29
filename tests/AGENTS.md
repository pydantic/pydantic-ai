# Testing Guidelines

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

## VCR Workflow

Record cassettes with `--record-mode=rewrite`, verify playback without the flag, and review diffs.
For detailed workflows see `.agents/skills/pytest-vcr/SKILL.md`.

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
- Test both positive and negative cases for optional capabilities (model features, server features, streaming) — ensures features work when supported AND fail gracefully when absent
- Ensure test assertions match test names and docstrings — tests without proper assertions or that verify opposite behavior create false positives
- Test MCP against real `tests.mcp_server` instance, not mocks — extend test server with helper tools to expose runtime context (instructions, client info, session state)
- Remove stale test docstrings, comments, and historical provider bug notes when behavior changes

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
