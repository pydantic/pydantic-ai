# Testing Guidelines

## Philosophy

1. **VCR-first**: prefer cassette-based integration tests over unit tests - provider APIs are the ultimate judges of correctness
2. **Centralized feature tests**: one file tests one feature across all providers (e.g., `test_multimodal_vcr.py`)
3. **Stacked `@pytest.mark.parametrize`**: use cartesian products for comprehensive coverage (provider x feature x config)
4. **Snapshots**: use `snapshot()` empty, fill via test run - never write contents manually

## Test File Structure

```python
from __future__ import annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
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

For snapshot-based expectations per case:

```python
from inline_snapshot import snapshot

EXPECTATIONS: dict[tuple[str, bool], str] = {
    ('openai', False): snapshot(),
    ('openai', True): snapshot(),
    ('anthropic', False): snapshot(),
    ('anthropic', True): snapshot(),
}

@pytest.mark.parametrize('model', ['openai', 'anthropic'], indirect=True)
@pytest.mark.parametrize('stream', [False, True])
async def test_feature_with_snapshots(model: Model, stream: bool, request: pytest.FixtureRequest):
    agent = Agent(model)
    result = await agent.run('hello')

    model_name = request.node.callspec.params['model']
    # snapshots auto-fill per parameter combination
    assert result.all_messages() == EXPECTATIONS[(model_name, stream)]
```

## VCR Workflow

> Use the `/pytest-vcr` skill for guided cassette recording, debugging, and cassette inspection workflows.

## Key Fixtures

### From `conftest.py`

#### Model requests
- `allow_model_requests` - bypasses the default `ALLOW_MODEL_REQUESTS = False`

#### API key and provider fixtures

API key fixtures are session-scoped and default to `'mock-api-key'` (real keys loaded from env when recording).
Provider fixtures handle auth setup for providers that need more than an API key.
See `tests/conftest.py` for the full list and the `model` fixture for supported `indirect=True` values.

#### The `model` fixture (use with `indirect=True`)
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

#### VCR configuration
- `vcr_config` - module-scoped, configures header filtering and localhost ignore

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

### Custom helpers
- `IsSameStr()` - asserts same string value across multiple uses in one assertion
  ```python
  assert events == [
      {'id': (msg_id := IsSameStr())},
      {'id': msg_id},  # must match first
  ]
  ```

## Anti-patterns

- **Don't** create provider-specific test files for new features - centralize in feature files
- **Don't** write snapshot contents manually - let tests fill them
- **Don't** use unit tests when VCR can cover the logic
- **Don't** instantiate providers inline - use fixtures

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
<!-- braindump: rules extracted from PR review patterns -->

# tests/ Guidelines

## Testing

<!-- rule:177 -->
- Test through public APIs, not private methods (prefixed with `_`) or helpers — Prevents brittle tests tied to implementation details, reduces maintenance burden when refactoring internals, and validates actual user-facing behavior rather than isolated units
<!-- rule:173 -->
- Maintain 1:1 correspondence between test files and source modules (`test_{module}.py`) — consolidate related tests instead of splitting by feature, config, or test type — Prevents test suite fragmentation and makes tests easier to locate by matching source structure; use fixtures/markers to distinguish test types within the file
<!-- rule:86 -->
- Use `snapshot()` for complex structured outputs (objects, message sequences, API responses, nested dicts, span attributes) — prevents brittle field-by-field assertions and improves test maintainability — Snapshot testing catches unexpected changes in complex structures more reliably than manual assertions, and `IsStr` matchers handle variable values gracefully
<!-- rule:318 -->
- Use `pytest-vcr` cassettes (not mocks) in `tests/models/` — records real HTTP interactions for deterministic replay, captures both success and error cases — Ensures integration tests validate real API behavior without live calls on every run, making tests faster and preventing flakiness from network issues or rate limits
<!-- rule:334 -->
- Assert meaningful behavior in tests, not just code execution or type checks — validates correctness and data flow — Prevents false confidence from tests that pass without verifying actual functionality works as intended
<!-- rule:194 -->
- In agent/model/stream tests, assert on final output AND snapshot `result.all_messages()` — validates complete execution trace, not just end result — Catches regressions in tool calls, intermediate steps, and message flow that final output assertions miss
<!-- rule:363 -->
- Test through real APIs, not mocks — mock only slow/external dependencies outside your control — Improves refactoring safety, documents real usage patterns, and catches integration issues — use lightweight local infrastructure (test servers, in-memory DBs) for systems you control (provider APIs, Temporal workflows, frameworks) in files like `test_{provider}.py`; reserve mocks for third-party HTTP APIs and unreliable external services
<!-- rule:11 -->
- Parametrize tests across all providers that support the feature (or at minimum OpenAI, Anthropic, Google) — catches provider-specific regressions and ensures cross-provider compatibility — Prevents breaking unchanged providers when modifying shared model logic, and surfaces integration issues across different provider APIs before they reach production
<!-- rule:385 -->
- Ensure test assertions match test names and docstrings — prevents false confidence in test coverage and catches actual regressions — Tests without proper assertions or that verify opposite behavior create false positives and fail to catch bugs they claim to prevent.
<!-- rule:89 -->
- Test both positive and negative cases for optional capabilities (model features, server features, streaming) — ensures features work when supported AND fail gracefully when absent — Prevents false confidence from tests that only check unsupported cases, catching both implementation bugs and missing error handling
<!-- rule:630 -->
- Test MCP against real `tests.mcp_server` instance, not mocks — extend test server with helper tools to expose runtime context (instructions, client info, session state) — Verifies actual data flow and integration behavior rather than just testing mock interfaces, catching real-world issues that mocks would miss

## General

<!-- rule:463 -->
- Remove stale test docstrings, comments, and historical provider bug notes when behavior changes — Outdated test documentation misleads developers about what's actually being tested and why

<!-- /braindump -->
