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


@pytest.mark.parametrize('provider', ['openai', 'anthropic', 'google'], indirect=True)
@pytest.mark.parametrize('stream', [False, True])
async def test_feature(provider: str, stream: bool):
    ...
```

## Parametrization with Expectations

For cartesian product tests, use a dict to map parameter combinations to expected results:

```python
# expectation can be a dataclass, for more complex cases
EXPECTATIONS: dict[tuple[str, bool], str] = {
    ('openai', False): 'expected output for openai non-streaming',
    ('openai', True): 'expected output for openai streaming',
    ('anthropic', False): 'expected output for anthropic non-streaming',
    ('anthropic', True): 'expected output for anthropic streaming',
}


@pytest.mark.parametrize('provider', ['openai', 'anthropic'], indirect=True)
@pytest.mark.parametrize('stream', [False, True])
async def test_feature(provider: str, stream: bool, request: pytest.FixtureRequest, vcr: Cassette):
    """What the test is asserting.

    Use the `request` fixture to access test parameter values.

    Use the `vcr` to make assertions about the HTTP requests if needed.
    Another creative way of, for instance, asserting headers, is to use a patched httpx client fixture.
    This spares us the overhead of parsing cassette fields, so it is to be preferred whenever optimal.
    """
    provider_name = request.node.callspec.params['provider']
    expected = EXPECTATIONS[(provider_name, stream)]

    agent = Agent(provider)
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
    (provider, stream): snapshot(),
}

@pytest.mark.parametrize('provider', ['openai', 'anthropic'], indirect=True)
@pytest.mark.parametrize('stream', [False, True])
async def test_feature_with_snapshots(provider: str, stream: bool):
    agent = Agent(provider)
    result = await agent.run('hello')

    # snapshots auto-fill per parameter combination
    assert result.all_messages() == expected
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

## Assertion Helpers

#### From `conftest.py`
- `IsNow(tz=timezone.utc)` - datetime within 10 seconds of now
- `IsStr()` - any string, supports `regex=r'...'`
- `IsDatetime()` - any datetime
- `IsBytes()` - any bytes
- `IsInt()` - any int
- `IsFloat()` - any float
- `IsList()` - any list
- `IsInstance(SomeClass)` - instance of class

#### Custom helpers
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
