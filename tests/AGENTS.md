# Testing Guidelines

## Philosophy

1. **VCR-first**: prefer cassette-based integration tests over unit tests - provider APIs are the ultimate judges of correctness
2. **Centralized feature tests**: one file tests one feature across all providers (e.g., `test_multimodal_vcr.py`)
3. **Stacked `@pytest.mark.parametrize`**: use cartesian products for comprehensive coverage (provider x feature x config)
4. **Snapshots**: use `snapshot({})` empty, fill via test run - never write contents manually

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

> **Tip:** Use the `/pytest-vcr` skill for guided cassette recording and debugging workflows.

**Recording new cassettes:**
```bash
source .env && uv run pytest tests/path/to/test.py::test_name --record-mode=new_episodes
```

**Rewriting existing cassettes:**
```bash
source .env && uv run pytest tests/path/to/test.py::test_name --record-mode=rewrite
```

**Verification (fills snapshots):**
```bash
uv run pytest tests/path/to/test.py::test_name -v
git diff tests/
```

**Useful flags:**
- `--lf` - run only last failed tests
- `--tb=line` - short traceback output
- `-k="pattern"` - run tests matching substring

**Rules:**
- Always verify playback after recording
- Review cassette diffs before committing

## Parsing Cassettes

Parse VCR cassette YAML files to inspect request/response bodies:

```bash
# Parse all interactions
uv run python .claude/skills/pytest-vcr/parse_cassette.py tests/models/cassettes/test_foo/test_bar.yaml

# Parse specific interaction (0-indexed)
uv run python .claude/skills/pytest-vcr/parse_cassette.py tests/models/cassettes/test_foo/test_bar.yaml --interaction 1
```

Shows request (method, URI, body) and response (status, body) with truncated base64 for readability.

## Key Fixtures

### From `conftest.py`

**Model requests:**
- `allow_model_requests` - bypasses the default `ALLOW_MODEL_REQUESTS = False`

**API keys (session-scoped, default to `'mock-api-key'`):**
- `openai_api_key`
- `anthropic_api_key`
- `gemini_api_key`
- `groq_api_key`
- `mistral_api_key`
- `co_api_key`
- `deepseek_api_key`
- `openrouter_api_key`
- `huggingface_api_key`
- `heroku_inference_key`
- `cerebras_api_key`
- `xai_api_key`
- `voyage_api_key`

**The `model` fixture (use with `indirect=True`):**
```python
@pytest.mark.parametrize('model', ['openai', 'anthropic'], indirect=True)
async def test_something(model: Model):
    ...
```

Supported values:
- `'test'` - `TestModel()` for deterministic testing
- `'openai'` - `OpenAIChatModel('o3-mini')`
- `'anthropic'` - `AnthropicModel('claude-sonnet-4-5')`
- `'mistral'` - `MistralModel('ministral-8b-latest')`
- `'groq'` - `GroqModel('llama3-8b-8192')`
- `'cohere'` - `CohereModel('command-r-plus')`
- `'gemini'` - `GeminiModel('gemini-1.5-flash')` (deprecated)
- `'google'` - `GoogleModel('gemini-1.5-flash')`
- `'bedrock'` - `BedrockConverseModel('us.amazon.nova-micro-v1:0')`
- `'huggingface'` - `HuggingFaceModel('Qwen/Qwen2.5-72B-Instruct')`
- `'outlines'` - `OutlinesModel(...)` with local transformers

**Provider fixtures:**
- `bedrock_provider` - session-scoped, handles AWS auth
- `vertex_provider` - function-scoped, requires `vertex_provider_auth`
- `xai_provider` - function-scoped, uses protobuf cassettes

**Environment management:**
- `env` - `TestEnv` instance for temporary env var changes
  ```python
  def test_missing_key(env: TestEnv):
      env.remove('OPENAI_API_KEY')
      with pytest.raises(UserError):
          ...
  ```

**Binary content (session-scoped):**
- `assets_path` - `Path` to `tests/assets/`
- `image_content` - `BinaryImage` (kiwi.jpg)
- `audio_content` - `BinaryContent` (marcelo.mp3)
- `video_content` - `BinaryContent` (small_video.mp4)
- `document_content` - `BinaryContent` (dummy.pdf)
- `text_document_content` - `BinaryContent` (dummy.txt)

**VCR configuration:**
- `vcr_config` - module-scoped, configures header filtering and localhost ignore

## Assertion Helpers

**From `dirty_equals` (re-exported in conftest):**
- `IsNow(tz=timezone.utc)` - datetime within 10 seconds of now
- `IsStr()` - any string, supports `regex=r'...'`
- `IsDatetime()` - any datetime
- `IsBytes()` - any bytes
- `IsInt()` - any int
- `IsFloat()` - any float
- `IsList()` - any list
- `IsInstance(SomeClass)` - instance of class

**Custom helpers:**
- `IsSameStr()` - asserts same string value across multiple uses in one assertion
  ```python
  assert events == [
      {'id': (msg_id := IsSameStr())},
      {'id': msg_id},  # must match first
  ]
  ```

**Snapshots:**
```python
from inline_snapshot import snapshot

# always write empty, run test to fill
assert result == snapshot()

# after test run, becomes:
assert result == snapshot({'key': 'value'})
```

## Custom VCR Serializer

The `json_body_serializer.py` handles:
- Smart quote normalization (curly quotes → ASCII)
- Sensitive header scrubbing (authorization, api keys, dates)
- Token scrubbing (`access_token` → `'scrubbed'`)
- Response decompression (gzip, brotli)
- Literal YAML style for readable multi-line strings

## Anti-patterns

- **Don't** create provider-specific test files for new features - centralize in feature files
- **Don't** write snapshot contents manually - let tests fill them
- **Don't** use `-v` when recording cassettes
- **Don't** use unit tests when VCR can cover the logic
- **Don't** instantiate providers inline - use fixtures
- **Don't** skip `allow_model_requests` fixture for tests hitting APIs

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
