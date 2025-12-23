# Anthropic

## Install

To use `AnthropicModel` models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `anthropic` optional group:

```bash
pip/uv-add "pydantic-ai-slim[anthropic]"
```

## Configuration

To use [Anthropic](https://anthropic.com) through their API, go to [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys) to generate an API key.

`AnthropicModelName` contains a list of available Anthropic models.

## Environment variable

Once you have the API key, you can set it as an environment variable:

```bash
export ANTHROPIC_API_KEY='your-api-key'
```

You can then use `AnthropicModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('anthropic:claude-sonnet-4-5')
...
```

Or initialise the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel

model = AnthropicModel('claude-sonnet-4-5')
agent = Agent(model)
...
```

## `provider` argument

You can provide a custom `Provider` via the `provider` argument:

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

model = AnthropicModel(
    'claude-sonnet-4-5', provider=AnthropicProvider(api_key='your-api-key')
)
agent = Agent(model)
...
```

## Custom HTTP Client

You can customize the `AnthropicProvider` with a custom `httpx.AsyncClient`:

```python
from httpx import AsyncClient

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

custom_http_client = AsyncClient(timeout=30)
model = AnthropicModel(
    'claude-sonnet-4-5',
    provider=AnthropicProvider(api_key='your-api-key', http_client=custom_http_client),
)
agent = Agent(model)
...
```

## Cloud Platform Integrations

You can use Anthropic models through cloud platforms by passing a custom client to [`AnthropicProvider`][pydantic_ai.providers.anthropic.AnthropicProvider].

### AWS Bedrock

To use Claude models via [AWS Bedrock](https://aws.amazon.com/bedrock/claude/), follow the [Anthropic documentation](https://docs.anthropic.com/en/api/claude-on-amazon-bedrock) on how to set up an `AsyncAnthropicBedrock` client and then pass it to `AnthropicProvider`:

```python {test="skip"}
from anthropic import AsyncAnthropicBedrock

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

bedrock_client = AsyncAnthropicBedrock()  # Uses AWS credentials from environment
provider = AnthropicProvider(anthropic_client=bedrock_client)
model = AnthropicModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0', provider=provider)
agent = Agent(model)
...
```

!!! note "Bedrock vs BedrockConverseModel"
    This approach uses Anthropic's SDK with AWS Bedrock credentials. For an alternative using AWS SDK (boto3) directly, see [`BedrockConverseModel`](bedrock.md).

### Google Vertex AI

To use Claude models via [Google Cloud Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/use-claude), follow the [Anthropic documentation](https://docs.anthropic.com/en/api/claude-on-vertex-ai) on how to set up an `AsyncAnthropicVertex` client and then pass it to `AnthropicProvider`:

```python {test="skip"}
from anthropic import AsyncAnthropicVertex

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

vertex_client = AsyncAnthropicVertex(region='us-east5', project_id='your-project-id')
provider = AnthropicProvider(anthropic_client=vertex_client)
model = AnthropicModel('claude-sonnet-4-5', provider=provider)
agent = Agent(model)
...
```

!!! note "Vertex vs GoogleModel"
    This approach uses Anthropic's SDK with Vertex AI credentials. For an alternative using Google's Vertex AI SDK directly, see [`GoogleModel`](google.md).

### Microsoft Foundry

To use Claude models via [Microsoft Foundry](https://ai.azure.com/), follow the [Anthropic documentation](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry) on how to set up an `AsyncAnthropicFoundry` client and then pass it to `AnthropicProvider`:

```python {test="skip"}
from anthropic import AsyncAnthropicFoundry

from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

foundry_client = AsyncAnthropicFoundry(
    api_key='your-foundry-api-key',  # Or set ANTHROPIC_FOUNDRY_API_KEY
    resource='your-resource-name',
)
provider = AnthropicProvider(anthropic_client=foundry_client)
model = AnthropicModel('claude-sonnet-4-5', provider=provider)
agent = Agent(model)
...
```

See [Anthropic's Microsoft Foundry documentation](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry) for setup instructions including Entra ID authentication.

## Prompt Caching

Anthropic supports [prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) to reduce costs by caching parts of your prompts. Pydantic AI provides four ways to use prompt caching:

1. **Cache User Messages with [`CachePoint`][pydantic_ai.messages.CachePoint]**: Insert a `CachePoint` marker in your user messages to cache everything before it
2. **Cache System Instructions**: Set [`AnthropicModelSettings.anthropic_cache_instructions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_instructions] to `True` (uses 5m TTL by default) or specify `'5m'` / `'1h'` directly
3. **Cache Tool Definitions**: Set [`AnthropicModelSettings.anthropic_cache_tool_definitions`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_tool_definitions] to `True` (uses 5m TTL by default) or specify `'5m'` / `'1h'` directly
4. **Cache All Messages**: Set [`AnthropicModelSettings.anthropic_cache_messages`][pydantic_ai.models.anthropic.AnthropicModelSettings.anthropic_cache_messages] to `True` to automatically cache all messages

!!! note "Amazon Bedrock"
    When using `AsyncAnthropicBedrock`, the TTL parameter is automatically omitted from all cache control settings (including `CachePoint`, `anthropic_cache_instructions`, `anthropic_cache_tool_definitions`, and `anthropic_cache_messages`) because Bedrock doesn't support explicit TTL.

### Example 1: Automatic Message Caching

Use `anthropic_cache_messages` to automatically cache all messages up to and including the newest user message:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='You are a helpful assistant.',
    model_settings=AnthropicModelSettings(
        anthropic_cache_messages=True,  # Automatically caches the last message
    ),
)

# The last message is automatically cached - no need for manual CachePoint
result1 = agent.run_sync('What is the capital of France?')

# Subsequent calls with similar conversation benefit from cache
result2 = agent.run_sync('What is the capital of Germany?')
print(f'Cache write: {result1.usage().cache_write_tokens}')
print(f'Cache read: {result2.usage().cache_read_tokens}')
```

### Example 2: Comprehensive Caching Strategy

Combine multiple cache settings for maximum savings:

```python {test="skip"}
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Detailed instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions=True,      # Cache system instructions
        anthropic_cache_tool_definitions='1h',  # Cache tool definitions with 1h TTL
        anthropic_cache_messages=True,          # Also cache the last message
    ),
)

@agent.tool
def search_docs(ctx: RunContext, query: str) -> str:
    """Search documentation."""
    return f'Results for {query}'


result = agent.run_sync('Search for Python best practices')
print(result.output)
```

### Example 3: Fine-Grained Control with CachePoint

Use manual `CachePoint` markers to control cache locations precisely:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Instructions...',
)

# Manually control cache points for specific content blocks
result = agent.run_sync([
    'Long context from documentation...',
    CachePoint(),  # Cache everything up to this point
    'First question'
])
print(result.output)
```

### Accessing Cache Usage Statistics

Access cache usage statistics via `result.usage()`:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions=True  # Default 5m TTL
    ),
)

result = agent.run_sync('Your question')
usage = result.usage()
print(f'Cache write tokens: {usage.cache_write_tokens}')
print(f'Cache read tokens: {usage.cache_read_tokens}')
```

### Cache Point Limits

Anthropic enforces a maximum of 4 cache points per request. Pydantic AI automatically manages this limit to ensure your requests always comply without errors.

#### How Cache Points Are Allocated

Cache points can be placed in three locations:

1. **System Prompt**: Via `anthropic_cache_instructions` setting (adds cache point to last system prompt block)
2. **Tool Definitions**: Via `anthropic_cache_tool_definitions` setting (adds cache point to last tool definition)
3. **Messages**: Via `CachePoint` markers or `anthropic_cache_messages` setting (adds cache points to message content)

Each setting uses **at most 1 cache point**, but you can combine them.

#### Example: Using All 3 Cache Point Sources

Define an agent with all cache settings enabled:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Detailed instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions=True,      # 1 cache point
        anthropic_cache_tool_definitions=True,  # 1 cache point
        anthropic_cache_messages=True,          # 1 cache point
    ),
)

@agent.tool_plain
def my_tool() -> str:
    return 'result'


# This uses 3 cache points (instructions + tools + last message)
# You can add 1 more CachePoint marker before hitting the limit
result = agent.run_sync([
    'Context', CachePoint(),  # 4th cache point - OK
    'Question'
])
print(result.output)
usage = result.usage()
print(f'Cache write tokens: {usage.cache_write_tokens}')
print(f'Cache read tokens: {usage.cache_read_tokens}')
```

#### Automatic Cache Point Limiting

When cache points from all sources (settings + `CachePoint` markers) exceed 4, Pydantic AI automatically removes excess cache points from **older message content** (keeping the most recent ones).

Define an agent with 2 cache points from settings:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint
from pydantic_ai.models.anthropic import AnthropicModelSettings

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    instructions='Instructions...',
    model_settings=AnthropicModelSettings(
        anthropic_cache_instructions=True,      # 1 cache point
        anthropic_cache_tool_definitions=True,  # 1 cache point
    ),
)

@agent.tool_plain
def search() -> str:
    return 'data'

# Already using 2 cache points (instructions + tools)
# Can add 2 more CachePoint markers (4 total limit)
result = agent.run_sync([
    'Context 1', CachePoint(),  # Oldest - will be removed
    'Context 2', CachePoint(),  # Will be kept (3rd point)
    'Context 3', CachePoint(),  # Will be kept (4th point)
    'Question'
])
# Final cache points: instructions + tools + Context 2 + Context 3 = 4
print(result.output)
usage = result.usage()
print(f'Cache write tokens: {usage.cache_write_tokens}')
print(f'Cache read tokens: {usage.cache_read_tokens}')
```

**Key Points**:
- System and tool cache points are **always preserved**
- The cache point created by `anthropic_cache_messages` is **always preserved** (as it's the newest message cache point)
- Additional `CachePoint` markers in messages are removed from oldest to newest when the limit is exceeded
- This ensures critical caching (instructions/tools) is maintained while still benefiting from message-level caching
