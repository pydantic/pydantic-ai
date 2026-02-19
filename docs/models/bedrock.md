# Bedrock

## Install

To use `BedrockConverseModel`, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `bedrock` optional group:

```bash
pip/uv-add "pydantic-ai-slim[bedrock]"
```

## Configuration

To use [AWS Bedrock](https://aws.amazon.com/bedrock/), you'll need an AWS account with Bedrock enabled and appropriate credentials. You can use either AWS credentials directly or a pre-configured boto3 client.

`BedrockModelName` contains a list of available Bedrock models, including models from Anthropic, Amazon, Cohere, Meta, and Mistral.

## Environment variables

You can set your AWS credentials as environment variables ([among other options](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html#using-environment-variables)):

```bash
export AWS_BEARER_TOKEN_BEDROCK='your-api-key'
# or:
export AWS_ACCESS_KEY_ID='your-access-key'
export AWS_SECRET_ACCESS_KEY='your-secret-key'
export AWS_DEFAULT_REGION='us-east-1'  # or your preferred region
```

You can then use `BedrockConverseModel` by name:

```python
from pydantic_ai import Agent

agent = Agent('bedrock:anthropic.claude-sonnet-4-5-20250929-v1:0')
...
```

Or initialize the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel

model = BedrockConverseModel('anthropic.claude-sonnet-4-5-20250929-v1:0')
agent = Agent(model)
...
```

## Customizing Bedrock Runtime API

You can customize the Bedrock Runtime API calls by adding additional parameters, such as [guardrail
configurations](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html) and [performance settings](https://docs.aws.amazon.com/bedrock/latest/userguide/latency-optimized-inference.html). For a complete list of configurable parameters, refer to the
documentation for [`BedrockModelSettings`][pydantic_ai.models.bedrock.BedrockModelSettings].

```python {title="customize_bedrock_model_settings.py"}
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings

# Define Bedrock model settings with guardrail and performance configurations
bedrock_model_settings = BedrockModelSettings(
    bedrock_guardrail_config={
        'guardrailIdentifier': 'v1',
        'guardrailVersion': 'v1',
        'trace': 'enabled'
    },
    bedrock_performance_configuration={
        'latency': 'optimized'
    }
)


model = BedrockConverseModel(model_name='us.amazon.nova-pro-v1:0')

agent = Agent(model=model, model_settings=bedrock_model_settings)
```

## Prompt Caching

Bedrock supports [prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) on Anthropic models so you can reuse expensive context across requests. Pydantic AI provides four ways to use prompt caching:

1. **Cache User Messages with [`CachePoint`][pydantic_ai.messages.CachePoint]**: Insert a `CachePoint` marker to cache everything before it in the current user message.
2. **Cache System Instructions**: Enable [`BedrockModelSettings.bedrock_cache_instructions`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_cache_instructions] to append a cache point after the system prompt.
3. **Cache Tool Definitions**: Enable [`BedrockModelSettings.bedrock_cache_tool_definitions`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_cache_tool_definitions] to cache your tool schemas.
4. **Cache All Messages**: Set [`BedrockModelSettings.bedrock_cache_messages`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_cache_messages] to `True` to automatically cache the last user message.

!!! note "No TTL Support"
    Unlike the direct Anthropic API, Bedrock manages cache TTL automatically. All cache settings are boolean only — no `'5m'` or `'1h'` options.

!!! note "Minimum Token Threshold"
    AWS only serves cached content once a segment crosses the provider-specific minimum token thresholds (see the [Bedrock prompt caching docs](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)). Short prompts or tool definitions below those limits will bypass the cache, so don't expect savings for tiny payloads.

### Example 1: Automatic Message Caching

Use `bedrock_cache_messages` to automatically cache the last user message:

```python {test="skip"}
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockModelSettings

agent = Agent(
    'bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    system_prompt='You are a helpful assistant.',
    model_settings=BedrockModelSettings(
        bedrock_cache_messages=True,  # Automatically caches the last message
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
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings

model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
agent = Agent(
    model,
    system_prompt='Detailed instructions...',
    model_settings=BedrockModelSettings(
        bedrock_cache_instructions=True,       # Cache system instructions
        bedrock_cache_tool_definitions=True,   # Cache tool definitions
        bedrock_cache_messages=True,           # Also cache the last message
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
    'bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    system_prompt='Instructions...',
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

Access cache usage statistics via [`RequestUsage`][pydantic_ai.usage.RequestUsage]:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint

agent = Agent('bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0')


async def main():
    result = await agent.run(
        [
            'Reference material...',
            CachePoint(),
            'What changed since last time?',
        ]
    )
    usage = result.usage()
    print(f'Cache writes: {usage.cache_write_tokens}')
    print(f'Cache reads: {usage.cache_read_tokens}')
```

### Cache Point Limits

Bedrock enforces a maximum of 4 cache points per request. Pydantic AI automatically manages this limit to ensure your requests always comply without errors.

#### How Cache Points Are Allocated

Cache points can be placed in three locations:

1. **System Prompt**: Via `bedrock_cache_instructions` setting (adds cache point to last system prompt block)
2. **Tool Definitions**: Via `bedrock_cache_tool_definitions` setting (adds cache point to last tool definition)
3. **Messages**: Via `CachePoint` markers or `bedrock_cache_messages` setting (adds cache points to message content)

Each setting uses **at most 1 cache point**, but you can combine them.

#### Automatic Cache Point Limiting

When cache points from all sources (settings + `CachePoint` markers) exceed 4, Pydantic AI automatically removes excess cache points from **older message content** (keeping the most recent ones).

```python {test="skip"}
from pydantic_ai import Agent, CachePoint
from pydantic_ai.models.bedrock import BedrockModelSettings

agent = Agent(
    'bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0',
    system_prompt='Instructions...',
    model_settings=BedrockModelSettings(
        bedrock_cache_instructions=True,      # 1 cache point
        bedrock_cache_tool_definitions=True,  # 1 cache point
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
```

**Key Points**:

- System and tool cache points are **always preserved**
- The cache point created by `bedrock_cache_messages` is **always preserved** (as it's the newest message cache point)
- Additional `CachePoint` markers in messages are removed from oldest to newest when the limit is exceeded
- This ensures critical caching (instructions/tools) is maintained while still benefiting from message-level caching

## `provider` argument

You can provide a custom `BedrockProvider` via the `provider` argument. This is useful when you want to specify credentials directly or use a custom boto3 client:

```python
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Using AWS credentials directly
model = BedrockConverseModel(
    'anthropic.claude-sonnet-4-5-20250929-v1:0',
    provider=BedrockProvider(
        region_name='us-east-1',
        aws_access_key_id='your-access-key',
        aws_secret_access_key='your-secret-key',
    ),
)
agent = Agent(model)
...
```

You can also pass a pre-configured boto3 client:

```python
import boto3

from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Using a pre-configured boto3 client
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
model = BedrockConverseModel(
    'anthropic.claude-sonnet-4-5-20250929-v1:0',
    provider=BedrockProvider(bedrock_client=bedrock_client),
)
agent = Agent(model)
...
```

## Using AWS Application Inference Profiles

AWS Bedrock supports [custom application inference profiles](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-create.html) for cost tracking and resource management. When using these profiles, you should specify a [model profile](https://ai.pydantic.dev/models/overview/#models-and-providers) to ensure Pydantic AI can correctly identify model capabilities (streaming, tool use, caching, etc.) while still using the custom inference profile for cost tracking.

Without explicit configuration, an inference profile ARN like `arn:aws:bedrock:us-east-2:*****:application-inference-profile/****` doesn't contain enough information for Pydantic AI to determine the underlying model. You can work around this by:

1. Passing the inference profile ARN as the model name to [`BedrockConverseModel`][pydantic_ai.models.bedrock.BedrockConverseModel]
2. Using the `profile` parameter to specify the logical model name for feature detection

```python
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Create provider with your AWS configuration
provider = BedrockProvider(region_name='us-east-2')

# Create a profile with the logical model name for feature detection
profile = provider.model_profile('us.anthropic.claude-opus-4-5-20251101-v1:0')

# Pass the inference profile ARN as the model name
model = BedrockConverseModel(
    'arn:aws:bedrock:us-east-2:123456789012:application-inference-profile/my-profile',
    provider=provider,
    profile=profile,  # Provides the logical model name for feature detection
)

agent = Agent(model)
```

## Configuring Retries

Bedrock uses boto3's built-in retry mechanisms. You can configure retry behavior by passing a custom boto3 client with retry settings:

```python
import boto3
from botocore.config import Config

from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Configure retry settings
config = Config(
    retries={
        'max_attempts': 5,
        'mode': 'adaptive'  # Recommended for rate limiting
    }
)

bedrock_client = boto3.client(
    'bedrock-runtime',
    region_name='us-east-1',
    config=config
)

model = BedrockConverseModel(
    'us.amazon.nova-micro-v1:0',
    provider=BedrockProvider(bedrock_client=bedrock_client),
)
agent = Agent(model)
```

### Retry Modes

- `'legacy'` (default): 5 attempts, basic retry behavior
- `'standard'`: 3 attempts, more comprehensive error coverage
- `'adaptive'`: 3 attempts with client-side rate limiting (recommended for handling `ThrottlingException`)

For more details on boto3 retry configuration, see the [AWS boto3 documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/retries.html).

!!! note
    Unlike other providers that use httpx for HTTP requests, Bedrock uses boto3's native retry mechanisms. The retry strategies described in [HTTP Request Retries](../retries.md) do not apply to Bedrock.
