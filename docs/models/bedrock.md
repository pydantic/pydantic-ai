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

agent = Agent('bedrock:anthropic.claude-3-sonnet-20240229-v1:0')
...
```

Or initialize the model directly with just the model name:

```python
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel

model = BedrockConverseModel('anthropic.claude-3-sonnet-20240229-v1:0')
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

Bedrock supports [prompt caching](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html) on Anthropic models so you can reuse expensive context across requests. Pydantic AI exposes the same three strategies as Anthropic:

1. **Cache User Messages with [`CachePoint`][pydantic_ai.messages.CachePoint]**: Insert a `CachePoint` marker to cache everything before it in the current user message.
2. **Cache System Instructions**: Enable [`BedrockModelSettings.bedrock_cache_instructions`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_cache_instructions] to append a cache point after the system prompt.
3. **Cache Tool Definitions**: Enable [`BedrockModelSettings.bedrock_cache_tool_definitions`][pydantic_ai.models.bedrock.BedrockModelSettings.bedrock_cache_tool_definitions] to cache your tool schemas.

> [!NOTE]
> AWS only serves cached content once a segment crosses the provider-specific minimum token thresholds (see the [Bedrock prompt caching docs](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)). Short prompts or tool definitions below those limits will bypass the cache, so don't expect savings for tiny payloads.

You can combine all of them:

```python {test="skip"}
from pydantic_ai import Agent, CachePoint, RunContext
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings

model = BedrockConverseModel('us.anthropic.claude-sonnet-4-5-20250929-v1:0')
agent = Agent(
    model,
    system_prompt='Detailed instructions...',
    model_settings=BedrockModelSettings(
        bedrock_cache_instructions=True,
        bedrock_cache_tool_definitions=True,
    ),
)


@agent.tool
async def search_docs(ctx: RunContext, query: str) -> str:
    return f'Results for {query}'


async def main():
    result1 = await agent.run(
        [
            'Long cached context...',
            CachePoint(),
            'First question',
        ]
    )
    result2 = await agent.run(
        [
            'Long cached context...',
            CachePoint(),
            'Second question',
        ]
    )
    print(result1.output, result1.usage())
    print(result2.output, result2.usage())
```

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

## `provider` argument

You can provide a custom `BedrockProvider` via the `provider` argument. This is useful when you want to specify credentials directly or use a custom boto3 client:

```python
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel
from pydantic_ai.providers.bedrock import BedrockProvider

# Using AWS credentials directly
model = BedrockConverseModel(
    'anthropic.claude-3-sonnet-20240229-v1:0',
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
    'anthropic.claude-3-sonnet-20240229-v1:0',
    provider=BedrockProvider(bedrock_client=bedrock_client),
)
agent = Agent(model)
...
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
