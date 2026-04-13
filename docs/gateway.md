---
title: Pydantic AI Gateway
status: new
---

# Pydantic AI Gateway

**[Pydantic AI Gateway](https://logfire.pydantic.dev/)** is a unified interface for accessing multiple AI providers with a single key, managed through [Pydantic Logfire](https://logfire.pydantic.dev/). Features include built-in OpenTelemetry observability, real-time cost monitoring, failover management, and native integration with the other tools in the [Pydantic stack](https://pydantic.dev/).

!!! warning "Migrated to Pydantic Logfire"
    The AI Gateway has moved from `gateway.pydantic.dev` to [Pydantic Logfire](https://logfire.pydantic.dev/). If you were using the standalone gateway, see [Pydantic AI Gateway is Moving to Pydantic Logfire](https://logfire.pydantic.dev/docs/gateway-migration/).

Sign up at [logfire.pydantic.dev](https://logfire.pydantic.dev/).

!!! question "Questions?"
    For questions and feedback, contact us on [Slack](https://logfire.pydantic.dev/docs/join-slack/).

## Documentation Integration

To help you get started with Pydantic AI Gateway, some code examples on the Pydantic AI documentation include a "Via Pydantic AI Gateway" tab, alongside a "Direct to Provider API" tab with the standard Pydantic AI model string. The main difference between them is that when using Gateway, model strings use the `gateway/` prefix.

## Key features

- **API key management**: Access multiple LLM providers with a single Gateway key.
- **Cost Limits**: Set spending limits at project, user, and API key levels with daily, weekly, and monthly caps.
- **BYOK and managed providers:** Bring your own API keys (BYOK) from LLM providers, or pay for inference directly through the platform.
- **Multi-provider support:** Access models from OpenAI, Anthropic, Google Vertex, Groq, and AWS Bedrock. _More providers coming soon_.
- **Backend observability:** Log every request through [Pydantic Logfire](https://pydantic.dev/logfire) or any OpenTelemetry backend (_coming soon_).
- **Zero translation**: Unlike traditional AI gateways that translate everything to one common schema, **Pydantic AI Gateway** allows requests to flow through directly in each provider's native format. This gives you immediate access to new model features as soon as they are released.
- **Enterprise ready**: Inherits Logfire's enterprise features — including SSO, custom roles and permissions.

```python {title="hello_world.py"}
from pydantic_ai import Agent

agent = Agent('gateway/openai:gpt-5.2')

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

## Quick Start

This section contains instructions on how to set up your account and run your app with Pydantic AI Gateway credentials.

### Create an account

1. Sign up at [logfire.pydantic.dev](https://logfire.pydantic.dev/)
2. Choose a region and create an account.
3. Activate the gateway in your organizations settings.

### Create Gateway API keys

Go to your organization's Gateway settings in Logfire and create an API key.

## Usage

After setting up your account with the instructions above, you will be able to make an AI model request with the Pydantic AI Gateway.
The code snippets below show how you can use Pydantic AI Gateway with different frameworks and SDKs.

To use different models, change the model string `gateway/<api_format>:<model_name>` to other models offered by the supported providers.

Examples of providers and models that can be used are:

| **Provider** | **API Format**  | **Example Model**                        |
| --- |-----------------|------------------------------------------|
| OpenAI | `openai`        | `gateway/openai:gpt-5.2`                 |
| Anthropic | `anthropic`     | `gateway/anthropic:claude-sonnet-4-6`    |
| Google Vertex | `google-vertex` | `gateway/google-vertex:gemini-3-flash-preview` |
| Groq | `groq`          | `gateway/groq:openai/gpt-oss-120b`       |
| AWS Bedrock | `bedrock`       | `gateway/bedrock:amazon.nova-micro-v1:0` |

### Pydantic AI

Before you start, make sure you are on version 1.16 or later of `pydantic-ai`. To update to the latest version run:

=== "uv"

    ```bash
    uv sync -P pydantic-ai
    ```

=== "pip"

    ```bash
    pip install -U pydantic-ai
    ```

Set the `PYDANTIC_AI_GATEWAY_API_KEY` environment variable to your Gateway API key:

```bash
export PYDANTIC_AI_GATEWAY_API_KEY="pylf_v..."
```

You can access multiple models with the same API key, as shown in the code snippet below.

```python {title="hello_world.py"}
from pydantic_ai import Agent

agent = Agent('gateway/openai:gpt-5.2')

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

#### Passing API Key directly

Pass your API key directly using the [`gateway_provider`][pydantic_ai.providers.gateway.gateway_provider]:

```python {title="passing_api_key.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.gateway import gateway_provider

provider = gateway_provider('openai', api_key='pylf_v...')
model = OpenAIChatModel('gpt-5.2', provider=provider)
agent = Agent(model)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

#### Using a different upstream provider

To use an alternate provider or routing group, you can specify it in the route parameter:

```python {title="routing_via_provider.py"}
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.gateway import gateway_provider

provider = gateway_provider(
    'openai',
    api_key='pylf_v...',
    route='builtin-openai'
)
model = OpenAIChatModel('gpt-5.2', provider=provider)
agent = Agent(model)

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```

### Claude Code

Before you start, log out of Claude Code using `/logout`.

Set your gateway credentials as environment variables, using the base URL that matches your Logfire region:

=== "US"

    ```bash
    export ANTHROPIC_BASE_URL="https://gateway-us.pydantic.dev/proxy/anthropic"
    export ANTHROPIC_AUTH_TOKEN="YOUR_GATEWAY_API_KEY"
    ```

=== "EU"

    ```bash
    export ANTHROPIC_BASE_URL="https://gateway-eu.pydantic.dev/proxy/anthropic"
    export ANTHROPIC_AUTH_TOKEN="YOUR_GATEWAY_API_KEY"
    ```

Replace `YOUR_GATEWAY_API_KEY` with the API key from your Logfire organization's Gateway settings.

Launch Claude Code by typing `claude`. All requests will now route through the Pydantic AI Gateway.

### Codex

Codex uses the OpenAI Responses API, so it should use the Gateway's `openai-responses` route.

Set your gateway API key as an environment variable:

```bash
export PYDANTIC_AI_GATEWAY_API_KEY="YOUR_GATEWAY_API_KEY"
```

Then add the following configuration to `~/.codex/config.toml`, using the base URL that matches your Logfire region:

=== "US"

    ```toml
    model = "gpt-5.4"
    model_provider = "pydantic_gateway"

    [model_providers.pydantic_gateway]
    name = "Pydantic AI Gateway"
    base_url = "https://gateway-us.pydantic.dev/proxy/openai-responses"
    env_key = "PYDANTIC_AI_GATEWAY_API_KEY"
    env_key_instructions = "Create a Gateway API key in your Logfire organization's Gateway settings."
    wire_api = "responses"
    ```

=== "EU"

    ```toml
    model = "gpt-5.4"
    model_provider = "pydantic_gateway"

    [model_providers.pydantic_gateway]
    name = "Pydantic AI Gateway"
    base_url = "https://gateway-eu.pydantic.dev/proxy/openai-responses"
    env_key = "PYDANTIC_AI_GATEWAY_API_KEY"
    env_key_instructions = "Create a Gateway API key in your Logfire organization's Gateway settings."
    wire_api = "responses"
    ```

For more details on configuring custom providers in Codex, see the [Codex custom model providers docs](https://developers.openai.com/codex/config-advanced#custom-model-providers) and the [Codex configuration reference](https://developers.openai.com/codex/config-reference/).

If you already have a `~/.codex/config.toml`, add the `[model_providers.pydantic_gateway]` block and update `model_provider` instead of replacing the whole file. Replace `gpt-5.4` with whichever OpenAI Responses model you want Codex to use.

Launch Codex by typing `codex`. All requests will now route through the Pydantic AI Gateway.

### SDKs

#### OpenAI SDK

Use the base URL that matches your Logfire region (`gateway-us` or `gateway-eu`).

=== "US"

    ```python {title="openai_sdk.py" test="skip"}
    import openai

    client = openai.Client(
        base_url='https://gateway-us.pydantic.dev/proxy/chat/',
        api_key='pylf_v...',
    )

    response = client.chat.completions.create(
        model='gpt-5.2',
        messages=[{'role': 'user', 'content': 'Hello world'}],
    )
    print(response.choices[0].message.content)
    #> Hello user
    ```

=== "EU"

    ```python {title="openai_sdk.py" test="skip"}
    import openai

    client = openai.Client(
        base_url='https://gateway-eu.pydantic.dev/proxy/chat/',
        api_key='pylf_v...',
    )

    response = client.chat.completions.create(
        model='gpt-5.2',
        messages=[{'role': 'user', 'content': 'Hello world'}],
    )
    print(response.choices[0].message.content)
    #> Hello user
    ```

#### Anthropic SDK

Use the base URL that matches your Logfire region (`gateway-us` or `gateway-eu`).

=== "US"

    ```python {title="anthropic_sdk.py" test="skip"}
    import anthropic

    client = anthropic.Anthropic(
        base_url='https://gateway-us.pydantic.dev/proxy/anthropic/',
        auth_token='pylf_v...',
    )

    response = client.messages.create(
        max_tokens=1000,
        model='claude-sonnet-4-5',
        messages=[{'role': 'user', 'content': 'Hello world'}],
    )
    print(response.content[0].text)
    #> Hello user
    ```

=== "EU"

    ```python {title="anthropic_sdk.py" test="skip"}
    import anthropic

    client = anthropic.Anthropic(
        base_url='https://gateway-eu.pydantic.dev/proxy/anthropic/',
        auth_token='pylf_v...',
    )

    response = client.messages.create(
        max_tokens=1000,
        model='claude-sonnet-4-5',
        messages=[{'role': 'user', 'content': 'Hello world'}],
    )
    print(response.content[0].text)
    #> Hello user
    ```

## Self-hosting

Pydantic AI Gateway can also be deployed as part of a self-hosted Pydantic Logfire installation via the [Logfire Helm chart](https://github.com/pydantic/logfire-helm-chart). This is intended for customers who need to run the Gateway within their own infrastructure.

The Gateway ships as the `logfire-ai-gateway` service in the chart and is disabled by default. To enable it, set the following in your `values.yaml`:

```yaml
logfire-ai-gateway:
  enabled: true
```

You will also need a `logfire-gateway` Kubernetes secret with a `key` (gateway encryption key) and an `internalSecret`. The chart can manage this for you, or you can provide an existing secret via `existingGatewaySecret`:

```yaml
existingGatewaySecret:
  enabled: true
  name: my-logfire-gateway-secret
```

Gateway API keys are then created from your self-hosted Logfire organization's Gateway settings, and used with the [Gateway base URL](#usage) of your deployment.

For the full list of configuration options (autoscaling, resources, HPA/KEDA, pod disruption budgets, etc.), see the [Logfire Helm chart README](https://github.com/pydantic/logfire-helm-chart/blob/main/charts/logfire/README.md).

## Troubleshooting

### Unable to calculate spend

The gateway needs to know the cost of the request in order to provide insights about the spend, and to enforce spending limits.
If it's unable to calculate the cost, it will return a 400 error with the message "Unable to calculate spend".

When configuring a provider, you need to decide if you want the gateway to block
the API key if it's unable to calculate the cost. If you choose to block the API key, any further requests using that API key will fail.

We are actively working on supporting more providers, and models.
If you have a specific provider that you would like to see supported, please let us know on [Slack](https://logfire.pydantic.dev/docs/join-slack/) or [open an issue on `genai-prices`](https://github.com/pydantic/genai-prices/issues/new).
