---
title: Pydantic AI Gateway
status: new
---

# Pydantic AI Gateway

**[Pydantic AI Gateway](https://pydantic.dev/ai-gateway)** is a unified interface for accessing multiple AI providers with a single key. Features include built-in OpenTelemetry observability, real-time cost monitoring, failover management, and native integration with the other tools in the [Pydantic stack](https://pydantic.dev/).

!!! note "Free while we collect feedback"
    You can bring your own key (BYOK) or buy inference through the Gateway (we will eat the card fee for now).

Sign up at [gateway.pydantic.dev](https://gateway.pydantic.dev/).

!!! question "Questions?"
    For questions and feedback, contact us on [Slack](https://logfire.pydantic.dev/docs/join-slack/).

## Documentation Integration

To help you get started with [Pydantic AI Gateway](https://gateway.pydantic.dev), some code examples on the Pydantic AI documentation include a "Via Pydantic AI Gateway" tab, alongside a "Direct to Provider API" tab with the standard Pydantic AI model string. The main difference between them is that when using Gateway, model strings use the `gateway/` prefix.

## Key features

- **API key management**: access multiple LLM providers with a single Gateway key.
- **Cost Limits**: set spending limits at project, user, and API key levels with daily, weekly, and monthly caps.
- **BYOK and managed providers:** Bring your own API keys (BYOK) from LLM providers, or pay for inference directly through the platform.
- **Multi-provider support:** Access models from OpenAI, Anthropic, Google Vertex, Groq, and AWS Bedrock. _More providers coming soon_.
- **Backend observability:** Log every request through [Pydantic Logfire](https://pydantic.dev/logfire) or any OpenTelemetry backend (_coming soon_).
- **Zero translation**: Unlike traditional AI gateways that translate everything to one common schema, **Pydantic AI Gateway** allows requests to flow through directly in each provider's native format. This gives you immediate access to the new model features as soon as they are released.
- **Open source with self-hosting**: Pydantic AI Gateway core is [open source](https://github.com/pydantic/pydantic-ai-gateway/) (under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)), allowing self-hosting with file-based configuration, instead of using the managed service.
- **Enterprise ready**: Includes SSO (with OIDC support), granular permissions, and flexible deployment options. Deploy to your Cloudflare account, or run on-premises with our [consulting support](https://pydantic.dev/contact).

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

Using your  GitHub or Google account, sign in at [gateway.pydantic.dev](https://gateway.pydantic.dev).
Choose a name for your organization (or accept the default). You will automatically be assigned the Admin role.

A default project will be created for you. You can choose to use it, or create a new one on the [Projects](https://gateway.pydantic.dev/admin/projects) page.

### Add **Providers**

There are two ways to use Providers in the Pydantic AI Gateway: you can bring your own key (BYOK) or buy inference through the platform.

#### Bringing your own API key (BYOK)

On the [Providers](https://gateway.pydantic.dev/admin/providers) page, fill in the form to add a provider.
Paste your API key into the form under Credentials, and make sure to **select the Project that will be associated to this provider**.
It is possible to add multiple keys from the same provider.

#### Use Built-in Providers

Go to the [Billing page](https://gateway.pydantic.dev/admin/billing), add a payment method, and purchase $15 in credits to activate built-in providers.
This gives you single-key access to all available models from OpenAI, Anthropic, Google Vertex, AWS Bedrock, and Groq.

### Grant access to your team

On the [Users](https://gateway.pydantic.dev/admin/users) page, create an invitation and share the URL with your team to allow them to access the project.

### Create Gateway project keys

On the Keys page, Admins can create project keys which are not affected by spending limits.
Users can only create personal keys, that will inherit spending caps from both User and Project levels, whichever is more restrictive.

## Usage

After setting up your account with the instructions above, you will be able to make an AI model request with the Pydantic AI Gateway.
The code snippets below show how you can use Pydantic AI Gateway with different frameworks and SDKs.
You can add `gateway/` as prefix on every known provider that

To use different models, change the model string `gateway/<api_format>:<model_name>` to other models offered by the supported providers.

Examples of providers and models that can be used are:

| **Provider** | **API Format**  | **Example Model**                        |
| --- |-----------------|------------------------------------------|
| OpenAI | `openai`        | `gateway/openai:gpt-5.2`                 |
| Anthropic | `anthropic`     | `gateway/anthropic:claude-sonnet-4-5`    |
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
export PYDANTIC_AI_GATEWAY_API_KEY="paig_<example_key>"
```

You can access multiple models with the same API key, as shown in the code snippet below.

=== "Hello World"

    ```python {title="hello_world.py"}
    from pydantic_ai import Agent

    agent = Agent('gateway/openai:gpt-5.2')

    result = agent.run_sync('Where does "hello world" come from?')
    print(result.output)
    """
    The first known use of "hello, world" was in a 1974 textbook about the C programming language.
    """
    ```

=== "Passing API Key directly"

    Pass your API key directly using the [`gateway_provider`][pydantic_ai.providers.gateway.gateway_provider]:

    ```python {title="passing_api_key.py"}
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.gateway import gateway_provider

    provider = gateway_provider('openai', api_key='paig_<example_key>')
    model = OpenAIChatModel('gpt-5.2', provider=provider)
    agent = Agent(model)

    result = agent.run_sync('Where does "hello world" come from?')
    print(result.output)
    """
    The first known use of "hello, world" was in a 1974 textbook about the C programming language.
    """
    ```

=== "Using a different upstream provider"

    To use an alternate provider or routing group, you can specify it in the route parameter:

    ```python {title="routing_via_provider.py"}
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.gateway import gateway_provider

    provider = gateway_provider(
        'openai',
        api_key='paig_<example_key>',
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

Set your gateway credentials as environment variables:

```bash
export ANTHROPIC_BASE_URL="https://gateway.pydantic.dev/proxy/anthropic"
export ANTHROPIC_AUTH_TOKEN="YOUR_PYDANTIC_AI_GATEWAY_API_KEY"
```

Replace `YOUR_PYDANTIC_AI_GATEWAY_API_KEY` with the API key from the Keys page.

Launch Claude Code by typing `claude`. All requests will now route through the Pydantic AI Gateway.

### SDKs

#### OpenAI SDK

```python {title="openai_sdk.py" test="skip"}
import openai

client = openai.Client(
    base_url='https://gateway.pydantic.dev/proxy/chat/',
    api_key='paig_...',
)

response = client.chat.completions.create(
    model='gpt-5.2',
    messages=[{'role': 'user', 'content': 'Hello world'}],
)
print(response.choices[0].message.content)
#> Hello user
```

#### Anthropic SDK

```python {title="anthropic_sdk.py" test="skip"}
import anthropic

client = anthropic.Anthropic(
    base_url='https://gateway.pydantic.dev/proxy/anthropic/',
    auth_token='paig_...',
)

response = client.messages.create(
    max_tokens=1000,
    model='claude-sonnet-4-5',
    messages=[{'role': 'user', 'content': 'Hello world'}],
)
print(response.content[0].text)
#> Hello user
```

## Troubleshooting

### Unable to calculate spend

The gateway needs to know the cost of the request in order to provide insights about the spend, and to enforce spending limits.
If it's unable to calculate the cost, it will return a 400 error with the message "Unable to calculate spend".

When [configuring a provider](https://gateway.pydantic.dev/admin/providers/new), you need to decide if you want the gateway to block
the API key if it's unable to calculate the cost. If you choose to block the API key, any further requests using that API key will fail.

We are actively working on supporting more providers, and models.
If you have a specific provider that you would like to see supported, please let us know on [Slack](https://logfire.pydantic.dev/docs/join-slack/) or [open an issue on `genai-prices`](https://github.com/pydantic/genai-prices/issues/new).
