# Pydantic AI Gateway

**Pydantic AI Gateway** (PAIG) is a unified interface for accessing multiple AI providers with a single key. Features include built-in OpenTelemetry observability, real-time cost monitoring, failover management , and native integration with the Pydantic stack.

!!! note "Free while in Beta"
    The Pydantic AI Gateway is currently in Beta. You can bring your own key (BYOK) or buy inference through the Gateway (we will eat the card fee for now).

Sign up at [gateway.pydantic.dev](https://gateway.pydantic.dev/).

For questions and feedback, contact us on [Slack](https://logfire.pydantic.dev/docs/join-slack/).

## Documentation Integration

To help you get started with [Pydantic AI Gateway](https://gateway.pydantic.dev), most code examples throughout the Pydantic AI docs include a "Pydantic AI Gateway" tab alongside the standard Pydantic AI call string named here "Direct to Provider API" tab. This allows you to see how to adapt examples for Gateway usage by simply switching tabs.

The main difference is that when using Gateway, model strings use the `gateway/` prefix.

## Key features
- **API key management**: access multiple LLM providers with a single Gateway key.
- **Cost Limits**: set spending limits at project, user, and API key levels with daily, weekly, and monthly caps.
- **BYOK and managed providers:** Bring your own API keys (BYOK) from LLM providers, or pay for API usage directly through the platform (_coming soon_).
- **Multi-provider support:** Access models from OpenAI, Anthropic, Google Vertex, Groq, and AWS Bedrock. _More providers coming soon_.
- **Backend observability:** Log every request through [Pydantic Logfire](https://pydantic.dev/logfire) or any OpenTelemetry backend (_coming soon_).
- **Zero translation**: Unlike traditional AI gateways that translate everything to one common schema, PAIG allows requests to flow through directly in each provider's native format. This gives you immediate access to the new model features as soon as they are released.
- **Open source with self-hosting**: PAIG's core is [open source](https://github.com/pydantic/pydantic-ai-gateway/) (under [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html)), allowing self-hosting with file-based configuration, instead of using the managed service.
- **Enterprise ready**: Includes SSO (with OIDC support), granular permissions, and flexible deployment options. Deploy to your Cloudflare account, or run on-premises with our [consulting support](https://pydantic.dev/contact).

```python {title="hello_world.py" test="skip"}
from pydantic_ai import Agent

agent = Agent('gateway/chat:gpt-4.1')

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```
# Quick Start
This section contains instructions on how to set up your account and run your app with Pydantic AI Gateway credentials.

## Create an account
Using your  GitHub or Google account, sign in at https://gateway.pydantic.dev.
Choose a name for your organization (or accept the default). You will automatically be assigned the Admin role.

A default project will be created for you. You can choose to use it, or create a new one on the [Projects](https://gateway.pydantic.dev/admin/projects) page.

## Add **Providers** by bringing your own API keys (BYOK)
Pydantic AI Gateway allows you to bring your API keys from your favourite provider(s).

On the [Providers](https://gateway.pydantic.dev/admin/providers) page, fill in the form to add a provider. Paste your API key into the form under Credentials, and make sure to **select the Project that will be associated to this provider**. It is possible to add multiple keys from the same provider.

## Grant access to your team
On the [Users](https://gateway.pydantic.dev/admin/users) page, create an invitation and share the URL with your team to allow them to access the project.

## Create gateway project keys
On the Keys page, Admins can create project keys which are not affected by spending limits. Users can only create personal keys, that will inherit spending caps from both User and Project levels, whichever is more restrictive.

# Usage
After setting up your account with the instructions above, you will be able to make an AI model request with the Pydantic AI Gateway.
The code snippets below show how you can use PAIG with different frameworks and SDKs.

To use different models, change the model string `gateway/<api_type>:<model_name>` to other models offered by the supported providers.

Examples of providers and models that can be used are:

| **Provider** | **API Type**    | **Example Model**                        |
| --- |-----------------|------------------------------------------|
| OpenAI | `openai`        | `gateway/openai:gpt-5`                   |
| Anthropic | `anthropic`     | `gateway/anthropic:claude-sonnet-4-5`    |
| Google Vertex | `google-vertex` | `gateway/google-vertex:gemini-2.5-flash` |
| Groq | `groq`          | `gateway/groq:openai/gpt-oss-120b`       |
| AWS Bedrock | `bedrock`       | `gateway/bedrock:amazon.nova-micro-v1:0` |

## Pydantic AI
Before you start, update to the latest version of `pydantic-ai`:

=== "uv"

    ```bash
    uv sync -P pydantic-ai
    ```

=== "pip"

    ```bash
    pip install -U pydantic-ai
    ```

Set the `PYDANTIC_AI_GATEWAY_API_KEY`  environment variable to your gateway API key:

```bash
export PYDANTIC_AI_GATEWAY_API_KEY="YOUR_PAIG_TOKEN"
```

You can access multiple models with the same API key, as shown in the code snippet below.

```python {title="hello_world.py"}
from pydantic_ai import Agent

agent = Agent('gateway/openai:gpt-5')

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
"""
The first known use of "hello, world" was in a 1974 textbook about the C programming language.
"""
```


## Claude Code
Before you start, log out of Claude Code using `/logout`.

Set your gateway credentials as environment variables:

```bash
export ANTHROPIC_BASE_URL="https://gateway.pydantic.dev/proxy/anthropic"
export ANTHROPIC_AUTH_TOKEN="YOUR_PAIG_TOKEN"
```

Replace `YOUR_PAIG_TOKEN` with the API key from the Keys page.

Launch Claude Code by typing `claude`. All requests will now route through the Pydantic AI Gateway.

## SDKs

=== "OpenAI SDK"

    ```python {title="openai_sdk.py" test="skip"}
    import openai

    client = openai.Client(
        base_url='https://gateway.pydantic.dev/proxy/chat/',
        api_key='paig_...',
    )

    response = client.chat.completions.create(
        model='gpt-5',
        messages=[{'role': 'user', 'content': 'Hello world'}],
    )
    print(response.choices[0].message.content)
    #> Hello user
    ```
=== "Anthropic SDK"

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
