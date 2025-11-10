# Usage
After setting up your account with the instructions from the [Quick Start](./quick-start.md) page, you will be able to make an AI model request with the Pydantic AI Gateway.
This page contains the example code snippets you can use to test your keys with different frameworks and SDKs.

To use different models, change the model string `gateway/<api_type>:<model_name>` to other models offered by the supported providers.

Examples of providers and models that can be used are:

| **Provider** | **Provider ID** | **Example Model** |
| --- | --- | --- |
| OpenAI | `openai` | `gateway/openai:gpt-4.1` |
| Anthropic | `anthropic` | `gateway/anthropic:claude-sonnet-4-5` |
| Google Vertex | `google-vertex`  | `gateway/google-vertex:gemini-2.5-flash` |
| Groq | `groq`  | `gateway/groq:openai/gpt-oss-120b` |
| AWS Bedrock | `bedrock`  | `gateway/bedrock:amazon.nova-micro-v1:0` |

## Pydantic AI
Before you start, update to the latest version of `pydantic-ai`:

=== "uv"

    ```python
    uv sync -P pydantic-ai
    ```
=== "pip"

    ```python
    pip install -U pydantic-ai
    ```

Set the `PYDANTIC_AI_GATEWAY_API_KEY`  environment variable to your gateway API key:

```bash
export PYDANTIC_AI_GATEWAY_API_KEY="YOUR_PAIG_TOKEN"
```

You can access multiple models with the same API key, as shown in the code snippet below.

```python title="hello_world.py"
from pydantic_ai import Agent

agent = Agent(
    'gateway/openai:gpt-5',
    instructions='Be concise, reply with one sentence.'
)

result = agent.run_sync('Hello World')
print(result.output)
```


## Claude Code
Before you start, log out of Claude Code using `/logout`.

Set your gateway credentials as environment variables:

```bash
export ANTHROPIC_AUTH_TOKEN="YOUR_PAIG_TOKEN"

export ANTHROPIC_BASE_URL="https://gateway.pydantic.dev/proxy/anthropic"
```

Replace `YOUR_PAIG_TOKEN` with the API key from the Keys page.

Launch Claude Code by typing `claude`. All requests will now route through the Pydantic AI Gateway.

## SDKs

=== "OpenAI SDK"

    ```python title="hello_world.py"
    import openai

    client = openai.Client(
        base_url='https://gateway.pydantic.dev/proxy/openai/',
        api_key='paig_...',
    )

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{'role': 'user', 'content': 'Hello world'}],
    )
    print(response.choices[0].message.content)
    ```
=== "Anthropic SDK"

    ```python title="hello_world.py"
    import anthropic

    client = anthropic.Anthropic(
        base_url='https://gateway.pydantic.dev/proxy/anthropic/',
        auth_token='paig_...',
    )

    response = client.messages.create(
        max_tokens=1000,
        model='claude-3-haiku-20240307',
        messages=[{'role': 'user', 'content': 'Hello world'}],
    )
    print(response.content[0].text)
    ```
