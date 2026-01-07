# Azure AI Foundry

Azure AI Foundry provides access to OpenAI models and other AI models through Microsoft Azure's cloud platform.

## Install

To use Azure AI Foundry, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[openai]'
```

## Configuration

To use [Azure AI Foundry](https://ai.azure.com/) as your provider, you can set the `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `OPENAI_API_VERSION` environment variables and use [`AzureProvider`][pydantic_ai.providers.azure.AzureProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('azure:gpt-5')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.azure import AzureProvider

model = OpenAIChatModel(
    'gpt-5',
    provider=AzureProvider(
        azure_endpoint='your-azure-endpoint',
        api_version='your-api-version',
        api_key='your-api-key',
    ),
)
agent = Agent(model)
...
```
