# GitHub Models

GitHub Models provides access to AI models through GitHub's infrastructure.

## Install

To use GitHub Models, you need to either install `pydantic-ai`, or install `pydantic-ai-slim` with the `openai` optional group:

```bash
pip/uv-add 'pydantic-ai-slim[openai]'
```

## Configuration

To use [GitHub Models](https://docs.github.com/en/github-models), you'll need a GitHub personal access token with the `models: read` permission.

You can set the `GITHUB_API_KEY` environment variable and use [`GitHubProvider`][pydantic_ai.providers.github.GitHubProvider] by name:

```python
from pydantic_ai import Agent

agent = Agent('github:xai/grok-3-mini')
...
```

Or initialise the model and provider directly:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.github import GitHubProvider

model = OpenAIChatModel(
    'xai/grok-3-mini',  # GitHub Models uses prefixed model names
    provider=GitHubProvider(api_key='your-github-token'),
)
agent = Agent(model)
...
```

GitHub Models supports various model families with different prefixes. You can see the full list on the [GitHub Marketplace](https://github.com/marketplace?type=models) or the public [catalog endpoint](https://models.github.ai/catalog/models).
