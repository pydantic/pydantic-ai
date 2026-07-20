# Resolve Model ID

[`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] is a [capability](overview.md) that turns application-specific model IDs into [`Model`][pydantic_ai.models.Model] instances. The resolver can use run dependencies to look up tenant-specific providers, credentials, or model registries:

```python {title="resolve_model_id.py"}
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, ModelResolutionContext
from pydantic_ai.capabilities import ResolveModelId
from pydantic_ai.models import Model, infer_model
from pydantic_ai.providers import Provider, infer_provider
from pydantic_ai.providers.openai import OpenAIProvider


@dataclass
class Deps:
    """Per-user provider credentials."""

    openai_api_key: str


def resolve_model(ctx: ModelResolutionContext[Deps], model_id: str) -> Model | None:
    """Resolve IDs in the `user:` namespace with the current user's credentials."""
    if not model_id.startswith('user:'):
        return None

    def provider_factory(provider_name: str) -> Provider[Any]:
        if provider_name == 'openai':
            return OpenAIProvider(api_key=ctx.deps.openai_api_key)
        return infer_provider(provider_name)

    return infer_model(model_id.removeprefix('user:'), provider_factory)


agent = Agent(
    'user:openai:gpt-5.6-sol',
    deps_type=Deps,
    capabilities=[ResolveModelId(resolve_model)],
)
```

The resolver may be synchronous or asynchronous. Its full callable signature is
`(ModelResolutionContext[Deps], str) -> Model | None | Awaitable[Model | None]`.
The convenience capability adapts both forms to the asynchronous
[`resolve_model_id()`][pydantic_ai.capabilities.AbstractCapability.resolve_model_id] hook.

Resolvers form a chain in capability order: the first non-`None` result wins, and Pydantic AI falls back to normal model inference if every resolver returns `None`. See [Resolving model IDs](custom.md#resolving-model-ids) to implement the hook in a custom capability and understand when each resolver tree is used.
