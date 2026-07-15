from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

from pydantic_ai._utils import is_async_callable
from pydantic_ai.models import KnownModelName, Model, ModelResolutionContext
from pydantic_ai.tools import AgentDepsT

from .abstract import AbstractCapability

_SyncResolver = Callable[[ModelResolutionContext[AgentDepsT], str], 'Model | None']
_AsyncResolver = Callable[[ModelResolutionContext[AgentDepsT], str], Awaitable['Model | None']]

ModelIdResolverFunc = _SyncResolver[AgentDepsT] | _AsyncResolver[AgentDepsT]
"""A function that resolves a model-name string to a [`Model`][pydantic_ai.models.Model], or `None` to defer.

Receives the [`ModelResolutionContext`][pydantic_ai.models.ModelResolutionContext] (carrying the
agent and the run's `deps`) and the model-name string. Can be sync or async.
"""


@dataclass
class ResolveModelId(AbstractCapability[AgentDepsT]):
    """A capability that resolves model-name strings to `Model` instances, with access to the run's deps.

    Wraps a resolver function that is called at run setup for any model-name string in play —
    the agent's default, `run(model=...)`, or `override(model=...)` — with a
    [`ModelResolutionContext`][pydantic_ai.models.ModelResolutionContext] carrying the agent and
    the run's `deps`. Return a [`Model`][pydantic_ai.models.Model] to use it, or `None` to defer
    to the next capability or the default [`infer_model`][pydantic_ai.models.infer_model] flow.

    The flagship use case is per-user provider authentication: carry per-request credentials on
    `deps` and build the model with a provider that uses them, by forwarding to
    [`infer_model`][pydantic_ai.models.infer_model] with `provider_factory=...`:

    ```python {test="skip"}
    from dataclasses import dataclass

    from pydantic_ai import Agent
    from pydantic_ai.capabilities import ResolveModelId
    from pydantic_ai.models import ModelResolutionContext, infer_model
    from pydantic_ai.providers.openai import OpenAIProvider


    @dataclass
    class Deps:
        openai_api_key: str


    def resolve(ctx: ModelResolutionContext[Deps], model_id: str):
        return infer_model(model_id, provider_factory=lambda _: OpenAIProvider(api_key=ctx.deps.openai_api_key))


    agent = Agent('openai:gpt-5.2', deps_type=Deps, capabilities=[ResolveModelId(resolve)])
    ```

    Under durable execution (Temporal, DBOS, Prefect), the resolver runs again inside the
    activity/step/task to rebuild the model on the worker, so it must be deterministic for a
    given `(model_id, deps)` and must not perform external I/O — carry credentials and registry
    data on `deps` instead.
    """

    resolver: ModelIdResolverFunc[AgentDepsT]

    async def resolve_model_id(
        self,
        model_id: KnownModelName | str,
        ctx: ModelResolutionContext[AgentDepsT],
    ) -> Model | None:
        if is_async_callable(self.resolver):
            return await cast('_AsyncResolver[Any]', self.resolver)(ctx, model_id)
        # Sync resolvers run inline rather than in a thread: resolution is expected to be
        # fast and deterministic (see the class docstring), and under Temporal it runs
        # inside the workflow sandbox where threads are unavailable. Use an async
        # resolver for anything slow.
        return cast('_SyncResolver[Any]', self.resolver)(ctx, model_id)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (takes a callable)
