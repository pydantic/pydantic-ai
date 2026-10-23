"""Provider-adaptive advisor capability."""

from __future__ import annotations

import copy
import inspect
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Self

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, ModelSelection, NativeOrLocalTool
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext, parse_model_id
from pydantic_ai.native_tools import AdvisorTool
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, AgentNativeTool, RunContext, Tool
from pydantic_ai_harness._warn import HarnessDeprecationWarning

_LIMIT_REACHED = 'Advisor consultation limit reached for this model request. Continue without further advice.'


def _caller_stacklevel() -> int:
    """The `warnings.warn` stacklevel, from `Advisor.combine`, of the nearest frame in the user's code.

    Skips this module, Pydantic AI, Pydantic Graph, and the `asyncio` and `contextlib` frames a run
    goes through, so the warning points at the caller whether they call `combine` directly, build an
    `Agent`, or pass capabilities to a run.
    """
    frame = inspect.currentframe()
    stacklevel = 0
    while frame is not None and (
        (module := frame.f_globals.get('__name__', '')) == __name__
        or module.partition('.')[0] in {'pydantic_ai', 'pydantic_graph', 'asyncio', 'contextlib'}
    ):
        frame = frame.f_back
        stacklevel += 1
    return stacklevel


@dataclass(init=False)
class Advisor(NativeOrLocalTool[AgentDepsT]):
    """Let an agent consult another model through a provider-native tool or local fallback.

    In `auto` mode, `Advisor` uses Pydantic AI's native `AdvisorTool` when an
    explicit provider-qualified model name matches a compatible Anthropic or
    OpenRouter executor. On every other model, it exposes an `advisor` function
    tool backed by a separate Pydantic AI agent.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.advisor import Advisor

    agent = Agent(
        'openai:gpt-5.4',
        capabilities=[Advisor('anthropic:claude-opus-4-8')],
    )
    ```
    """

    model: ModelSelection
    """The model to consult.

    Accepts the same model names and model instances as `Agent`. In `auto`
    mode, model instances use local execution so their provider configuration
    is preserved.
    """

    mode: Literal['auto', 'native', 'local']
    """How advisor consultations are executed.

    `auto` uses a native advisor only for an explicit same-provider model name.
    `native` requires a provider-native advisor, and `local` always runs a
    separate Pydantic AI agent.
    """

    output_type: OutputSpec[object]
    """Output specification for local consultations, defaulting to text.

    A non-default specification selects local execution in `auto` mode and is
    incompatible with `native` mode. Successful consultations return the
    validated output directly to the executor.
    """

    max_uses: int | None
    """Maximum consultations in one executor model request.

    The limit resets on the next executor request. OpenRouter's native advisor
    does not honor this option, so setting it selects the local fallback there.
    """

    max_tokens: int | None
    """Maximum output tokens for each advisor consultation.

    Values below 1024 are rejected so the setting remains valid on every native
    and local execution path.
    """

    caching: Literal['5m', '1h'] | None
    """Anthropic-native advisor prompt caching.

    This is an opportunistic optimization. OpenRouter and the local fallback do
    not provide an equivalent cache control.
    """

    forward_history: bool
    """Whether local consultations receive the executor's completed message history.

    Native execution keeps the provider's transcript behavior unchanged.
    """

    id: str | None = 'advisor'
    """One-off: an agent has one advisor, so the id is fixed.

    Two `Advisor` capabilities resolve to one via
    [`combine`][pydantic_ai.capabilities.AbstractCapability.combine].
    """

    _local_uses: int = field(init=False, repr=False, compare=False, default=0)

    def __init__(
        self,
        model: ModelSelection,
        *,
        mode: Literal['auto', 'native', 'local'] = 'auto',
        output_type: OutputSpec[object] = str,
        max_uses: int | None = None,
        max_tokens: int | None = None,
        caching: Literal['5m', '1h'] | None = None,
        forward_history: bool = False,
    ) -> None:
        if mode not in {'auto', 'native', 'local'}:
            raise ValueError("Advisor.mode must be 'auto', 'native', or 'local'")
        if max_uses is not None and max_uses < 1:
            raise ValueError('Advisor.max_uses must be at least 1')
        if max_tokens is not None and max_tokens < 1024:
            raise ValueError('Advisor.max_tokens must be at least 1024')

        if mode == 'native' and output_type is not str:
            raise ValueError("Advisor.output_type is not supported in mode='native'")

        self.output_type = output_type
        self.model = model
        self.mode = mode
        self.max_uses = max_uses
        self.max_tokens = max_tokens
        self.caching = caching
        self.forward_history = forward_history
        self._local_uses = 0
        native_provider, _ = self._parse_native_model(model)
        if mode == 'native' and native_provider is None:
            raise ValueError(
                "Advisor(mode='native') requires an 'anthropic:<model>' or 'openrouter:<model>' model name"
            )
        if mode == 'native' and native_provider == 'openrouter' and max_uses is not None:
            raise ValueError("Advisor.max_uses is not supported by OpenRouter in mode='native'")

        self._build_tools()

    def _build_tools(self) -> None:
        """Build the native factory and local tool from the fields, and initialise the base capability."""
        model, mode, output_type = self.model, self.mode, self.output_type
        max_uses, max_tokens, forward_history = self.max_uses, self.max_tokens, self.forward_history
        native: AgentNativeTool[AgentDepsT] | bool
        local: Tool[AgentDepsT] | bool
        if mode == 'native':
            native = self._required_native_advisor
            local = False
        else:

            async def advisor(ctx: RunContext[AgentDepsT], prompt: str) -> object:
                if max_uses is not None:
                    if self._local_uses >= max_uses:
                        return _LIMIT_REACHED
                    self._local_uses += 1

                settings = ModelSettings(max_tokens=max_tokens) if max_tokens is not None else None
                advisor_agent = Agent(
                    model,
                    name='advisor',
                    output_type=output_type,
                    instructions=(
                        'You are an expert advisor. '
                        + (
                            'Give concise, actionable advice to the executor model about the question it sends you. '
                            if output_type is str
                            else "Answer the executor model's question using the configured output format. "
                        )
                        + 'Do not address the end user.'
                    ),
                    model_settings=settings,
                )
                try:
                    result = await advisor_agent.run(
                        prompt,
                        message_history=ctx.messages[:-1] if forward_history else None,
                        conversation_id=ctx.conversation_id,
                        usage=ctx.usage,
                        usage_limits=ctx.usage_limits,
                    )
                except UnexpectedModelBehavior as e:
                    raise ModelRetry(str(e)) from e
                return result.output

            local = Tool(
                advisor,
                description=(
                    'Consult a stronger model about a difficult or high-impact decision. '
                    + ('' if output_type is str else 'Returns a validated answer in the configured output format. ')
                    + 'Include the complete question and all relevant context in `prompt` '
                    'because conversation history may not be available to the advisor.'
                ),
            )
            native = False if mode == 'local' or output_type is not str else self._native_advisor
        super().__init__(
            native=native,
            local=local,
            id=self.id,
        )

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge the fields, then rebuild the native factory and local tool from the merged result.

        Both are built in `__init__` from one instance's configuration, so the inherited merge alone
        would send the later instance's settings while the merged fields report the union.
        """
        merged = super().combine(capabilities)
        assert isinstance(merged, cls)
        max_uses = merged.max_uses
        if (
            max_uses is not None
            and merged.mode == 'native'
            and cls._parse_native_model(merged.model)[0] == 'openrouter'
        ):
            # No single `Advisor` accepts this, but merging used to send the native tool without
            # `max_uses`; keep doing that rather than make an agent that used to build raise.
            warnings.warn(
                f"Merged `Advisor` capabilities combine `mode='native'` on OpenRouter with `max_uses={max_uses}`, "
                "which OpenRouter's native advisor does not support, so `max_uses` is ignored. "
                "This will raise `UserError` in a future release: remove `max_uses`, or use `mode='auto'`.",
                HarnessDeprecationWarning,
                stacklevel=_caller_stacklevel(),
            )
            max_uses = None
        return merged._rebuilt(max_uses=max_uses)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Self:
        """Return a fresh capability with local usage isolated to this run."""
        if self.max_uses is None or self.mode == 'native':
            return self
        return self._rebuilt(max_uses=self.max_uses)

    def _rebuilt(self, *, max_uses: int | None) -> Self:
        """Copy this capability and rebuild its native factory and local tool from its fields.

        Only `Advisor`'s tool building runs, so a subclass keeps its type and fields without its
        `__init__` being called with arguments it may not take.
        """
        rebuilt = copy.copy(self)
        rebuilt.max_uses = max_uses
        rebuilt._local_uses = 0
        rebuilt._build_tools()
        # `_build_tools` resets these inherited fields, which callers may set after construction.
        rebuilt.description = self.description
        rebuilt.defer_loading = self.defer_loading
        return rebuilt

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Reset the local consultation allowance for each executor response."""
        self._local_uses = 0
        return response

    def _native_unique_id(self) -> str:
        """Identify the native tool paired with the local fallback."""
        return AdvisorTool.kind

    def _native_advisor(self, ctx: RunContext[AgentDepsT]) -> AdvisorTool | None:
        """Build a native tool only when an explicit advisor model name is lossless."""
        native_provider, native_model_name = self._parse_native_model(self.model)
        provider = ctx.model.system
        if provider != native_provider or native_model_name is None:
            return None
        if provider == 'openrouter' and self.max_uses is not None:
            return None
        return self._advisor_tool(native_model_name)

    def _required_native_advisor(self, ctx: RunContext[AgentDepsT]) -> AdvisorTool:
        """Build a required native tool or reject a cross-provider executor."""
        native_provider, native_model_name = self._parse_native_model(self.model)
        if ctx.model.system != native_provider:
            raise UserError(f"Advisor(mode='native') requires a {native_provider} executor, not {ctx.model.system}")
        assert native_model_name is not None
        return self._advisor_tool(native_model_name)

    def _advisor_tool(self, model_name: str) -> AdvisorTool:
        return AdvisorTool(
            model=model_name,
            max_uses=self.max_uses,
            max_tokens=self.max_tokens,
            caching=self.caching,
        )

    @staticmethod
    def _parse_native_model(
        model: ModelSelection,
    ) -> tuple[Literal['anthropic', 'openrouter'] | None, str | None]:
        """Extract a native model ID only from an explicit provider-qualified name."""
        if not isinstance(model, str):
            return None, None
        provider, model_name = parse_model_id(model)
        if model_name and (provider == 'anthropic' or provider == 'openrouter'):
            return provider, model_name
        return None, None
