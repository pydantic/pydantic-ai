"""Provider-adaptive advisor capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic_ai import Agent
from pydantic_ai.capabilities import ModelSelection, NativeOrLocalTool
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import ModelRequestContext, parse_model_id
from pydantic_ai.native_tools import AdvisorTool
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, AgentNativeTool, RunContext, Tool

_LIMIT_REACHED = 'Advisor consultation limit reached for this model request. Continue without further advice.'


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
    """One-off: an agent has one advisor, and its tool name is fixed.

    Declared here rather than only passed up from `__init__`, so the class states it where a reader
    -- and Pydantic AI, deciding what two of this capability under one `id` mean -- can see it.
    """

    _local_uses: int = field(init=False, repr=False, default=0)

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
            id='advisor',
        )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Advisor[AgentDepsT]:
        """Return a fresh capability with local usage isolated to this run."""
        if self.max_uses is None or self.mode == 'native':
            return self
        return Advisor(
            self.model,
            mode=self.mode,
            output_type=self.output_type,
            max_uses=self.max_uses,
            max_tokens=self.max_tokens,
            caching=self.caching,
            forward_history=self.forward_history,
        )

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
