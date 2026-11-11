"""Provider-adaptive advisor capability."""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Self, TypeVar

from typing_extensions import TypeIs

from pydantic_ai import Agent
from pydantic_ai._utils import replace_no_init
from pydantic_ai.capabilities import AbstractCapability, ModelSelection, NativeOrLocalTool, durable_operation
from pydantic_ai.exceptions import UnexpectedModelBehavior, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import AbstractModel, KnownModelName, Model, ModelRequestContext, parse_model_id
from pydantic_ai.native_tools import AdvisorTool
from pydantic_ai.output import OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext, Tool
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness._durable import RetryRequest, raise_retry
from pydantic_ai_harness._warn import HarnessDeprecationWarning


@dataclass
class _Advice:
    """A consultation's output, wrapped so a journaled output can't be read back as a `RetryRequest`."""

    output: object


_Mode = Literal['auto', 'native', 'local']
"""The deprecated `Advisor.mode` values."""

_T = TypeVar('_T')


@dataclass(init=False)
class Advisor(NativeOrLocalTool[AgentDepsT]):
    """Let an agent consult another model through a provider-native tool or local fallback.

    By default, `Advisor` uses Pydantic AI's native `AdvisorTool` when an explicit
    provider-qualified model name matches a compatible Anthropic or OpenRouter
    executor. On every other model, it exposes an `advisor` function tool backed
    by a separate Pydantic AI agent. Pass `local=False` to require the native
    advisor, or `native=False` to always consult locally.

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

    Accepts the same model names and model instances as `Agent`. A model
    instance is consulted locally so its provider configuration is preserved;
    only an explicit `'anthropic:<model>'` or `'openrouter:<model>'` name can
    select the native advisor.
    """

    output_type: OutputSpec[object]
    """Output specification for local consultations, defaulting to text.

    Provider-native advisors return text, so a non-default specification
    consults locally and cannot be combined with `local=False`. Successful
    consultations return the validated output directly to the executor.
    """

    max_uses: int | None
    """Maximum consultations in one executor model request.

    The limit resets on the next executor request. OpenRouter's native advisor
    does not honor this option, so setting it consults locally there.
    """

    max_tokens: int | None
    """Maximum output tokens for each advisor consultation.

    Values below 1024 are rejected so the setting remains valid on every native
    and local execution path.
    """

    caching: Literal['5m', '1h'] | None
    """Prompt-cache TTL for the advisor's transcript. Native-only; ignored by local tools.

    Supported by: Anthropic.
    """

    forward_history: bool
    """Whether local consultations receive the executor's completed message history.

    Local-only: native execution keeps the provider's transcript behavior unchanged.
    """

    id: str | None = 'advisor'
    """One-off: an agent has one advisor, so the id is fixed.

    Two `Advisor` capabilities resolve to one via
    [`combine`][pydantic_ai.capabilities.AbstractCapability.combine].
    """

    _local_uses: int = field(init=False, repr=False, compare=False, default=0)

    _MODE_DECLARATIONS: ClassVar[dict[_Mode, tuple[bool, bool]]] = {
        'auto': (True, True),
        'native': (True, False),
        'local': (False, True),
    }
    """The `(native, local)` pair each deprecated `mode` stands for."""

    _MODE_REPLACEMENTS: ClassVar[dict[_Mode, str]] = {
        'auto': 'omit `mode`; the defaults use the native advisor where it applies and the local tool otherwise',
        'native': 'pass `local=False`',
        'local': 'pass `native=False`',
    }

    def __init__(
        self,
        model: ModelSelection,
        *,
        native: AdvisorTool
        | Callable[[RunContext[AgentDepsT]], Awaitable[AdvisorTool | None] | AdvisorTool | None]
        | bool = True,
        local: Tool[AgentDepsT] | Callable[..., Any] | bool | None = True,
        output_type: OutputSpec[object] = str,
        max_uses: int | None = None,
        max_tokens: int | None = None,
        caching: Literal['5m', '1h'] | None = None,
        forward_history: bool = False,
        id: str | None = 'advisor',
        defer_loading: bool = False,
        description: str | None = None,
        mode: _Mode | None = None,
    ) -> None:
        """Create an advisor capability.

        Args:
            model: The model to consult.
            native: The provider-native advisor: `True` builds it from the fields, `False` disables
                it, and an `AdvisorTool` or a function returning one replaces it. A replacement
                carries its own configuration, so `max_uses`, `max_tokens` and `caching` don't
                apply to it, and a later default `Advisor` in a merge restores the built one.
            local: The local fallback: `True` (the default) or `None` builds the `advisor` tool from
                the fields, `False` disables it, and a `Tool` or function replaces it. A replacement
                ignores `output_type` and `forward_history`, while `max_uses`, `max_tokens` and
                `caching` still shape the native advisor. `None` is stored as `True`, so that, as
                with the earlier `mode='auto'`, a later default `Advisor` in a merge restores the
                built local fallback, replacing an earlier `False` or replacement tool.
            output_type: Output specification for local consultations.
            max_uses: Maximum consultations in one executor model request.
            max_tokens: Maximum output tokens for each consultation.
            caching: Prompt-cache TTL for the native advisor's transcript.
            forward_history: Whether local consultations receive the executor's message history.
            id: The capability id.
            defer_loading: Whether to make this an on-demand capability, whose tools the model loads by `id`.
            description: The capability description.
            mode: Deprecated: use `native` and `local`.
        """
        if local is None:
            # `None` and `True` build the same tool; stating `True` makes them merge the same way too.
            local = True
        if mode is not None:
            self._check_mode(mode)
            if native is not True or local is not True:
                raise UserError('`Advisor`: pass either the deprecated `mode` or `native`/`local`, not both')
            warnings.warn(
                f'`Advisor(mode={mode!r})` is deprecated: {self._MODE_REPLACEMENTS[mode]}. '
                '`mode` will be removed in a future release.',
                HarnessDeprecationWarning,
                stacklevel=self._caller_stacklevel(),
            )
            native, local = self._MODE_DECLARATIONS[mode]
        self.model = model
        self.output_type = output_type
        self.max_uses = max_uses
        self.max_tokens = max_tokens
        self.caching = caching
        self.forward_history = forward_history
        self._local_uses = 0
        super().__init__(native=native, local=local, id=id, defer_loading=defer_loading, description=description)

    @property
    def mode(self) -> _Mode:
        """Deprecated: read the `native` and `local` declarations instead.

        `'native'` when the local fallback is disabled, `'local'` when the native advisor is, and
        `'auto'` otherwise.
        """
        warnings.warn(
            '`Advisor.mode` is deprecated: `native=False` means local only, and `local=False` means native only. '
            '`mode` will be removed in a future release.',
            HarnessDeprecationWarning,
            stacklevel=2,
        )
        if self._declared_native is False:
            return 'local'
        return 'native' if self._declared_local is False else 'auto'

    @mode.setter
    def mode(self, mode: _Mode) -> None:
        self._check_mode(mode)
        warnings.warn(
            f'`Advisor.mode` is deprecated: {self._MODE_REPLACEMENTS[mode]}. '
            '`mode` will be removed in a future release.',
            HarnessDeprecationWarning,
            stacklevel=2,
        )
        previous = self.native, self.local
        self.native, self.local = self._MODE_DECLARATIONS[mode]
        try:
            self.__post_init__()
        except Exception:
            self.native, self.local = previous
            raise

    @classmethod
    def _check_mode(cls, mode: str) -> None:
        if mode not in cls._MODE_DECLARATIONS:
            raise ValueError("`Advisor.mode` must be 'auto', 'native', or 'local'")

    @classmethod
    def from_spec(
        cls,
        model: KnownModelName | str,
        *,
        native: bool = True,
        local: bool = True,
        max_uses: int | None = None,
        max_tokens: int | None = None,
        caching: Literal['5m', '1h'] | None = None,
        forward_history: bool = False,
        id: str | None = 'advisor',
        defer_loading: bool = False,
        description: str | None = None,
        mode: _Mode | None = None,
        **kwargs: Any,
    ) -> Advisor[Any]:
        """Build from an agent spec, covering the fields a spec can express.

        Every JSON-expressible parameter is named because that signature is what core reads to
        generate the spec's JSON schema. A spec names the model by string and can only switch the
        native and local halves on or off. `output_type` has no JSON form, so it is passed through
        `**kwargs`, which the schema omits. `mode` is deprecated, as on the constructor.
        """
        return cls(
            model,
            native=native,
            local=local,
            max_uses=max_uses,
            max_tokens=max_tokens,
            caching=caching,
            forward_history=forward_history,
            id=id,
            defer_loading=defer_loading,
            description=description,
            mode=mode,
            **kwargs,
        )

    def __post_init__(self) -> None:
        """Validate the declared configuration, then resolve `native` and `local`.

        Runs on construction and again on every merged or per-run copy, so a merge can't reach a
        combination no constructor accepts.
        """
        if self.max_uses is not None and self.max_uses < 1:
            raise ValueError('`Advisor.max_uses` must be at least 1')
        if self.max_tokens is not None and self.max_tokens < 1024:
            raise ValueError('`Advisor.max_tokens` must be at least 1024')
        if self.output_type is not str:
            if self.local is False:
                raise ValueError('`Advisor.output_type` is not supported with `local=False`')
            if self.native is not True and self.native is not False:
                raise UserError(
                    '`Advisor.output_type` is not supported by a native advisor tool, which returns text: '
                    'pass `native=False`, or drop `output_type`'
                )
        if self.native is True and self.local is False:
            native_provider, _ = self._parse_native_model(self.model)
            if native_provider is None:
                raise ValueError(
                    "`Advisor(local=False)` requires an 'anthropic:<model>' or 'openrouter:<model>' model name"
                )
            if native_provider == 'openrouter' and self.max_uses is not None:
                raise ValueError('`Advisor.max_uses` is not supported by OpenRouter with `local=False`')
        super().__post_init__()

    def _default_native(self) -> Callable[[RunContext[AgentDepsT]], AdvisorTool | None]:
        """Build the native advisor per run, since whether it applies depends on the executor."""
        return self._required_native_advisor if self.local is False else self._native_advisor

    def _default_local(self) -> Tool[AgentDepsT]:
        """Build the `advisor` tool, which consults `model` through a separate agent."""
        output_type, max_uses, forward_history = self.output_type, self.max_uses, self.forward_history

        async def advisor(ctx: RunContext[AgentDepsT], prompt: str) -> object:
            if max_uses is not None:
                if self._local_uses >= max_uses:
                    return 'Advisor consultation limit reached for this model request. Continue without further advice.'
                self._local_uses += 1

            history = ctx.messages[:-1] if forward_history else None
            return raise_retry(await self._consult(ctx, prompt, history)).output

        return Tool(
            advisor,
            description=(
                'Consult a stronger model about a difficult or high-impact decision. '
                + ('' if output_type is str else 'Returns a validated answer in the configured output format. ')
                + 'Include the complete question and all relevant context in `prompt` '
                'because conversation history may not be available to the advisor.'
            ),
        )

    def _resolve_local_strategy(self, name: str | bool) -> Tool[AgentDepsT] | AbstractToolset[AgentDepsT]:
        if name is True:
            return self._default_local()
        return super()._resolve_local_strategy(name)

    @durable_operation('consult')
    async def _consult(
        self, ctx: RunContext[AgentDepsT], prompt: str, message_history: list[ModelMessage] | None
    ) -> _Advice | RetryRequest:
        output_type = self.output_type
        advisor_agent = Agent(
            self.model,
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
            model_settings=ModelSettings(max_tokens=self.max_tokens) if self.max_tokens is not None else None,
        )
        try:
            result = await advisor_agent.run(
                prompt,
                message_history=message_history,
                conversation_id=ctx.conversation_id,
                usage=ctx.usage,
                usage_limits=ctx.usage_limits,
            )
        except UnexpectedModelBehavior as e:
            return RetryRequest(str(e))
        return _Advice(result.output)

    @classmethod
    def combine(cls, capabilities: Sequence[AbstractCapability[AgentDepsT]]) -> AbstractCapability[AgentDepsT]:
        """Merge like any native-or-local capability, keeping one combination that used to merge.

        No single `Advisor` accepts `local=False` on OpenRouter with `max_uses`, but merging used to
        send the native tool without `max_uses`. That merge keeps working, with a warning, rather than
        make an agent that used to build raise.
        """
        advisors = [capability for capability in capabilities if isinstance(capability, Advisor)]
        assert len(advisors) == len(capabilities)
        max_uses = cls._last_stated([advisor.max_uses for advisor in advisors])
        if (
            max_uses is not None
            and cls._last_stated([advisor._declared_local for advisor in advisors]) is False
            and cls._parse_native_model(advisors[-1].model)[0] == 'openrouter'
        ):
            warnings.warn(
                f'Merged `Advisor` capabilities combine `local=False` on OpenRouter with `max_uses={max_uses}`, '
                "which OpenRouter's native advisor does not support, so `max_uses` is ignored. "
                'This will raise `UserError` in a future release: remove `max_uses` or `local=False`.',
                HarnessDeprecationWarning,
                stacklevel=cls._caller_stacklevel(),
            )
            capabilities = [replace_no_init(advisor, max_uses=None) for advisor in advisors]
        return super().combine(capabilities)

    @staticmethod
    def _last_stated(values: Sequence[_T | None]) -> _T | None:
        """The value a merge keeps: the last one stated, where `None` means not stated."""
        stated = [value for value in values if value is not None]
        return stated[-1] if stated else None

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Self:
        """Return a fresh capability with local usage isolated to this run.

        The `advisor` tool counts consultations on the capability it was built from, so the copy
        builds its own.
        """
        if self.max_uses is None:
            return self
        fresh = replace_no_init(self, native=self._declared_native, local=self._declared_local)
        fresh._local_uses = 0
        fresh.__post_init__()
        return fresh

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
        provider = self._executor_service(ctx.model)
        if provider != native_provider or native_model_name is None or self.output_type is not str:
            return None
        # OpenRouter's native advisor ignores `max_uses`, so the local tool enforces it there. Core has
        # no `restricting_settings` for `AdvisorTool` yet, which is where this check belongs.
        if provider == 'openrouter' and self.max_uses is not None:
            return None
        return self._advisor_tool(native_model_name)

    def _required_native_advisor(self, ctx: RunContext[AgentDepsT]) -> AdvisorTool:
        """Build the native tool for `local=False`, rejecting a cross-provider executor.

        With no local fallback, omitting the native tool would leave the executor without an advisor.
        """
        native_provider, native_model_name = self._parse_native_model(self.model)
        if self._executor_service(ctx.model) != native_provider:
            raise UserError(f'`Advisor(local=False)` requires a {native_provider} executor, not {ctx.model.system}')
        assert native_model_name is not None
        return self._advisor_tool(native_model_name)

    def _advisor_tool(self, model_name: str) -> AdvisorTool:
        return AdvisorTool(
            model=model_name,
            max_uses=self.max_uses,
            max_tokens=self.max_tokens,
            provider_settings={'anthropic': {'caching': self.caching}} if self.caching is not None else None,
        )

    @staticmethod
    def _caller_stacklevel() -> int:
        """The `warnings.warn` stacklevel, from a function in this module, of the nearest frame in the user's code.

        Skips this module, Pydantic AI, Pydantic Graph, and the `asyncio`, `contextlib` and `typing`
        frames a call goes through, so the warning points at the caller whether they construct
        `Advisor[None](...)`, call `combine` directly, build an `Agent`, or pass capabilities to a run.
        """
        frame = inspect.currentframe()
        stacklevel = 0
        while frame is not None and (
            (module := frame.f_globals.get('__name__', '')) == __name__
            or module.partition('.')[0] in {'pydantic_ai', 'pydantic_graph', 'asyncio', 'contextlib', 'typing'}
        ):
            frame = frame.f_back
            stacklevel += 1
        return stacklevel

    @staticmethod
    def _is_request_response_model(model: AbstractModel) -> TypeIs[Model]:
        """Narrow a run's model without losing its provider client type."""
        return isinstance(model, Model)

    @staticmethod
    def _executor_service(model: AbstractModel) -> str:
        """The service the executor reaches, which decides the native advisor's model-ID namespace.

        The provider's canonical system rather than its name, so a provider subclass that only renames
        itself, such as a `claude-code` `AnthropicProvider`, still gets the Anthropic advisor. Not the
        model's, which `BedrockConverseModel` reports as `'anthropic'` for Claude's thinking signatures.
        """
        provider = model.provider if Advisor._is_request_response_model(model) else None
        return provider.canonical_system if provider is not None else model.system

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
