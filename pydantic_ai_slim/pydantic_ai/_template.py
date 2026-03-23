"""Template string support for dynamic instructions."""

from __future__ import annotations

import inspect
from typing import Any, Generic, cast, get_args, get_origin

from pydantic import GetCoreSchemaHandler, TypeAdapter
from pydantic_core import CoreSchema, core_schema

from pydantic_ai._run_context import RunContext
from pydantic_ai._utils import get_function_type_hints
from pydantic_ai.tools import AgentDepsT


class TemplateStr(Generic[AgentDepsT]):
    """A Handlebars template string that renders against `RunContext.deps`.

    When used in type hints, strings containing `{{` are automatically
    compiled as Handlebars templates during Pydantic validation.

    Uses [pydantic-handlebars](https://github.com/pydantic/pydantic-handlebars)
    for template compilation, schema validation, and rendering.

    When used with an `Agent`, `deps_type` is inferred automatically from
    the agent's validation context, so you only need to pass it when constructing
    a `TemplateStr` outside of an agent (e.g. for standalone rendering).

    Example:
        ```python {test="skip"}
        from dataclasses import dataclass

        from pydantic_ai import Agent, TemplateStr


        @dataclass
        class MyDeps:
            name: str

        agent = Agent(
            'openai:gpt-5',
            deps_type=MyDeps,
            instructions=TemplateStr('Hello {{name}}'),
        )
        ```
    """

    __slots__ = ('_source', '_deps_type', '_deps_schema', '_compiled_typed', '_compiled_untyped')

    def __init__(
        self,
        source: str,
        *,
        deps_type: type[Any] | None = None,
        deps_schema: dict[str, Any] | None = None,
    ) -> None:
        self._source = source
        self._deps_type = deps_type
        self._deps_schema = deps_schema

        hbs = _import_pydantic_handlebars()

        if deps_type is not None:
            self._compiled_typed = hbs.compile(source, deps_type)
            self._compiled_untyped = None
        else:
            if deps_schema is not None:
                hbs.check_template_compatibility(source, deps_schema, raise_on_error=True)
            self._compiled_typed = None
            self._compiled_untyped = hbs.compile(source)

    def render(self, deps: AgentDepsT | None = None) -> str:
        """Render the template against the given deps object."""
        if self._compiled_typed is not None:
            return self._compiled_typed.render(deps)

        assert self._compiled_untyped is not None
        if deps is not None:
            ta = TypeAdapter(type(deps))
            deps_data = ta.dump_python(deps, mode='python')
            if isinstance(deps_data, dict):
                return self._compiled_untyped.render(deps_data)
        return self._compiled_untyped.render()

    def __call__(self, ctx: RunContext[AgentDepsT]) -> str:
        """Render the template against `ctx.deps`."""
        return self.render(ctx.deps)

    @classmethod
    def __get_pydantic_core_schema__(
        cls,
        source_type: type[Any],
        handler: GetCoreSchemaHandler,
    ) -> CoreSchema:
        def validate(value: Any, info: core_schema.ValidationInfo) -> TemplateStr[Any]:
            if isinstance(value, TemplateStr):
                return cast(TemplateStr[Any], value)
            if not isinstance(value, str):
                raise ValueError(f'Expected string, got {type(value).__name__}')
            if '{{' not in value:
                # Intentional: in Union[TemplateStr, str], this validation failure causes Pydantic to fall through to the str branch
                raise ValueError('Not a template string (no {{ found)')

            context: dict[str, Any] = info.context or {}
            deps_type: type[Any] | None = context.get('deps_type')
            deps_schema: dict[str, Any] | None = context.get('deps_schema')

            return TemplateStr(value, deps_type=deps_type, deps_schema=deps_schema)

        return core_schema.with_info_plain_validator_function(
            validate,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda v: v._source if isinstance(v, TemplateStr) else v,
                info_arg=False,
            ),
        )

    def __repr__(self) -> str:
        return f'TemplateStr({self._source!r})'

    def __str__(self) -> str:
        return self._source


def validate_from_spec_args(
    cls: type[Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    validation_context: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Validate from_spec arguments, resolving TemplateStr types via Pydantic.

    Inspects the `from_spec` method's type hints to find parameters that accept
    TemplateStr. For those parameters, values are validated through Pydantic's
    `TypeAdapter`, which invokes `TemplateStr.__get_pydantic_core_schema__`
    to automatically compile template strings (containing `{{`) into TemplateStr
    instances using the deps_type/deps_schema from the validation context.
    """
    try:
        hints = get_function_type_hints(cls.from_spec)
    except Exception:
        return args, kwargs

    hints.pop('return', None)
    if not any(_hint_contains_template_str(h) for h in hints.values()):
        return args, kwargs

    sig = inspect.signature(cls.from_spec)
    params = [p for p in sig.parameters.values() if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]

    new_args = list(args)
    new_kwargs = dict(kwargs)

    for i, param in enumerate(params):
        hint = hints.get(param.name)
        if hint is None or not _hint_contains_template_str(hint):
            continue

        ta = TypeAdapter(hint)
        if i < len(args):
            new_args[i] = ta.validate_python(args[i], context=validation_context)
        elif param.name in kwargs:
            new_kwargs[param.name] = ta.validate_python(kwargs[param.name], context=validation_context)

    return tuple(new_args), new_kwargs


def _hint_contains_template_str(hint: Any) -> bool:
    """Check if a type hint includes TemplateStr."""
    if hint is TemplateStr or get_origin(hint) is TemplateStr:
        return True
    args = get_args(hint)
    if args:
        return any(_hint_contains_template_str(a) for a in args)
    return False


def _import_pydantic_handlebars() -> Any:
    """Lazily import pydantic-handlebars with a helpful error message."""
    try:
        import pydantic_handlebars

        return pydantic_handlebars
    except ImportError as e:  # pragma: no cover — optional dependency
        raise ImportError(
            'pydantic-handlebars is required for TemplateStr support. '
            'Install it with: pip install "pydantic-ai-slim[spec]"'
        ) from e
