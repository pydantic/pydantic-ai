"""Agent specification for constructing agents from YAML/JSON/dict specs."""

from __future__ import annotations

import functools
import inspect
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NotRequired, Union, cast, get_args, get_origin

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PydanticUserError,
    SecretStr,
    TypeAdapter,
    ValidationError,
    create_model,
    model_serializer,
    with_config,
)
from pydantic_core import from_json, to_json
from pydantic_core.core_schema import SerializationInfo, SerializerFunctionWrapHandler
from typing_extensions import TypedDict

from pydantic_ai._agent_graph import EndStrategy
from pydantic_ai._spec import CapabilitySpec, ProviderSpec, build_registry, build_schema_types
from pydantic_ai._utils import get_function_type_hints, is_str_dict
from pydantic_ai.agent.abstract import AgentRetries
from pydantic_ai.exceptions import UserError
from pydantic_ai.providers import Provider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.template import TemplateStr

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import AbstractCapability
    from pydantic_ai.models import Model

__all__ = ['CapabilitySpec', 'ProviderSpec']  # re-exported from _spec

DEFAULT_SCHEMA_PATH_TEMPLATE = './{stem}_schema.json'
"""Default template for schema file paths, where {stem} is replaced with the spec filename stem."""

_YAML_SCHEMA_LINE_PREFIX = '# yaml-language-server: $schema='


class ModelSpec(BaseModel, extra='forbid'):
    """A model with a configured provider, for the `model` field of an [`AgentSpec`][pydantic_ai.agent.AgentSpec].

    See [Models with a configured provider](../agent-spec.md#models-with-a-configured-provider).
    """

    name: str
    """The model name, in the `'<provider>:<model>'` form [`Agent`][pydantic_ai.agent.Agent] takes.

    Its prefix picks the model class, like `openai-chat` for
    [`OpenAIChatModel`][pydantic_ai.models.openai.OpenAIChatModel], as it does for a plain model name.
    """

    provider: ProviderSpec
    """The provider the model uses, by the class-style name of its class, like `OpenAICompatible`, or a provider name
    from a model name, like `azure` or `gateway/openai`, with its arguments."""


class AgentSpec(BaseModel):
    """Specification for constructing an Agent from a dict/YAML/JSON."""

    # $schema is included to avoid validation fails from the `$schema` key, see `_add_json_schema` below for context
    json_schema_path: str | None = Field(default=None, alias='$schema')
    model: str | ModelSpec | None = None
    name: str | None = None
    description: TemplateStr[Any] | str | None = None
    instructions: TemplateStr[Any] | str | list[TemplateStr[Any] | str] | None = None
    deps_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None
    model_settings: dict[str, Any] | None = None
    retries: int | AgentRetries | None = None
    end_strategy: EndStrategy = 'graceful'
    tool_timeout: float | None = None
    metadata: dict[str, Any] | None = None
    capabilities: list[CapabilitySpec] = []

    @classmethod
    def from_file(
        cls,
        path: Path | str,
        fmt: Literal['yaml', 'json'] | None = None,
    ) -> AgentSpec:
        """Load an agent spec from a YAML or JSON file.

        Args:
            path: Path to the file to load.
            fmt: Format of the file. If None, inferred from file extension.

        Returns:
            A new AgentSpec instance.
        """
        path = Path(path)
        fmt = _infer_fmt(path, fmt)
        content = path.read_text(encoding='utf-8')
        return cls.from_text(content, fmt=fmt)

    @classmethod
    def from_text(
        cls,
        text: str,
        fmt: Literal['yaml', 'json'] = 'yaml',
    ) -> AgentSpec:
        """Parse YAML or JSON text into an AgentSpec.

        Args:
            text: The string content to parse.
            fmt: Format of the content. Must be either 'yaml' or 'json'.

        Returns:
            A new AgentSpec instance.
        """
        if fmt == 'json':
            data = from_json(text)
        else:
            try:
                import yaml
            except ImportError:  # pragma: no cover
                # Reaching this requires PyYAML to not be installed.
                raise ImportError(
                    'PyYAML is required to load YAML agent specs. Install it with: pip install "pydantic-ai-slim[spec]"'
                ) from None
            data = yaml.safe_load(text)
        if not is_str_dict(data):
            raise UserError(f'Agent spec must parse to an object, got {type(data).__name__}')
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentSpec:
        """Validate a dictionary into an AgentSpec.

        Args:
            data: Dictionary representation of the agent spec.

        Returns:
            A new AgentSpec instance.
        """
        return cls.model_validate(data)

    def to_file(
        self,
        path: Path | str,
        fmt: Literal['yaml', 'json'] | None = None,
        schema_path: Path | str | None = DEFAULT_SCHEMA_PATH_TEMPLATE,
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        custom_provider_types: Sequence[type[Provider[Any]]] = (),
    ) -> None:
        """Save the agent spec to a YAML or JSON file.

        Args:
            path: Path to save the spec to.
            fmt: Format to use. If None, inferred from file extension.
            schema_path: Path to save the JSON schema to. If None, no schema will be saved.
                Can be a string template with {stem} which will be replaced with the spec filename stem.
            custom_capability_types: Custom capability classes to include in the schema.
            custom_provider_types: Custom provider classes to include in the schema.
        """
        path = Path(path)
        fmt = _infer_fmt(path, fmt)

        schema_ref: str | None = None
        if schema_path is not None:
            if isinstance(schema_path, str):
                schema_path = Path(schema_path.format(stem=path.stem))

            if not schema_path.is_absolute():
                schema_ref = str(schema_path)
                schema_path = path.parent / schema_path
            elif schema_path.is_relative_to(path.parent):
                schema_ref = str(schema_path.relative_to(path.parent))
            else:
                schema_ref = str(schema_path)
            self._save_schema(schema_path, custom_capability_types, custom_provider_types)

        context: dict[str, Any] = {'use_short_form': True}
        if fmt == 'yaml':
            try:
                import yaml
            except ImportError:  # pragma: no cover
                # Reaching this requires PyYAML to not be installed.
                raise ImportError(
                    'PyYAML is required to save YAML agent specs. Install it with: pip install "pydantic-ai-slim[spec]"'
                ) from None
            dumped_data = self.model_dump(mode='json', by_alias=True, context=context, exclude_defaults=True)
            content = yaml.dump(dumped_data, sort_keys=False, allow_unicode=True)
            if schema_ref:
                content = f'{_YAML_SCHEMA_LINE_PREFIX}{schema_ref}\n{content}'
            path.write_text(content, encoding='utf-8')
        else:
            context['$schema'] = schema_ref
            json_data = self.model_dump_json(indent=2, by_alias=True, context=context, exclude_defaults=True)
            path.write_text(json_data + '\n', encoding='utf-8')

    @model_serializer(mode='wrap')
    def _add_json_schema(self, nxt: SerializerFunctionWrapHandler, info: SerializationInfo) -> dict[str, Any]:
        """Add the JSON schema path to the serialized output when provided via context."""
        context = cast(dict[str, Any] | None, info.context)
        if isinstance(context, dict) and (schema := context.get('$schema')):
            return {'$schema': schema} | nxt(self)
        return nxt(self)

    @classmethod
    def model_json_schema_with_capabilities(
        cls,
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        custom_provider_types: Sequence[type[Provider[Any]]] = (),
    ) -> dict[str, Any]:
        """Generate a JSON schema for this agent spec type, including capability and provider details.

        This is useful for generating a schema that can be used to validate YAML-format agent spec files.

        Args:
            custom_capability_types: Custom capability classes to include in the schema.
            custom_provider_types: Custom provider classes to include in the schema.

        Returns:
            A dictionary representing the JSON schema.
        """
        capability_schema_types = _build_capability_schema_types(get_capability_registry(custom_capability_types))
        provider_schema_types = _build_provider_schema_types(custom_provider_types)

        # Named `ModelSpec` in the schema, like the class it describes.
        model_spec_schema = create_model(
            'ModelSpec',
            __config__=ConfigDict(extra='forbid'),
            name=str,
            provider=Union[tuple(provider_schema_types)],  # pyright: ignore[reportArgumentType]  # noqa: UP007
        )

        # Build a schema-only model with the resolved capability union.
        # NOTE: This duplicates the field list from AgentSpec above. We can't inherit from
        # AgentSpec because the types intentionally differ for schema generation:
        # - TemplateStr is replaced with plain str (templates are just strings in YAML/JSON)
        # - capabilities uses a resolved Union of typed schema models instead of CapabilitySpec
        # - a model spec's provider uses a resolved Union of typed provider schemas instead of ProviderSpec
        # - extra='forbid' enables strict validation in the generated schema
        # When adding or removing fields on AgentSpec, update this class to match.
        class _AgentSpecSchema(BaseModel, extra='forbid', arbitrary_types_allowed=True):
            model: str | model_spec_schema | None = None  # pyright: ignore[reportInvalidTypeForm]
            name: str | None = None
            description: str | None = None
            instructions: str | list[str] | None = None
            deps_schema: dict[str, Any] | None = None
            output_schema: dict[str, Any] | None = None
            model_settings: ModelSettings | None = None
            retries: int | AgentRetries | None = None
            end_strategy: EndStrategy = 'graceful'
            tool_timeout: float | None = None
            metadata: dict[str, Any] | None = None
            if capability_schema_types:  # pragma: no branch
                capabilities: list[Union[tuple(capability_schema_types)]] = []  # pyright: ignore[reportUnknownVariableType, reportInvalidTypeArguments, reportInvalidTypeForm]  # noqa: UP007

        json_schema = _AgentSpecSchema.model_json_schema()
        json_schema['title'] = 'AgentSpec'
        json_schema['properties']['$schema'] = {'type': 'string'}

        # ModelSettings should allow additional properties for provider-specific settings;
        # extra='forbid' on _AgentSpecSchema propagates additionalProperties:false to nested
        # types, so we remove it from ModelSettings.
        model_settings_def: dict[str, Any] = json_schema.get('$defs', {}).get('ModelSettings', {})
        model_settings_def.pop('additionalProperties', None)

        # Inline the single-key object for each provider name, so a provider's names share its arguments' $def
        # without adding one of their own.
        defs: dict[str, Any] = json_schema['$defs']
        provider_items: list[dict[str, Any]] = defs['ModelSpec']['properties']['provider']['anyOf']
        for i, item in enumerate(provider_items):
            if (ref := item.get('$ref', '')).startswith('#/$defs/provider_spec_'):
                provider_items[i] = defs.pop(ref.removeprefix('#/$defs/'))
                provider_items[i].pop('title')

        # Replace CapabilitySpec $refs with the capability items Union,
        # so nested capability fields (e.g. PrefixTools.capability) show
        # the same rich schema as the top-level capabilities array.
        cap_items_schema = json_schema['properties']['capabilities']['items']
        _replace_capability_spec_refs(json_schema, cap_items_schema)

        return json_schema

    @classmethod
    def _save_schema(
        cls,
        path: Path | str,
        custom_capability_types: Sequence[type[AbstractCapability[Any]]] = (),
        custom_provider_types: Sequence[type[Provider[Any]]] = (),
    ) -> None:
        """Save the JSON schema for this agent spec type to a file.

        Args:
            path: Path to save the schema to.
            custom_capability_types: Custom capability classes to include in the schema.
            custom_provider_types: Custom provider classes to include in the schema.
        """
        path = Path(path)
        json_schema = cls.model_json_schema_with_capabilities(custom_capability_types, custom_provider_types)
        schema_content = to_json(json_schema, indent=2).decode() + '\n'
        if not path.exists() or path.read_text(encoding='utf-8') != schema_content:
            path.write_text(schema_content, encoding='utf-8')


def _infer_fmt(path: Path, fmt: Literal['yaml', 'json'] | None) -> Literal['yaml', 'json']:
    """Infer the format to use for a file based on its extension."""
    if fmt is not None:
        return fmt
    suffix = path.suffix.lower()
    if suffix in {'.yaml', '.yml'}:
        return 'yaml'
    elif suffix == '.json':
        return 'json'
    raise ValueError(
        f'Could not infer format for filename {path.name!r}. Use the `fmt` argument to specify the format.'
    )


def get_capability_registry(
    custom_types: Sequence[type[AbstractCapability[Any]]] = (),
) -> Mapping[str, type[AbstractCapability[Any]]]:
    """Create a registry of capability types from default and custom types."""
    from pydantic_ai.capabilities import CAPABILITY_TYPES
    from pydantic_ai.capabilities.abstract import AbstractCapability

    def _validate_capability(cls: type[AbstractCapability[Any]]) -> None:
        if not issubclass(cls, AbstractCapability):
            raise ValueError(
                f'All custom capability classes must be subclasses of AbstractCapability, but {cls} is not'
            )
        if '__dataclass_fields__' not in cls.__dict__:
            raise ValueError(f'All custom capability classes must be decorated with `@dataclass`, but {cls} is not')

    return build_registry(
        custom_types=custom_types,
        defaults=tuple(CAPABILITY_TYPES.values()),
        get_name=lambda cls: cls.get_serialization_name(),
        label='capability',
        validate=_validate_capability,
    )


class CapabilitySpecContext:
    """Holds the registry and instantiation callback for the current spec-loading scope."""

    __slots__ = ('registry', 'instantiate')

    def __init__(
        self,
        registry: Mapping[str, type[AbstractCapability[Any]]],
        instantiate: Callable[
            [type[AbstractCapability[Any]], tuple[Any, ...], dict[str, Any]], AbstractCapability[Any]
        ],
    ) -> None:
        self.registry = registry
        self.instantiate = instantiate


capability_spec_context: ContextVar[CapabilitySpecContext | None] = ContextVar('capability_spec_context', default=None)


def load_capability_from_nested_spec(spec: CapabilitySpec | dict[str, Any] | str) -> AbstractCapability[Any]:
    """Load a capability from a nested spec, reusing the current spec-loading context.

    When called inside `Agent.from_spec()` or `Agent._resolve_spec()`, this uses the same
    registry (including custom capability types) and template context as the outer loading.
    When called outside a spec-loading context, falls back to the default registry.

    This is intended for use in `from_spec()` methods of wrapper capabilities like
    [`PrefixTools`][pydantic_ai.capabilities.PrefixTools] that need to instantiate
    a nested capability from a spec argument.
    """
    from pydantic_ai._spec import load_from_registry

    cap_spec = spec if isinstance(spec, CapabilitySpec) else CapabilitySpec.model_validate(spec)
    ctx = capability_spec_context.get()
    if ctx is not None:
        return load_from_registry(
            ctx.registry,
            cap_spec,
            label='capability',
            custom_types_param='custom_capability_types',
            instantiate=ctx.instantiate,
        )
    else:
        return load_from_registry(
            get_capability_registry(),
            cap_spec,
            label='capability',
            custom_types_param='custom_capability_types',
            instantiate=lambda cap_cls, args, kwargs: cap_cls.from_spec(*args, **kwargs),
        )


def _build_capability_schema_types(registry: Mapping[str, type[Any]]) -> list[Any]:
    """Build a list of schema types for capabilities from a registry."""

    def _get_schema_target(cls: type[Any]) -> Any:
        # When from_spec is not overridden, it delegates to cls(*args, **kwargs).
        # Use __init__ directly so build_schema_types sees the actual parameter types.
        # Fall back to from_spec if __init__ hints can't be resolved (e.g. TYPE_CHECKING imports).
        if 'from_spec' not in cls.__dict__:
            try:
                get_function_type_hints(cls.__init__)
                return cls.__init__
            except (NameError, TypeError, AttributeError):
                pass
        return cls.from_spec

    return build_schema_types(
        registry,
        get_schema_target=_get_schema_target,
    )


def _replace_capability_spec_refs(schema: dict[str, Any], cap_items_schema: dict[str, Any]) -> None:
    """Walk the schema and replace any $ref to CapabilitySpec with the capability items Union."""
    cap_ref = '#/$defs/CapabilitySpec'

    if schema.get('$ref') == cap_ref:
        schema.clear()
        schema.update(cap_items_schema)
        return
    for value in schema.values():
        if isinstance(value, dict):
            _replace_capability_spec_refs(cast(dict[str, Any], value), cap_items_schema)
        elif isinstance(value, list):
            for item in value:  # pyright: ignore[reportUnknownVariableType]
                if isinstance(item, dict):
                    _replace_capability_spec_refs(cast(dict[str, Any], item), cap_items_schema)

    # Clean up the CapabilitySpec $def entry
    defs: dict[str, Any] = schema.get('$defs', {})
    defs.pop('CapabilitySpec', None)


_PROVIDER_CLASS_KEYS: dict[str, str] = {
    'openai': 'OpenAI',
    'openai-chat': 'OpenAI',
    'openai-responses': 'OpenAI',
    'openai-decisions': 'OpenAI',
    'openai-codex': 'OpenAICodex',
    'openai-compatible': 'OpenAICompatible',
    'deepseek': 'DeepSeek',
    'openrouter': 'OpenRouter',
    'vercel': 'Vercel',
    'azure': 'Azure',
    'azure-responses': 'Azure',
    'google': 'Google',
    'google-cloud': 'GoogleCloud',
    'bedrock': 'Bedrock',
    'bedrock-mantle': 'BedrockMantle',
    'groq': 'Groq',
    'anthropic': 'Anthropic',
    'mistral': 'Mistral',
    'cerebras': 'Cerebras',
    'cohere': 'Cohere',
    'system-one': 'SystemOne',
    'crusoe': 'Crusoe',
    'elevenlabs': 'ElevenLabs',
    'xai': 'Xai',
    'moonshotai': 'MoonshotAI',
    'meta': 'Meta',
    'fireworks': 'Fireworks',
    'together': 'Together',
    'heroku': 'Heroku',
    'huggingface': 'HuggingFace',
    'ollama': 'Ollama',
    'github-copilot': 'GitHubCopilot',
    'litellm': 'LiteLLM',
    'vllm': 'VLLM',
    'nebius': 'Nebius',
    'ovhcloud': 'OVHcloud',
    'alibaba': 'Alibaba',
    'sambanova': 'SambaNova',
    'sentence-transformers': 'SentenceTransformers',
    'snowflake': 'Snowflake',
    'typesafe': 'TypeSafe',
    'voyageai': 'VoyageAI',
    'zai': 'Zai',
}
"""The provider names a spec takes, with the class-style name of the built-in provider class each resolves to.

These are the names `infer_provider` takes, plus `'openai-compatible'`, which only specs take. Left out are
deprecated providers and `'openai-chatgpt'`, whose provider needs credentials a spec can't hold. Listing the class
names lets a class be imported on its own when a spec names it.
"""


@with_config(ConfigDict(extra='forbid'))
class EnvVar(TypedDict):
    """A reference to an environment variable, which is read when the model is built."""

    env: str
    """The name of the environment variable."""


def model_from_spec(model: str | ModelSpec, custom_provider_types: Sequence[type[Provider[Any]]] = ()) -> Model | str:
    """Turn a spec's model into what `Agent` takes: a model name stays a name, and a model spec is built.

    Building a model spec creates its provider, reading the environment variables its arguments reference.

    Raises:
        UserError: If the model spec is invalid, an environment variable it references is not set,
            or its provider can't be used with its model class.
    """
    if isinstance(model, str):
        return model

    from pydantic_ai.models import infer_model, parse_model_id
    from pydantic_ai.providers import infer_provider_class

    provider_name, _ = parse_model_id(model.name)
    if provider_name is None:
        raise UserError(f'A model spec `name` must have the form `<provider>:<model>`, got {model.name!r}.')
    try:
        default_provider_class = infer_provider_class(provider_name)
    except ValueError:
        raise UserError(f'Unknown model: {model.name}') from None

    provider = _load_provider(model.provider, custom_provider_types)
    client_types = _provider_client_types(type(provider))
    default_client_types = _provider_client_types(default_provider_class)
    if (
        client_types is not None
        and default_client_types is not None
        and not all(issubclass(client_type, default_client_types) for client_type in client_types)
    ):
        raise UserError(
            f'Provider {model.provider.name!r} ({type(provider).__name__}) cannot be used with model {model.name!r},'
            f' which needs a provider like `{default_provider_class.__name__}`.'
        )
    return infer_model(model.name, provider_factory=lambda _: provider)


def _load_provider(spec: ProviderSpec, custom_provider_types: Sequence[type[Provider[Any]]]) -> Provider[Any]:
    """Create the provider a provider spec describes, reading the environment variables its secret arguments reference."""
    name = spec.name
    if spec.args:
        raise UserError(f'The arguments of provider {name!r} in a spec must be given by name, as an object.')

    custom_registry = _get_custom_provider_registry(custom_provider_types)
    target: Callable[..., Provider[Any]]
    if name in custom_registry:
        target = custom_registry[name].from_spec
        params = _provider_spec_params(_provider_spec_target(custom_registry[name]))
    elif name.startswith('gateway/'):
        target = functools.partial(_gateway_provider_from_spec, name.removeprefix('gateway/'))
        params = _provider_spec_params(_gateway_provider_from_spec)
    else:
        provider_name = name if name in _PROVIDER_CLASS_KEYS else _provider_name_for_class_key(name)
        if provider_name is None:
            valid_names = [*custom_registry, *dict.fromkeys(_PROVIDER_CLASS_KEYS.values()), *_PROVIDER_CLASS_KEYS]
            raise UserError(
                f'Provider {name!r} is not a built-in provider or in the provided `custom_provider_types`.'
                f' Valid choices: {valid_names}, or `gateway/<upstream provider>`.'
            )
        provider_class = _builtin_provider_class(provider_name)
        target = provider_class.from_spec
        params = _provider_spec_params(_provider_spec_target(provider_class))

    kwargs = {
        param: _resolve_env_var(name, param, value, secret=param in params and _is_secret_type(params[param][0]))
        for param, value in spec.kwargs.items()
    }
    params_typed_dict, _ = _provider_params_typed_dict(params, 'ProviderArguments', for_schema=False)
    try:
        kwargs = TypeAdapter(params_typed_dict).validate_python(kwargs)
    except ValidationError as e:
        raise UserError(f'Invalid arguments for provider {name!r}: {e}') from e
    return target(**kwargs)


def _gateway_provider_from_spec(
    upstream_provider: str,
    /,
    *,
    route: str | None = None,
    api_key: SecretStr | None = None,
    base_url: str | None = None,
) -> Provider[Any]:
    """Create a Gateway provider from the arguments in a spec, which can't set a client."""
    from pydantic_ai.providers.gateway import gateway_provider

    return gateway_provider(upstream_provider, route=route, api_key=api_key, base_url=base_url)


def _resolve_env_var(provider: str, param: str, value: Any, *, secret: bool) -> Any:
    """The value for a provider argument, reading the environment variable a secret argument references."""
    is_env_var = is_str_dict(value) and value.keys() == {'env'} and isinstance(value['env'], str)
    if not secret:
        if is_env_var:
            raise UserError(
                f'Argument {param!r} of provider {provider!r} is not a secret, so it cannot be read from an environment variable.'
            )
        return value
    if value is None:
        return None
    if not is_env_var:
        raise UserError(
            f'Argument {param!r} of provider {provider!r} is a secret, so a spec can only reference an environment'
            ' variable for it, as `{env: VAR_NAME}`.'
        )
    env_var = cast(str, value['env'])
    if not (secret_value := os.environ.get(env_var)):
        raise UserError(
            f'The environment variable {env_var!r} referenced by argument {param!r} of provider {provider!r} is not set or empty.'
        )
    return SecretStr(secret_value)


def _provider_name_for_class_key(class_key: str) -> str | None:
    """A provider name that resolves to the built-in provider class with this class-style name, if any."""
    return next((name for name, key in _PROVIDER_CLASS_KEYS.items() if key == class_key), None)


def _builtin_provider_class(provider_name: str) -> type[Provider[Any]]:
    """The built-in provider class a provider name in a spec resolves to."""
    if provider_name == 'openai-compatible':
        from pydantic_ai.providers.openai_compatible import OpenAICompatibleProvider

        return OpenAICompatibleProvider

    from pydantic_ai.providers import infer_provider_class

    return infer_provider_class(provider_name)


def _get_custom_provider_registry(
    custom_types: Sequence[type[Provider[Any]]],
) -> Mapping[str, type[Provider[Any]]]:
    """Create a registry of custom provider types by their class-style names."""

    def _validate_provider(cls: type[Provider[Any]]) -> None:
        if not issubclass(cls, Provider):
            raise ValueError(f'All custom provider classes must be subclasses of Provider, but {cls} is not')
        name = cls.get_serialization_name()
        if name is not None and (name in _PROVIDER_CLASS_KEYS or name.startswith('gateway/')):
            raise ValueError(
                f'Custom provider class {cls.__name__} has the name {name!r}, which is a provider name.'
                ' Custom providers need class-style names, like the default derived from the class name.'
            )

    return build_registry(
        custom_types=custom_types,
        defaults=(),
        get_name=lambda cls: cls.get_serialization_name(),
        label='provider',
        validate=_validate_provider,
    )


def _provider_client_types(provider_class: type[Provider[Any]]) -> tuple[type[Any], ...] | None:
    """The client classes in the `Provider[...]` base of a provider class, or `None` if it doesn't declare classes.

    A union of client types, like `AnthropicProvider`'s, gives each of its members. A type variable, a forward reference
    or a parametrized type isn't a class to compare, so it makes the client types unknown.
    """
    client_type: Any = next(
        (
            get_args(base)[0]
            for cls in provider_class.__mro__
            for base in cls.__dict__.get('__orig_bases__', ())
            if get_origin(base) is Provider
        ),
        Any,
    )
    if client_type is Any:
        return None
    client_types = get_args(client_type) or (client_type,)
    return client_types if all(isinstance(member, type) for member in client_types) else None


def _provider_spec_target(provider_class: type[Provider[Any]]) -> Callable[..., Any]:
    """The function whose parameters are a provider's spec arguments: `from_spec` if it's overridden, `__init__` otherwise."""
    if provider_class.from_spec.__func__ is Provider.from_spec.__func__:
        return provider_class.__init__
    return provider_class.from_spec


def _is_secret_type(tp: Any) -> bool:
    return tp is SecretStr or SecretStr in get_args(tp)


def _has_json_schema(tp: Any) -> bool:
    try:
        TypeAdapter(tp).json_schema()
    except PydanticUserError:
        return False
    return True


def _provider_spec_params(target: Callable[..., Any]) -> dict[str, tuple[Any, bool]]:
    """The parameters of a provider's spec target that a spec can set, with their types and whether they're required.

    These are its secret parameters and those whose types have a JSON schema, which leaves out client objects.
    Positional-only parameters, like the Gateway's upstream provider, are left out too.
    """
    hints = get_function_type_hints(target)
    params: dict[str, tuple[Any, bool]] = {}
    for p in inspect.signature(target).parameters.values():
        if p.name == 'self' or p.kind in (p.POSITIONAL_ONLY, p.VAR_POSITIONAL, p.VAR_KEYWORD) or p.name not in hints:
            continue
        tp = hints[p.name]
        if _is_secret_type(tp) or _has_json_schema(tp):
            params[p.name] = (tp, p.default is p.empty)
    return params


def _provider_params_typed_dict(
    params: dict[str, tuple[Any, bool]], title: str, *, for_schema: bool
) -> tuple[Any, bool]:
    """A TypedDict of a provider's spec arguments, and whether any argument is required.

    For the JSON schema, secret arguments are typed as an `EnvVar` reference; otherwise they keep their type, to
    validate the `SecretStr` read from the environment variable.
    """
    fields: dict[str, Any] = {}
    for param, (tp, required) in params.items():
        if for_schema and _is_secret_type(tp):
            tp = EnvVar | None if type(None) in get_args(tp) else EnvVar
        fields[param] = tp if required else NotRequired[tp]
    td = TypedDict(title, fields)  # pyright: ignore[reportArgumentType]
    has_required = any(required for _, required in params.values())
    return with_config(ConfigDict(extra='forbid'))(td), has_required


def _build_provider_schema_types(custom_provider_types: Sequence[type[Provider[Any]]]) -> list[Any]:
    """Build the schema types for a model spec's provider: one per provider name, sharing one TypedDict per class."""
    from pydantic_ai.providers.gateway import UpstreamProvider

    builtin_classes: dict[str, type[Provider[Any]]] = {}
    for provider_name, class_key in _PROVIDER_CLASS_KEYS.items():
        if class_key in builtin_classes:
            continue
        # Providers whose optional dependencies aren't installed are left out. Their SDKs are only imported to read
        # the providers' signatures, so their import-time warnings, such as deprecations, aren't for the schema to show.
        with suppress(ImportError), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            builtin_classes[class_key] = _builtin_provider_class(provider_name)

    schema_types: list[Any] = []
    custom_registry = _get_custom_provider_registry(custom_provider_types)
    for class_key, provider_class in builtin_classes.items():
        # A provider name always resolves to the built-in class, also when a custom class took over its class-style name.
        names = [name for name, key in _PROVIDER_CLASS_KEYS.items() if key == class_key]
        if class_key not in custom_registry:
            names.insert(0, class_key)
        schema_types.extend(
            _provider_schema_entries(names, _provider_spec_target(provider_class), f'{provider_class.__name__}Spec')
        )
    for class_key, provider_class in custom_registry.items():
        schema_types.extend(
            _provider_schema_entries(
                [class_key], _provider_spec_target(provider_class), f'{provider_class.__name__}Spec'
            )
        )

    upstream_providers = [arg for literal in get_args(UpstreamProvider) for arg in get_args(literal)]
    gateway_names = [f'gateway/{upstream}' for upstream in upstream_providers]
    schema_types.extend(_provider_schema_entries(gateway_names, _gateway_provider_from_spec, 'GatewayProviderSpec'))
    return schema_types


def _provider_schema_entries(names: Sequence[str], target: Callable[..., Any], title: str) -> list[Any]:
    """The schema types for provider names that resolve to the same class or function, sharing its arguments TypedDict."""
    params_td, has_required = _provider_params_typed_dict(_provider_spec_params(target), title, for_schema=True)
    entries: list[Any] = []
    for name in names:
        if not has_required:
            entries.append(Literal[name])
        if params_td.__annotations__:
            fields = {name: params_td}
            td = TypedDict(f'provider_spec_{name}', fields)  # pyright: ignore[reportArgumentType]
            entries.append(with_config(ConfigDict(extra='forbid'))(td))
    return entries
