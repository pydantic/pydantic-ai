"""Parse and expand model request body overrides without terminal dependencies."""

from copy import deepcopy

from pydantic import JsonValue, TypeAdapter, ValidationError

from pydantic_clai2.models.profiles import provider_of

_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


def expand_params(*, pairs: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Expand dotted keys; later values replace earlier conflicting branches."""
    body: dict[str, JsonValue] = {}
    for key, value in pairs.items():
        parts = key.split('.')
        if any(not part.strip() for part in parts):
            raise ValueError('Parameter keys need nonempty dot-separated segments.')
        target = body
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                child = {}
                target[part] = child
            target = child
        target[parts[-1]] = deepcopy(value)
    return body


def parse_pair(*, text: str) -> tuple[str, JsonValue]:
    """JSON values retain their types; unquoted text stays a string."""
    key, separator, raw = (part.strip() for part in text.partition('='))
    if not separator or not key or not raw:
        raise ValueError('Expected format: key = value')
    try:
        value = _JSON.validate_json(raw)
    except ValidationError:
        value = raw
    expand_params(pairs={key: value})
    return key, value


def custom_effort_override(
    custom_params: dict[str, JsonValue], *, settings_as: str, key: str
) -> tuple[str, JsonValue] | None:
    """The custom body path and value replacing native effort, including scalar ancestors."""
    if key == 'anthropic_effort':
        path = ('output_config', 'effort')
    elif key == 'glm_reasoning_effort' or provider_of(settings_as) in ('openai-chat', 'openrouter', 'vllm'):
        path = ('reasoning_effort',)
    else:
        path = ('reasoning', 'effort')
    value: JsonValue = expand_params(pairs=custom_params)
    found: list[str] = []
    for part in path:
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
        found.append(part)
        # A scalar/null ancestor replaces the whole native reasoning/output_config object.
        if not isinstance(value, dict):
            break
    return '.'.join(found), value
