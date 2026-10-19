"""Clairic schema and input translation."""

from __future__ import annotations

import json
import re
from datetime import date
from math import isfinite

from pydantic import JsonValue, TypeAdapter, ValidationError

from .._json_schema import InlineDefsJsonSchemaTransformer
from ..exceptions import UserError
from ..messages import (
    BinaryContent,
    ImageUrl,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from ..output import OutputObjectDefinition
from . import download_item


async def clairic_state(messages: list[ModelMessage]) -> tuple[str, list[str]]:
    history: list[dict[str, str]] = []
    text_present = False
    images: list[str] = []
    for message in messages:
        for part in message.parts:
            if isinstance(message, ModelRequest) and isinstance(part, SystemPromptPart):
                text_present = True
                history.append({'role': 'system', 'content': part.content})
            elif isinstance(message, ModelRequest) and isinstance(part, UserPromptPart):
                content = [part.content] if isinstance(part.content, str) else part.content
                for item in content:
                    if isinstance(item, str):
                        text_present = text_present or bool(item.strip())
                        history.append({'role': 'user', 'content': item})
                    elif isinstance(item, (BinaryContent, ImageUrl)):
                        if isinstance(item, ImageUrl):
                            downloaded = await download_item(item, data_format='bytes')
                            image = BinaryContent(data=downloaded['data'], media_type=downloaded['data_type'])
                        else:
                            image = item
                        if image.media_type not in ('image/png', 'image/jpeg', 'image/webp'):
                            raise UserError('Cloudflare Clairic supports only PNG, JPEG, and WebP images.')
                        images.append(image.data_uri)
                        history.append({'role': 'user', 'content': f'<image {len(images)}>'})
                    else:
                        raise UserError(f'Cloudflare Clairic does not support prompt content {type(item).__name__!r}.')
            elif isinstance(message, ModelRequest) and isinstance(part, RetryPromptPart):
                text_present = True
                history.append({'role': 'user', 'content': part.model_response()})
            elif isinstance(message, ModelResponse) and isinstance(part, TextPart):
                text_present = True
                history.append({'role': 'assistant', 'content': part.content})
            else:
                raise UserError(f'Cloudflare Clairic does not support history part {type(part).__name__!r}.')
    if len(images) > 4:
        raise UserError('Cloudflare Clairic accepts at most four images.')
    state = (
        history[0]['content']
        if len(history) == 1 and history[0]['role'] == 'user'
        else json.dumps(history, ensure_ascii=False)
        if history
        else ''
    )
    if images and not text_present:
        state = ''
    if not state.strip() and not images:
        raise UserError('Cloudflare Clairic requires non-empty input text or images.')
    return state, images


def clairic_fields(
    output: OutputObjectDefinition, instruction_parts: list[InstructionPart]
) -> tuple[dict[str, dict[str, JsonValue]], dict[str, str]]:
    try:
        schema = _schema_adapter.validate_python(InlineDefsJsonSchemaTransformer(output.json_schema).walk())
    except UserError as e:
        raise UserError(f'Unsupported Cloudflare Clairic schema: {e}') from e
    properties = schema.get('properties')
    if (
        schema.get('type') != 'object'
        or not isinstance(properties, dict)
        or not properties
        or set(schema) - {'type', 'title', 'description', 'properties', 'required', 'additionalProperties'}
        or schema.get('additionalProperties', False) is not False
    ):
        raise UserError('Cloudflare Clairic requires a non-empty, flat output object with supported fields.')
    field_names: dict[str, str] = {name: name for name in properties}
    if any(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_.-]{0,63}', name) is None for name in properties):
        field_names = {f'f{index}': name for index, name in enumerate(properties)}
    fields: dict[str, dict[str, JsonValue]] = {}
    instruction_text = '\n\n'.join(part.content for part in instruction_parts)
    for wire_name, name in field_names.items():
        prop = properties[name]
        if not isinstance(prop, dict):
            raise UserError(f'Unsupported Cloudflare Clairic schema for field {name!r}.')
        spec = clairic_field(prop, name)
        instructions = [instruction_text, output.description, f'Field: {name}']
        if isinstance(description := schema.get('description'), str):
            instructions.append(description)
        if isinstance(description := prop.get('description'), str):
            instructions.append(description)
        spec['instructions'] = '\n\n'.join(instruction for instruction in instructions if instruction)
        fields[wire_name] = spec
    return fields, field_names


def clairic_field(schema: dict[str, JsonValue], name: str) -> dict[str, JsonValue]:  # noqa: C901
    """Compile native controls without weakening Python output constraints."""
    controls = schema.get('x-cloudflare-clairic', {})
    if not isinstance(controls, dict):
        raise UserError(f'Invalid `ClairicField` controls for field {name!r}.')
    if 'anyOf' in schema:
        variants = schema['anyOf']
        if isinstance(variants, list) and len(variants) == 2:
            non_null = [variant for variant in variants if variant != {'type': 'null'}]
            if len(non_null) == 1 and isinstance(non_null[0], dict):
                outer = {key: value for key, value in schema.items() if key != 'anyOf'}
                if controls and 'x-cloudflare-clairic' in non_null[0]:
                    raise UserError(f'Use one `ClairicField` marker per field: {name!r}.')
                return clairic_field({**outer, **non_null[0]}, name)
        raise UserError(f'Cloudflare Clairic supports only nullable unions for field {name!r}.')

    result_type = schema.get('x-cloudflare-clairic-result')
    spec: dict[str, JsonValue]
    if result_type in ('number-estimate', 'date-estimate', 'forecast'):
        spec = {'type': 'forecast' if result_type == 'forecast' else 'estimate'}
    else:
        unsupported = set(schema) - {
            'type',
            'title',
            'description',
            'default',
            'format',
            'items',
            'x-cloudflare-clairic',
        }
        if unsupported:
            raise UserError(
                f'Unsupported Cloudflare Clairic schema keywords for field {name!r}: {sorted(unsupported)}.'
            )
        field_type = schema.get('type')
        if field_type == 'string' and schema.get('format') in (None, 'date', 'time', 'date-time', 'duration'):
            format = schema.get('format')
            spec = {'type': 'text' if format is None else 'datetime' if format == 'date-time' else format}
            if format == 'duration':
                spec['unit'] = 'seconds'
        elif field_type in ('integer', 'number'):
            spec = {'type': 'number'}
            if field_type == 'integer':
                spec['integer'] = True
        elif field_type == 'array' and isinstance(items := schema.get('items'), dict):
            item_spec = clairic_field(items, name)
            if 'anyOf' in items or item_spec.keys() != {'type'} or item_spec['type'] not in ('text', 'number', 'date'):
                raise UserError(f'Cloudflare Clairic requires text, float or date list items for field {name!r}.')
            spec = {'type': 'list', 'items': item_spec['type']}
        else:
            raise UserError(f'Unsupported Cloudflare Clairic type for field {name!r}.')

    kind = controls.get('kind', spec['type'])
    if kind != spec['type'] and not (spec['type'] == 'number' and kind == 'duration'):
        raise UserError(f'`ClairicField.kind` is incompatible with the result type for field {name!r}.')
    spec['type'] = kind
    allowed = {'kind'}
    numeric = kind in ('number', 'duration', 'forecast') or result_type == 'number-estimate'
    number_list = kind == 'list' and spec['items'] == 'number'
    if numeric or number_list:
        allowed.add('unit')
    if kind in ('number', 'duration'):
        allowed.update(('integer', 'decimals'))
    if number_list:
        allowed.add('decimals')
    if kind in ('text', 'number', 'date', 'time', 'datetime', 'duration'):
        allowed.add('null_threshold')
    if kind in ('estimate', 'forecast'):
        allowed.update(('min', 'max', 'step', 'scale'))
    if kind == 'forecast':
        allowed.update(('horizon', 'interval'))
    if unsupported := controls.keys() - allowed:
        raise UserError(f'Incompatible `ClairicField` controls for field {name!r}: {sorted(unsupported)}.')
    for key, value in controls.items():
        if key != 'kind':
            spec[key] = value

    if 'integer' in spec and not isinstance(spec['integer'], bool):
        raise UserError(f'`ClairicField.integer` must be a boolean for field {name!r}.')
    if schema.get('type') == 'integer' and spec.get('integer') is False:
        raise UserError(f'`ClairicField.integer=False` conflicts with `int` field {name!r}.')
    if 'decimals' in spec:
        decimals = spec['decimals']
        if isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 6:
            raise UserError(f'`ClairicField.decimals` must be an integer from zero to six for field {name!r}.')
        if spec.get('integer') is True and decimals != 0:
            raise UserError(f'`ClairicField.decimals` conflicts with integer precision for field {name!r}.')
    if 'null_threshold' in spec:
        threshold = spec['null_threshold']
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold <= 1:
            raise UserError(
                f'`ClairicField.null_threshold` must be greater than zero and at most one for field {name!r}.'
            )
    if 'unit' in spec and not isinstance(spec['unit'], str):
        raise UserError(f'`ClairicField.unit` must be a string for field {name!r}.')
    if kind == 'duration':
        unit = spec.setdefault('unit', 'seconds')
        if unit not in ('seconds', 'minutes', 'hours', 'days', 'weeks', 'months', 'years'):
            raise UserError(f'Invalid duration unit for field {name!r}.')
        if schema.get('format') == 'duration' and unit != 'seconds':
            raise UserError(f'Use seconds for `timedelta` field {name!r}; use a numeric duration for other units.')
    if kind in ('estimate', 'forecast'):
        lower, upper = spec.get('min'), spec.get('max')
        if result_type == 'date-estimate':
            if not isinstance(lower, str) or not isinstance(upper, str):
                raise UserError(f'Date estimate field {name!r} requires date `min` and `max`.')
            try:
                min_date = _date_adapter.validate_python(lower)
                max_date = _date_adapter.validate_python(upper)
            except ValidationError as e:
                raise UserError(f'Date estimate field {name!r} requires date `min` and `max`.') from e
            if min_date >= max_date:
                raise UserError(f'`ClairicField.min` must be less than `max` for field {name!r}.')
        elif (
            isinstance(lower, bool)
            or not isinstance(lower, (int, float))
            or not isfinite(lower)
            or isinstance(upper, bool)
            or not isinstance(upper, (int, float))
            or not isfinite(upper)
            or lower >= upper
        ):
            raise UserError(f'Field {name!r} requires finite numeric `min` less than `max`.')
        if 'step' in spec:
            step = spec['step']
            if not (
                (result_type == 'date-estimate' and step in ('month', 'year'))
                or (not isinstance(step, bool) and isinstance(step, (int, float)) and isfinite(step) and step > 0)
            ):
                raise UserError(f'Invalid `ClairicField.step` for field {name!r}.')
        scale = spec.get('scale', 'linear')
        if scale not in ('linear', 'log'):
            raise UserError(f'Invalid `ClairicField.scale` for field {name!r}.')
        if scale == 'log' and (result_type == 'date-estimate' or not isinstance(lower, (int, float)) or lower <= 0):
            raise UserError(f'Log scale requires positive numeric bounds for field {name!r}.')
    if kind == 'forecast':
        horizon = spec.get('horizon')
        if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 64:
            raise UserError(f'Forecast field {name!r} requires an integer `horizon` from one to 64.')
        if spec.get('interval') not in ('step', 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'):
            raise UserError(f'Forecast field {name!r} requires a valid `interval`.')
    return spec


_schema_adapter = TypeAdapter(dict[str, JsonValue])
_date_adapter = TypeAdapter(date)
