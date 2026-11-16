"""The `/effort` shortcut over the model settings editor's validated controls."""

import json
from typing import TYPE_CHECKING

from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.models.custom_params import custom_effort_override

if TYPE_CHECKING:
    from pydantic_clai2.ui.menus.field_menu import FieldRow
    from pydantic_clai2.ui.menus.model_menu import ModelSettingsSource


def effort_command(context: CommandContext, args: list[str], *, model: str) -> str:
    """View, set, or reset the active model's effort; custom body overrides take precedence."""
    if len(args) > 1:
        raise ValueError('Usage: /effort [VALUE|reset]')
    source, row = _effort_settings(context, model=model)
    if row is None:
        return f'No reasoning effort control for {model}. Use /model settings for available controls.'
    custom = context.store.model_settings(model).get('custom_params')
    override = (
        custom_effort_override(custom, settings_as=context.settings_model(model), key=row.key)
        if isinstance(custom, dict)
        else None
    )
    warning = (
        f'Custom {override[0]}={json.dumps(override[1])} overrides configured effort for {model}. '
        'Remove it with /model settings before using /effort to change effort.'
        if override is not None
        else ''
    )
    if args == ['reset']:
        result = source.reset(row)
        return f'{result}\n{warning}' if warning else result
    if warning:
        return warning
    if args:
        if args[0] not in row.choices:
            raise ValueError(f'Choose {", ".join(row.choices)}, or reset.')
        return source.apply(row, args[0])
    return (
        f'Configured reasoning effort for {model}: {source.current(row)} ({row.key}).\n'
        f'Supported values: {", ".join(row.choices)}. Use /effort VALUE or /effort reset.\n'
        'Thinking controls and custom parameters in /model settings still apply.'
    )


def effort_completions(context: CommandContext, args: list[str], *, model: str) -> tuple[str, ...]:
    """Complete the same native effort choices as the model settings editor."""
    if len(args) > 1:
        return ()
    _, row = _effort_settings(context, model=model)
    return (*row.choices, 'reset') if row is not None else ()


def _effort_settings(context: CommandContext, *, model: str) -> tuple['ModelSettingsSource', 'FieldRow | None']:
    # Model settings and menus load provider SDKs; keep them off the startup path.
    from pydantic_clai2.models.model_options import model_options
    from pydantic_clai2.ui.menus.model_menu import ModelSettingsSource

    settings_as = context.settings_model(model)
    source = ModelSettingsSource(context.store, model, settings_as=settings_as)
    options = model_options(model=settings_as)
    rows = {row.key: row for row in source.rows()}
    # GLM's native effort wins over a control offered by its OpenAI-compatible transport.
    for key in ('glm_reasoning_effort', 'anthropic_effort', 'openai_reasoning_effort'):
        if key in options:
            return source, rows[key]
    return source, None
