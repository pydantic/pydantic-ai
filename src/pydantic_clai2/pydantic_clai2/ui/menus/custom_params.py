"""Free-form request body parameters, matching Code Puppy's dotted-key editor, asked through `Dialogs`."""

import json

from anyio import to_thread
from pydantic import JsonValue

from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.custom_params import parse_pair
from pydantic_clai2.plugins.menus import Dialogs, Option

_ADD = ''
"""The "add" choice: `parse_pair` rejects an empty key, so no parameter is called this."""
_HELP = (
    'Dotted keys nest in extra_body. Custom values override built-in request settings.\n\n'
    'Values accept JSON or unquoted text. Do not store secrets here.'
)


async def edit_custom_params(dialogs: Dialogs, *, store: SettingsStore, model: str) -> list[str]:
    """Add, change, or remove `model`'s custom parameters one at a time, saving each, until the user backs out.

    A parameter is edited as `key = value`, so renaming it is one edit; empty text removes it.
    """
    messages: list[str] = []
    while True:
        pairs = await to_thread.run_sync(_pairs, store, model)
        picked = await dialogs.choose(
            title=f'Custom Params - {model}',
            options=[
                *(Option(value=key, label=f'{key} = {json.dumps(value)}') for key, value in pairs.items()),
                Option(value=_ADD, label='+ Add param...'),
            ],
            description=_HELP,
        )
        if picked is None:
            return messages
        key = picked or None
        text = await dialogs.text(
            title=f'Custom param - {model}',
            label='key = value',
            placeholder='chat_template_kwargs.thinking = medium',
            initial='' if key is None else f'{key} = {json.dumps(pairs[key])}',
            check=_problem,
            description='' if key is None else 'Empty text removes it.',
        )
        if text is None or (not text and key is None):
            continue
        if key is not None:
            pairs.pop(key)
        if text:
            name, value = parse_pair(text=text)
            pairs[name] = value
        await to_thread.run_sync(_save, store, model, pairs)
        messages.append(f'Saved custom params for {model}.')


def _pairs(store: SettingsStore, model: str) -> dict[str, JsonValue]:
    pairs = store.model_settings(model).get('custom_params')
    return pairs if isinstance(pairs, dict) else {}


def _save(store: SettingsStore, model: str, pairs: dict[str, JsonValue]) -> None:
    """Replace the custom parameters, keeping the model's other settings."""
    saved = store.model_settings(model)
    if pairs:
        saved['custom_params'] = pairs
    else:
        saved.pop('custom_params', None)
    store.save_model_settings(model, saved)


def _problem(text: str) -> str | None:
    try:
        parse_pair(text=text)
    except ValueError as exc:
        return str(exc)
    return None
