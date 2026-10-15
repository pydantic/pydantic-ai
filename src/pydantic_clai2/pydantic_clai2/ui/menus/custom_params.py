"""Free-form request body parameters, matching Code Puppy's dotted-key editor."""

import json
from dataclasses import dataclass

from pydantic import JsonValue
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder
from termflow.tui.menu import Menu, MenuResult
from termflow.tui.textinput import TextInput

from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models.custom_params import parse_pair
from pydantic_clai2.ui.menus.field_menu import Runners
from pydantic_clai2.ui.menus.menu_worker import menu_key
from pydantic_clai2.ui.rendering._rendering import markdown_style


@dataclass(frozen=True, kw_only=True)
class DeleteParam:
    """Delete the selected key rather than edit it."""

    key: str


class CustomParamsMenu:
    """A persisted, per-model add/edit/delete menu with no pending changes."""

    def __init__(self, *, store: SettingsStore, model: str) -> None:
        """Keep edits scoped to one saved model."""
        self.store = store
        self.model = model

    def pairs(self) -> dict[str, JsonValue]:
        """Read fresh values after every edit."""
        pairs = self.store.model_settings(self.model).get('custom_params')
        return pairs if isinstance(pairs, dict) else {}

    def build(self) -> Menu:
        """The list and delete shortcut, with an always-available add row."""
        return (
            MenuBuilder(f'Custom Params - {self.model}')
            .style(markdown_style())
            .items(
                [
                    *[MenuItem(f'{key} = {json.dumps(value)}', value=key) for key, value in self.pairs().items()],
                    MenuItem('+ Add param...', value=False),
                ]
            )
            .preview(
                lambda _: (
                    'Dotted keys nest in extra_body. Custom values override built-in request settings.\n'
                    'Values accept JSON or unquoted text. Do not store secrets here.'
                )
            )
            .on_key('d', self.delete_marker)
            .footer_hint('Enter add/edit - d delete - Esc back')
            .key_source(menu_key)
            .build()
        )

    def delete_marker(self, menu: object, item: MenuItem) -> MenuResult | None:
        """Do not delete the add row."""
        if not isinstance(item.value, str):
            return None
        return MenuResult(item=MenuItem('', value=DeleteParam(key=item.value)))

    def editor(self, *, key: str | None) -> TextInput:
        """Edit a pair together so renaming is atomic."""

        def problem(text: str) -> str | None:
            try:
                parse_pair(text=text)
            except ValueError as exc:
                return str(exc)
            return None

        return (
            TextInputBuilder(f'Custom param - {self.model}')
            .style(markdown_style())
            .prompt('key = value: ')
            .initial(f'{key} = {json.dumps(self.pairs()[key])}' if key is not None else '')
            .placeholder('chat_template_kwargs.thinking = medium')
            .validator(problem)
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
            .build()
        )

    def save(self, *, pairs: dict[str, JsonValue]) -> None:
        """Preserve other model settings while replacing custom parameters."""
        saved = self.store.model_settings(self.model)
        if pairs:
            saved['custom_params'] = pairs
        else:
            saved.pop('custom_params', None)
        self.store.save_model_settings(self.model, saved)

    def run(self, *, runners: Runners) -> list[str]:
        """Run on the owning menu worker, saving every successful mutation."""
        messages: list[str] = []
        while True:
            result = runners.run_list(self.build())
            if result.cancelled or result.item is None:
                return messages
            selected = result.item.value
            pairs = self.pairs()
            if isinstance(selected, DeleteParam):
                pairs.pop(selected.key, None)
            else:
                key = selected if isinstance(selected, str) else None
                edited = runners.run_text(self.editor(key=key))
                if edited.cancelled or edited.value is None:
                    continue
                name, value = parse_pair(text=edited.value)
                if key is not None:
                    pairs.pop(key)
                pairs[name] = value
            self.save(pairs=pairs)
            messages.append(f'Saved custom params for {self.model}.')
