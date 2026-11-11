"""Keyboard setup and question routing through the shared settings editor."""

import json
from collections.abc import Callable, Sequence
from functools import partial

from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config.input_bindings import BINDABLE_KEYS, InputAction, InputBindings
from pydantic_clai2.plugins.menus import CloseMenu, Dialogs, FieldRow, Option, SettingsMenu
from pydantic_clai2.ui.menus.set_menu import SettingsSource

INPUT_SETTINGS = ('input.keyboard_labels', 'input.question_submit')
_ACTIONS: tuple[InputAction, ...] = ('send', 'steer', 'newline')


class InputSource(SettingsSource):
    """Existing scalar settings plus one row per binding and an explicit test action."""

    def __init__(self, context: CommandContext) -> None:
        super().__init__(context, keys=INPUT_SETTINGS, title='Input setup (labels do not detect terminal keys)')
        self.context = context

    def rows(self) -> list[FieldRow]:
        return [
            *super().rows(),
            *(
                FieldRow(
                    key=action,
                    label=action.capitalize(),
                    description='Choose a decoded key; test it on this client.',
                    default='',
                )
                for action in _ACTIONS
            ),
            FieldRow(
                key='test',
                label='Test a key',
                description='Safe capture. No sends or interrupts. Ctrl+G closes.',
                default='',
            ),
        ]

    def current(self, row: FieldRow) -> str:
        if row.key in _ACTIONS:
            return str(self.context.settings.input_bindings.model_dump()[row.key])
        return '' if row.key == 'test' else super().current(row)

    def reset(self, row: FieldRow) -> str:
        if row.key in _ACTIONS:
            values = self.context.settings.input_bindings.model_dump()
            values[row.key] = InputBindings().model_dump()[row.key]
            return self.context.set_setting(['input.bindings', json.dumps(values)])
        return 'No captured key is saved.' if row.key == 'test' else super().reset(row)


async def choose_binding(dialogs: Dialogs, *, context: CommandContext, action: InputAction) -> Sequence[str]:
    """Save through the same validation as `/set`, including decoded-key collisions."""
    chosen = await dialogs.choose(
        title=f'Key for {action}',
        options=tuple(Option(value=key, label=key) for key in BINDABLE_KEYS),
        current=str(context.settings.input_bindings.model_dump()[action]),
        description='Test your client with /input test. Mac labels cannot recover modifiers lost over SSH. Ctrl+J always inserts newline.',
    )
    if chosen is None:
        return ()
    values = context.settings.input_bindings.model_dump()
    values[action] = chosen
    try:
        return (context.set_setting(['input.bindings', json.dumps(values)]),)
    except ValueError as exc:
        return (str(exc),)


async def test_key(dialogs: Dialogs, *, start: Callable[[], None]) -> Sequence[str]:
    """Return input ownership to the composer before capturing keys."""
    start()
    return CloseMenu(['Test a key now. Ctrl+G closes; Ctrl+J is the newline fallback.'])


def input_menu(context: CommandContext, *, start_test: Callable[[], None]) -> SettingsMenu:
    """Expose setup, remapping, calibration and per-row reset in one reachable menu."""
    return SettingsMenu(
        source=InputSource(context),
        actions={
            **{action: partial(choose_binding, context=context, action=action) for action in _ACTIONS},
            'test': partial(test_key, start=start_test),
        },
    )
