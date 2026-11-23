"""Main-footer visibility and ordering through the shared settings-menu widgets."""

import json
from functools import partial

from anyio import to_thread

from pydantic_ai_pax.cli.command_context import CommandContext
from pydantic_ai_pax.config import DEFAULT_FOOTER_FIELDS, FOOTER_FIELDS, FooterField
from pydantic_ai_pax.plugins.menus import Action, Dialogs, FieldRow, Option, SettingsMenu

FOOTER_SETTING = 'display.footer_fields'
_RESET = 'reset_layout'


class FooterSource:
    """Keep hidden rows in place while editing; persist just the ordered visible fields."""

    title = 'Main footer layout'

    def __init__(self, context: CommandContext) -> None:
        self.context = context
        visible = context.settings.footer_fields
        self.order: list[FooterField] = [*visible, *(name for name in FOOTER_FIELDS if name not in visible)]

    def rows(self) -> list[FieldRow]:
        return [
            *(
                FieldRow(
                    key=name,
                    label=name,
                    description='Choose Show, Hide, Move earlier or Move later.',
                    default='shown' if name in DEFAULT_FOOTER_FIELDS else 'hidden',
                    button='Configure…',
                )
                for name in self.order
            ),
            FieldRow(key=_RESET, label='Reset layout', description='Restore the shipped layout only.', default=''),
        ]

    def current(self, row: FieldRow) -> str:
        if row.key == _RESET:
            return ''
        return 'shown' if row.key in self.context.settings.footer_fields else 'hidden'

    def problem(self, row: FieldRow, text: str) -> str | None:
        return None if text in ('shown', 'hidden') else 'Choose shown or hidden.'

    def apply(self, row: FieldRow, raw: str) -> str:
        if problem := self.problem(row, raw):
            raise ValueError(problem)
        visible = set(self.context.settings.footer_fields)
        fields = [name for name in self.order if (raw == 'shown' if name == row.key else name in visible)]
        return self.context.set_setting([FOOTER_SETTING, json.dumps(fields)])

    def reset(self, row: FieldRow) -> str:
        if row.key != _RESET:
            return self.apply(row, row.default)
        self.order = [*DEFAULT_FOOTER_FIELDS, *(name for name in FOOTER_FIELDS if name not in DEFAULT_FOOTER_FIELDS)]
        return self.context.reset_setting(FOOTER_SETTING)

    async def configure(self, name: FooterField, dialogs: Dialogs) -> list[str]:
        visible = name in self.context.settings.footer_fields
        action = await dialogs.choose(
            title=f'Footer: {name}',
            options=[
                Option(value='toggle', label='Hide' if visible else 'Show'),
                Option(value='earlier', label='Move earlier'),
                Option(value='later', label='Move later'),
            ],
        )
        if action is None:
            return []
        if action == 'toggle':
            row = next(row for row in self.rows() if row.key == name)
            return [await to_thread.run_sync(self.apply, row, 'hidden' if visible else 'shown')]
        if not visible:
            return ['Show this field before moving it.']
        index = self.order.index(name)
        neighbours = range(index - 1, -1, -1) if action == 'earlier' else range(index + 1, len(self.order))
        target = next((i for i in neighbours if self.order[i] in self.context.settings.footer_fields), None)
        if target is None:
            return []
        self.order[index], self.order[target] = self.order[target], self.order[index]
        fields = [field for field in self.order if field in self.context.settings.footer_fields]
        return [await to_thread.run_sync(self.context.set_setting, [FOOTER_SETTING, json.dumps(fields)])]

    async def reset_layout(self, dialogs: Dialogs) -> list[str]:
        return [await to_thread.run_sync(self.reset, self.rows()[-1])]


def footer_settings_menu(context: CommandContext) -> SettingsMenu:
    """The same layout actions in the terminal and web settings, with a visible layout-only reset."""
    source = FooterSource(context)
    actions: dict[str, Action] = {name: partial(source.configure, name) for name in FOOTER_FIELDS}
    actions[_RESET] = source.reset_layout
    return SettingsMenu(source=source, actions=actions)
