"""Composer bindings expressed as decoded keys, not keyboard or server detection."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

InputAction = Literal['send', 'steer', 'newline']
KeyboardLabels = Literal['unset', 'neutral', 'mac', 'pc']
QuestionSubmit = Literal['ask', 'answer', 'follow-up']
BusyDelivery = Literal['after-turn', 'next-checkpoint']

# These reports are already indistinguishable after the terminal decoder. Changing
# their action separately would promise a distinction the editor cannot observe.
KEY_ALIASES: dict[str, str] = {
    'ctrl-m': 'enter',
    'ctrl-enter': 'enter',
    'ctrl-shift-enter': 'enter',
    'option-enter': 'alt-enter',
    'cmd-enter': 'super-enter',
}
BINDABLE_KEYS: tuple[str, ...] = (
    'enter',
    'alt-enter',
    'shift-enter',
    'super-enter',
    'alt-shift-enter',
    'super-shift-enter',
    'ctrl-s',
    'alt-s',
    'ctrl-n',
    'alt-n',
    'f3',
)


def canonical_key(key: str) -> str:
    """Use the decoder's identity for aliases, rejecting unknown editing shortcuts."""
    key = KEY_ALIASES.get(key.lower(), key.lower())
    if key not in BINDABLE_KEYS:
        raise ValueError(f'Choose a decoded key from: {", ".join(BINDABLE_KEYS)}. Ctrl+J always inserts a newline.')
    return key


class InputBindings(BaseModel):
    """One explicit key per action; Ctrl+J remains a transport-independent newline fallback."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    send: str = 'enter'
    steer: str = 'alt-enter'
    newline: str = 'shift-enter'

    @model_validator(mode='after')
    def distinct_actions(self) -> 'InputBindings':
        """Reject aliases and collisions at the shared settings boundary."""
        keys = (canonical_key(self.send), canonical_key(self.steer), canonical_key(self.newline))
        if len(set(keys)) != len(keys):
            raise ValueError(
                'Input actions collide after decoding. Test a key with /input test; use Ctrl+J for newline.'
            )
        return self

    def action(self, key: str) -> InputAction | None:
        """The effective action for a decoded key, including the fixed newline fallback."""
        if key == 'ctrl-j' or key == canonical_key(self.newline):
            return 'newline'
        if key == canonical_key(self.send):
            return 'send'
        if key == canonical_key(self.steer):
            return 'steer'
        return None

    def label(self, action: InputAction, *, keyboard: KeyboardLabels = 'neutral') -> str:
        """Label keys for the user's keyboard, independent of the remote machine's OS."""
        key = self.send if action == 'send' else self.steer if action == 'steer' else self.newline
        parts = canonical_key(key).split('-')
        labels = {
            'alt': 'Option' if keyboard == 'mac' else 'Alt',
            'super': 'Cmd' if keyboard == 'mac' else 'Super',
            'enter': 'Return' if keyboard == 'mac' else 'Enter',
        }
        return '+'.join(labels.get(part, part.capitalize()) for part in parts)
