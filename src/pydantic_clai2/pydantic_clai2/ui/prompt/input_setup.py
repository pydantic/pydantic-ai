"""Explicit key calibration on the composer's existing input reader."""

from dataclasses import dataclass

from pydantic_clai2.config.input_bindings import InputBindings, KeyboardLabels


@dataclass(kw_only=True)
class KeyTest:
    """Observe decoded keys without running any composer action or recording input content."""

    report: str = 'Press the key to test. Ctrl+G closes; no prompts or interrupts run here.'

    def observe(self, key: str, *, bindings: InputBindings, keyboard: KeyboardLabels) -> None:
        """Show key identity and its effective action, never a draft or paste payload."""
        action = bindings.action(key)
        identity = key if len(key) > 1 else 'text key'
        self.report = f'Decoded: {identity}; action: {action or "edit"}.'
        if key == 'enter':
            self.report += ' Enter/Ctrl+M/Ctrl+Enter share this action; a lost modifier cannot be recovered.'
        if key == 'paste':
            self.report = 'Decoded: bracketed paste (contents hidden); action: insert text, never submit.'
        self.report += f' Newline fallback: Ctrl+J. Steer: {bindings.label("steer", keyboard=keyboard)}. Ctrl+G closes.'
