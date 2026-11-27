"""Explicit key calibration on the composer's existing input reader."""

from dataclasses import dataclass

from pydantic_ai_pax.config.input_bindings import KeyboardLabels


@dataclass(kw_only=True)
class KeyTest:
    """Observe decoded keys without running any composer action or recording input content."""

    report: str = 'Press the key to test. Ctrl+G closes; no prompts or interrupts run here.'

    def observe(self, key: str, *, keyboard: KeyboardLabels, action: str, steer: str, newline: str) -> None:
        """Show key identity and its effective action, never a draft or paste payload.

        `steer` and `newline` name the keys the composer's hints name for those actions.
        """
        identity = key if len(key) > 1 else 'text key'
        self.report = f'Decoded: {identity}; action: {action}.'
        if key == 'enter':
            enter = 'Return' if keyboard == 'mac' else 'Enter'
            self.report += f' {enter}/Ctrl+M share this action; a lost modifier cannot be recovered.'
        if key == 'paste':
            self.report = 'Decoded: bracketed paste (contents hidden); action: insert text, never submit.'
        self.report += f' Newline fallback: {newline}. Steer: {steer}. Ctrl+G closes.'
