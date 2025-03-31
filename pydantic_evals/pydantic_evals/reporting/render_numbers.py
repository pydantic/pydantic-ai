from __future__ import annotations

import math

__all__ = (
    'default_render_number',
    'default_render_number_diff',
    'default_render_duration',
    'default_render_duration_diff',
    'default_render_percentage',
)

# Configuration constants for number formatting
VALUE_SIG_FIGS = 3  # Significant figures for the default number formatting.

# Configuration constants for diff formatting
ABS_SIG_FIGS = 3  # Significant figures for the absolute difference.
PERC_DECIMALS = 1  # Decimal places for percentage formatting.
MULTIPLIER_ONE_DECIMAL_THRESHOLD = 100  # If |multiplier| is below this, use one decimal; otherwise, use none.
BASE_THRESHOLD = 1e-2  # If |old| is below this and delta is > MULTIPLIER_DROP_FACTOR * |old|, drop relative change.
MULTIPLIER_DROP_FACTOR = 10  # Factor used with BASE_THRESHOLD to drop the multiplier.


def default_render_number(value: float | int) -> str:
    """The default logic for formatting numerical values in an Evaluation report.

    * If the value is an integer, format it as an integer.
    * If the value is a float, include at least one decimal place and at least 3 significant figures.
    """
    # If it's an int, just return its string representation.
    if isinstance(value, int):
        return str(value)

    abs_val = abs(value)

    # Special case for zero:
    if abs_val == 0:
        return f'{value:,.{VALUE_SIG_FIGS}f}'

    if abs_val >= 1:
        # Count the digits in the integer part.
        digits = math.floor(math.log10(abs_val)) + 1
        # Number of decimals: at least one, and enough to reach 4 significant figures.
        decimals = max(1, VALUE_SIG_FIGS - digits)
    else:
        # For numbers between 0 and 1, determine the exponent.
        # For example: 0.1 -> log10(0.1) = -1, so we want -(-1) + 3 = 4 decimals.
        exponent = math.floor(math.log10(abs_val))
        decimals = -exponent + VALUE_SIG_FIGS - 1  # because the first nonzero digit is in the 10^exponent place.

    return f'{value:,.{decimals}f}'


def default_render_percentage(value: float) -> str:
    """The default logic for formatting percentages in an Evaluation report.

    * Include at least one decimal place and at least 3 significant figures.
    """
    return f'{value:,.{VALUE_SIG_FIGS - 2}%}'


def default_render_number_diff(old: float | int, new: float | int) -> str | None:
    """Return a string representing the difference between old and new values.

    Rules:
      - If the two values are equal, return None.
      - For integers, return the raw difference (with a leading sign), e.g.:
            _default_format_number_diff(3, 4) -> '+1'
      - For floats (or a mix of float and int):
          * Compute the raw delta = new - old and format it with ABS_SIG_FIGS significant figures.
          * If `old` is nonzero, compute a relative change:
              - If |delta|/|old| ≤ 1, render the relative change as a percentage with
                PERC_DECIMALS decimal places, e.g. '+0.7 / +70.0%'.
              - If |delta|/|old| > 1, render a multiplier (new/old). Use one decimal place
                if the absolute multiplier is less than MULTIPLIER_ONE_DECIMAL_THRESHOLD,
                otherwise no decimals.
          * However, if the percentage rounds to 0.0% (e.g. '+0.0%'), return only the absolute diff.
          * Also, if |old| is below BASE_THRESHOLD and |delta| exceeds MULTIPLIER_DROP_FACTOR×|old|,
            drop the relative change indicator.
    """
    if old == new:
        return None

    if isinstance(old, int) and isinstance(new, int):
        diff_int = new - old
        return f'{diff_int:+d}'

    delta = new - old
    abs_diff_str = _render_signed(delta, ABS_SIG_FIGS)
    rel_diff_str = _render_relative(new, old, BASE_THRESHOLD)
    if rel_diff_str is None:
        return abs_diff_str
    else:
        return f'{abs_diff_str} / {rel_diff_str}'


def default_render_duration(seconds: float) -> str:
    """Format a duration given in seconds to a string.

    If the duration is less than 1 millisecond, show microseconds.
    If it's less than one second, show milliseconds.
    Otherwise, show seconds.
    """
    return _render_duration(seconds, False)


def default_render_duration_diff(old: float, new: float) -> str | None:
    """Format a duration difference (in seconds) with an explicit sign."""
    if old == new:
        return None

    abs_diff_str = _render_duration(new - old, True)
    rel_diff_str = _render_relative(new, old, BASE_THRESHOLD)
    if rel_diff_str is None:
        return abs_diff_str
    else:
        return f'{abs_diff_str} / {rel_diff_str}'


def _render_signed(val: float, sig_figs: int) -> str:
    """Format a number with a fixed number of significant figures.

    If the result does not use scientific notation and lacks a decimal point,
    force a '.0' suffix. Always include a leading '+' for nonnegative numbers.
    """
    s = format(abs(val), f',.{sig_figs}g')
    if 'e' not in s and '.' not in s:
        s += '.0'
    return f'{"+" if val >= 0 else "-"}{s}'


def _render_relative(new: float, base: float, small_base_threshold: float) -> str | None:
    # If we cannot compute a relative change, return just the diff.
    if base == 0:
        return None

    delta = new - base

    # For very small base values with huge changes, drop the relative indicator.
    if abs(base) < small_base_threshold and abs(delta) > MULTIPLIER_DROP_FACTOR * abs(base):
        return None

    # Compute the relative change as a percentage.
    rel_change = (delta / base) * 100
    perc_str = f'{rel_change:+.{PERC_DECIMALS}f}%'
    # If the percentage rounds to 0.0%, return only the absolute difference.
    if perc_str in ('+0.0%', '-0.0%'):
        return None

    # Decide whether to use percentage style or multiplier style.
    if abs(delta) / abs(base) <= 1:
        # Percentage style.
        return perc_str
    else:
        # Multiplier style.
        multiplier = new / base
        if abs(multiplier) < MULTIPLIER_ONE_DECIMAL_THRESHOLD:
            mult_str = f'{multiplier:,.1f}x'
        else:
            mult_str = f'{multiplier:,.0f}x'
        return mult_str


def _render_duration(seconds: float, force_signed: bool) -> str:
    """Format a duration given in seconds to a string.

    If the duration is less than 1 millisecond, show microseconds.
    If it's less than one second, show milliseconds.
    Otherwise, show seconds.
    """
    if seconds == 0:
        return '0s'
    precision = 1
    if (abs_seconds := abs(seconds)) < 1e-3:
        value = seconds * 1_000_000
        unit = 'µs'
        if abs(value) >= 1:
            precision = 0
    elif abs_seconds < 1:
        value = seconds * 1_000
        unit = 'ms'
    else:
        value = seconds
        unit = 's'

    if force_signed:
        return f'{value:+,.{precision}f}{unit}'
    else:
        return f'{value:,.{precision}f}{unit}'
