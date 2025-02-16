import math

__all__ = (
    'default_render_number',
    'default_render_number_diff',
    'default_render_duration',
    'default_render_duration_diff',
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
    * If the value is a float, include at least one decimal place and at least four significant figures.

    Test cases for numbers:
    print(default_format_number(0))
    print(default_format_number(17348))
    print(default_format_number(17347.0))
    # prints 0; should print 0.0000 (four significant figures)
    print(default_format_number(0.1))
    # prints 0.1; should print 0.1000 (four significant figures)
    print(default_format_number(2.0))
    # prints 2.0; should print 2.000 (four significant figures)
    print(default_format_number(12.0))
    # prints 12.0; should print 12.00 (four significant figures)
    print(default_format_number(2398723.123))
    # prints 2.399e+06; should print 2398723.1 (one decimal place)
    """
    # If it's an int, just return its string representation.
    if isinstance(value, int):
        return str(value)

    abs_val = abs(value)

    # Special case for zero:
    if abs_val == 0:
        return f'{value:.{VALUE_SIG_FIGS}f}'

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

    return f'{value:.{decimals}f}'


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

    Test cases for diffs:
      _default_format_number_diff(3, 3)            is None
      _default_format_number_diff(127.3, 127.3)      is None

      _default_format_number_diff(3, 4)              == '+1'
      _default_format_number_diff(4, 3)              == '-1'

      _default_format_number_diff(1.0, 1.7)          == '+0.7 / +70.0%'
      _default_format_number_diff(2.5, 1.0)          == '-1.5 / -60.0%'

      _default_format_number_diff(10.023, 10.024)    == '+0.001'
      _default_format_number_diff(1.00024, 1.00023)  == '-1e-05'

      _default_format_number_diff(2.0, 25.0)         == '+23.0 / +12.5x'
      _default_format_number_diff(2.0, -25.0)        == '-27.0 / -12.5x'
      _default_format_number_diff(0.02, 25.0)        == '+25.0 / +1250x'
      _default_format_number_diff(0.02, -25.0)       == '-25.0 / -1250x'

      _default_format_number_diff(0.001, 25.0)       == '+25.0'
    """
    # No change.
    if old == new:
        return None

    # If both values are ints, just return the raw difference.
    if isinstance(old, int) and isinstance(new, int):
        diff_int = new - old
        return f'{diff_int:+d}'

    # Compute the raw difference.
    delta = new - old
    diff_str = _format_signed(delta, ABS_SIG_FIGS)

    # If we cannot compute a relative change, return just the diff.
    if old == 0:
        return diff_str

    # For very small base values with huge changes, drop the relative indicator.
    if abs(old) < BASE_THRESHOLD and abs(delta) > MULTIPLIER_DROP_FACTOR * abs(old):
        return diff_str

    # Compute the relative change as a percentage.
    rel_change = (delta / old) * 100
    perc_str = f'{rel_change:+.{PERC_DECIMALS}f}%'
    # If the percentage rounds to 0.0%, return only the absolute difference.
    if perc_str in ('+0.0%', '-0.0%'):
        return diff_str

    # Decide whether to use percentage style or multiplier style.
    if abs(delta) / abs(old) <= 1:
        # Percentage style.
        return f'{diff_str} / {perc_str}'
    else:
        # Multiplier style.
        multiplier = new / old
        if abs(multiplier) < MULTIPLIER_ONE_DECIMAL_THRESHOLD:
            mult_str = f'{multiplier:+.1f}x'
        else:
            mult_str = f'{multiplier:+.0f}x'
        return f'{diff_str} / {mult_str}'


def _format_signed(val: float, sig_figs: int = ABS_SIG_FIGS) -> str:
    """Format a number with a fixed number of significant figures.

    If the result does not use scientific notation and lacks a decimal point,
    force a '.0' suffix. Always include a leading '+' for nonnegative numbers.
    """
    s = format(abs(val), f'.{sig_figs}g')
    if 'e' not in s and '.' not in s:
        s += '.0'
    return f"{'+' if val >= 0 else '-'}{s}"


def default_render_duration(seconds: float) -> str:
    """Format a duration given in seconds to a string.

    If the duration is less than 1 millisecond, show microseconds.
    If it's less than one second, show milliseconds.
    Otherwise, show seconds.
    """
    precision = 1
    if seconds < 1e-3:
        value = seconds * 1_000_000
        unit = 'µs'
        if value >= 1:
            precision = 0
    elif seconds < 1:
        value = seconds * 1_000
        unit = 'ms'
    else:
        value = seconds
        unit = 's'
    return f'{value:.{precision}f}{unit}'


def default_render_duration_diff(old: float, new: float) -> str:
    """Format a duration difference (in seconds) with an explicit sign.

    Uses the same unit as format_duration.
    """
    diff = new - old
    precision = 1
    abs_diff = abs(diff)
    if abs_diff < 1e-3:
        value = diff * 1_000_000
        unit = 'µs'
        if abs(value) >= 1:
            precision = 0
    elif abs_diff < 1:
        value = diff * 1_000
        unit = 'ms'
    else:
        value = diff
        unit = 's'
    return f'{value:+.{precision}f}{unit}'
