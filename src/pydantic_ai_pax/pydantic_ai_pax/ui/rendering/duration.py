"""Human-readable elapsed time for terminal labels, without changing lifecycle clocks."""

from datetime import datetime


def format_duration(seconds: float, *, started_at: float | None = None) -> str:
    """Show up to three units; with a known start, show its local date after thirty days."""
    elapsed = max(0, int(seconds))
    if elapsed >= 30 * 86400 and started_at is not None:
        return datetime.fromtimestamp(started_at).astimezone().strftime('%b %d')
    days, remainder = divmod(elapsed, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds_left = divmod(remainder, 60)
    units = ((days, 'd'), (hours, 'h'), (minutes, 'm')) if days else ((hours, 'h'), (minutes, 'm'), (seconds_left, 's'))
    return ' '.join(f'{value}{suffix}' for value, suffix in units if value) or '0s'
