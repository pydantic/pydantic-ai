"""Claude's `LS` tool — list a workspace directory's entries."""

from .shared import attach_context, clip, resolve


def list_dir(path: str = '.') -> str:
    """List a workspace directory's entries (directories marked with `/`)."""
    try:
        p = resolve(path)
        listing = '\n'.join(sorted(e.name + ('/' if e.is_dir() else '') for e in p.iterdir())) or '(empty)'
        return clip(attach_context(path) + listing)
    except OSError as exc:
        return f'error: {exc}'
