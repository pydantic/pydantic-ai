from importlib.metadata import version as _metadata_version

from pydantic_ai import _cli

__all__ = '__version__', 'cli'
__version__ = _metadata_version('clai')


def cli():
    """Run the clai CLI and exit."""
    _cli.cli_exit('clai')
