"""This means `python -m clai` should run the CLI."""

from pydantic_ai import _cli

if __name__ == '__main__':
    _cli.cli_exit('clai')
