"""CLI argument parser for the PydanticAI AG-UI servers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class Args:
    """Custom namespace for command line arguments."""

    port: int
    reload: bool
    log_level: str


def parse_args() -> Args:
    """Parse command line arguments for the PydanticAI AG-UI servers.

    Returns:
        Args: A dataclass containing the parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='PydanticAI AG-UI Dojo server')
    parser.add_argument(
        '--port',
        '-p',
        type=int,
        default=9000,
        help='Port to run the server on (default: 9000)',
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        default=True,
        help='Enable auto-reload (default: True)',
    )
    parser.add_argument(
        '--no-reload', dest='reload', action='store_false', help='Disable auto-reload'
    )
    parser.add_argument(
        '--log-level',
        choices=['critical', 'error', 'warning', 'info', 'debug', 'trace'],
        default='info',
        help='Log level (default: info)',
    )

    return Args(**vars(parser.parse_args()))
