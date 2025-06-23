"""CLI argument parser for the PydanticAI AG-UI servers."""

import argparse
from typing import Any

from uvicorn.config import LOGGING_CONFIG

from pydantic_ai.models import dataclass


@dataclass
class Args:
    """Custom namespace for command line arguments."""

    port: int
    reload: bool
    log_level: str
    loggers: list[str]

    def log_config(self) -> dict[str, Any]:
        """Return the logging configuration based on the log level."""
        log_config: dict[str, Any] = LOGGING_CONFIG.copy()
        for logger in self.loggers:
            log_config['loggers'][logger] = {
                'handlers': ['default'],
                'level': self.log_level.upper(),
                'propagate': False,
            }

        return log_config


def parse_args() -> Args:
    """Parse command line arguments for the PydanticAI AG-UI servers.

    Returns:
        Args: A dataclass containing the parsed command line arguments.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description='PydanticAI AG-UI Dojo server'
    )
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
        help='Adapter log level (default: info)',
    )
    parser.add_argument(
        '--loggers',
        nargs='*',
        default=[
            'pydantic_ai.ag_ui.adapter',
        ],
        help='Logger names to configure (default: adapter and model loggers)',
    )

    args: argparse.Namespace = parser.parse_args()
    return Args(**vars(args))
