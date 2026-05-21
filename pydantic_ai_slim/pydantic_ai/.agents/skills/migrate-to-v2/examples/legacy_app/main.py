"""Entry point that triggers every module's deprecations at import time."""
from __future__ import annotations

import os

os.environ.setdefault('OPENAI_API_KEY', 'sk-dummy')

from . import agents, messages_legacy, evals_setup


def run() -> None:
    agents.build()
    messages_legacy.touch_legacy_fields()
    evals_setup.build()


if __name__ == '__main__':
    run()
