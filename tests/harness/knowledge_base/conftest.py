"""Shared collection rules for the knowledge base tests."""

from __future__ import annotations

import importlib.util

# The Bedrock and OpenAI backends need `boto3` and `openai`, which the slim CI lanes don't install.
# Conditional expressions rather than `if` statements: branch coverage traces statement arcs, and no
# single environment can take both arms of an install-dependent branch.
collect_ignore = [
    *(['test_bedrock.py'] if importlib.util.find_spec('boto3') is None else []),
    *(['test_openai.py'] if importlib.util.find_spec('openai') is None else []),
]
