#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "httpx",
# ]
# ///
"""MRE for the bug: question_graph.py imports BaseModel from groq instead of pydantic.

This script fetches the current file from the main branch on GitHub and verifies
the bug is present in the released source.
"""
from __future__ import annotations as _annotations

import ast
import sys

import httpx

URL = 'https://raw.githubusercontent.com/pydantic/pydantic-ai/main/examples/pydantic_ai_examples/question_graph.py'


def main() -> None:
    response = httpx.get(URL, follow_redirects=True)
    response.raise_for_status()
    source = response.text
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == 'groq':
                for alias in node.names:
                    if alias.name == 'BaseModel':
                        print('BUG REPRODUCED: BaseModel is imported from groq instead of pydantic')
                        sys.exit(0)

    print('BUG NOT REPRODUCED: BaseModel is not imported from groq')
    sys.exit(1)


if __name__ == '__main__':
    main()
