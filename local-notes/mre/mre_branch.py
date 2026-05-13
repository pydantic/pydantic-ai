#!/usr/bin/env -S uv run --script
# /// script
# dependencies = []
# ///
"""MRE for the fix: question_graph.py imports BaseModel from pydantic, not groq.

This script reads the local (fixed) file and verifies the import is correct.
"""
from __future__ import annotations as _annotations

import ast
import sys
from pathlib import Path

FILE = Path(__file__).parent.parent.parent / 'examples' / 'pydantic_ai_examples' / 'question_graph.py'


def main() -> None:
    source = FILE.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == 'groq':
                for alias in node.names:
                    if alias.name == 'BaseModel':
                        print('BUG STILL PRESENT: BaseModel is imported from groq')
                        sys.exit(1)
            if node.module == 'pydantic':
                for alias in node.names:
                    if alias.name == 'BaseModel':
                        print('FIX VERIFIED: BaseModel is imported from pydantic')
                        sys.exit(0)

    print('FIX NOT VERIFIED: BaseModel import from pydantic not found')
    sys.exit(1)


if __name__ == '__main__':
    main()
