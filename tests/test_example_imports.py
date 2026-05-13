from __future__ import annotations as _annotations

import ast
from pathlib import Path

import pytest


def test_question_graph_imports_base_model_from_pydantic() -> None:
    """Ensure BaseModel is imported from pydantic, not groq, in question_graph.py."""
    path = Path(__file__).parent.parent / 'examples' / 'pydantic_ai_examples' / 'question_graph.py'
    source = path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == 'groq':
                for alias in node.names:
                    assert alias.name != 'BaseModel', 'BaseModel should not be imported from groq'
            if node.module == 'pydantic':
                for alias in node.names:
                    if alias.name == 'BaseModel':
                        return

    pytest.fail('BaseModel should be imported from pydantic')
