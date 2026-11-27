"""`pydantic-clai2` is now `pydantic-ai-pax`: the new commands, and code names that no longer say `clai2`."""

import ast
import io
import tokenize
import tomllib
from pathlib import Path

SOURCES = Path(__file__).parents[2] / 'src'


def test_pax_is_the_command() -> None:
    pax = tomllib.loads((SOURCES / 'pydantic_ai_pax' / 'pyproject.toml').read_text())
    assert pax['project']['scripts'] == {
        'pax': 'pydantic_ai_pax.__main__:main',
        'pydantic-ai-pax': 'pydantic_ai_pax.__main__:main',
    }


def test_code_names_say_pax_not_clai2() -> None:
    """Identifiers and `ContextVar` names use the new name.

    Persisted keys, `CLAI2_*` variables, and the `pydantic_clai2` factory names keep the old one, in strings only.
    It reads source only, so it is not a VCR test.
    """
    package = SOURCES / 'pydantic_ai_pax' / 'pydantic_ai_pax'
    found: list[str] = []
    for path in sorted(package.rglob('*.py')):
        source = path.read_text()
        if 'clai2' not in source.lower():
            continue
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        names = [token.string for token in tokens if token.type == tokenize.NAME and 'clai2' in token.string.lower()]
        context_vars = [
            node.args[0].value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == 'ContextVar'
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and 'clai2' in str(node.args[0].value)
        ]
        found += [f'{path.relative_to(package)}: {name}' for name in [*names, *context_vars]]
    assert found == []
