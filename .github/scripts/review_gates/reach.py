"""Who runs a file's code: the reach map the `unaffected-users` gate reads.

Built from the default branch's own checkout, never from the pull request: a model module reaches the
providers `infer_model` routes to its classes, and to modules that build on them; a profile module
reaches the providers that import it.
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from pathlib import Path

_PACKAGE = 'pydantic_ai_slim/pydantic_ai'
CORE = {
    f'{_PACKAGE}/_agent_graph.py': 'every agent run',
    f'{_PACKAGE}/agent/': 'every agent run',
    f'{_PACKAGE}/messages.py': 'every agent run, on every model',
    f'{_PACKAGE}/_output.py': 'every agent with an output type',
    f'{_PACKAGE}/tools.py': 'every agent with tools',
    f'{_PACKAGE}/models/__init__.py': 'every model and provider',
    f'{_PACKAGE}/models/decision.py': 'every decision model backend: TypeSafe, System One and OpenAI Decisions',
    'pydantic_evals/pydantic_evals/evaluators/': 'every `pydantic_evals` evaluator, including every LLM judge',
}
_RELATIVE_IMPORT = re.compile(r'^from \.(\w+) import', re.MULTILINE)


def reach_map(root: Path) -> dict[str, str]:
    """Shared file or directory (ending in `/`) to who runs its code, for the checkout at `root`."""
    reach = dict(CORE)
    for module, kinds in _model_module_kinds(root).items():
        if len(kinds) > 1:
            reach[f'{_PACKAGE}/models/{module}.py'] = f'the providers {", ".join(f"`{k}`" for k in sorted(kinds))}'
    for profile, providers in _profile_importers(root).items():
        if len(providers) > 1:
            reach[f'{_PACKAGE}/profiles/{profile}.py'] = (
                f'the providers {", ".join(f"`{p}`" for p in sorted(providers))}'
            )
    return reach


def reach_of(path: str, reach: dict[str, str]) -> str | None:
    """Who runs `path`'s code, or `None` when only its own users do."""
    return reach.get(path) or next(
        (who for key, who in reach.items() if key.endswith('/') and path.startswith(key)), None
    )


def _model_module_kinds(root: Path) -> dict[str, set[str]]:
    """Each `models/` module to the model kinds `infer_model` routes to it or to a module importing from it."""
    models = root / _PACKAGE / 'models'
    tree = ast.parse((models / '__init__.py').read_text(encoding='utf-8'))
    aliases = {
        # The first argument is the alias's own name.
        target.id: {
            c.value
            for arg in node.value.args[1:]
            for c in ast.walk(arg)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        }
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == 'TypeAliasType'
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    infer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'infer_model')
    kinds: dict[str, set[str]] = defaultdict(set)
    for branch in ast.walk(infer):
        if not (
            isinstance(branch, ast.If)
            and isinstance(branch.test, ast.Compare)
            and isinstance(branch.test.left, ast.Name)
            and branch.test.left.id == 'model_kind'
        ):
            continue
        routed: set[str] = set()
        for value in ast.walk(branch.test.comparators[0]):
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                routed.add(value.value)
            elif isinstance(value, ast.Name) and value.id in aliases:
                routed |= aliases[value.id]
        for statement in branch.body:
            if isinstance(statement, ast.ImportFrom) and statement.level == 1 and statement.module:
                kinds[statement.module] |= routed
    # A module whose classes build on another's, such as `openrouter` on `openai`, runs the other's code too.
    for path in models.glob('*.py'):
        for imported in set(_RELATIVE_IMPORT.findall(path.read_text(encoding='utf-8'))) - {path.stem}:
            if imported in kinds and path.stem in kinds:
                kinds[imported] |= kinds[path.stem]
    return kinds


def _profile_importers(root: Path) -> dict[str, set[str]]:
    providers = root / _PACKAGE / 'providers'
    importers: dict[str, set[str]] = defaultdict(set)
    for path in providers.glob('*.py'):
        for profile in re.findall(
            r'from (?:\.\.|pydantic_ai\.)profiles\.(\w+) import', path.read_text(encoding='utf-8')
        ):
            importers[profile].add(path.stem)
    return importers
