"""Pin each `EmbeddingSettings` field's `Supported by:` list to the models that actually read it.

Same contract as `tests/models/test_model_settings_support.py`, which pins the `ModelSettings`
lists: these bullets are the only place a user can learn whether a general setting reaches a given
embedding model, and an unsupported setting is silently dropped rather than rejected, so a stale
list is indistinguishable from a broken provider. `EmbeddingSettings` was not covered by that
module, which is how `dimensions` came to claim flat `Bedrock` support while Titan v1 and Cohere v3
drop it.

Support is derived from the source rather than from a hand-maintained table, so the assertion is
not circular: `_settings_keys_read` walks each embedding module's AST for reads of the settings
key (`settings.get('x')`, `settings['x']`, `'x' in settings`) and the bullets are asserted against
that, in both directions.

Reading the AST rather than probing the wire is what keeps this offline and provider-agnostic — an
embedding provider's SDK is an optional dependency, so a wire probe would skip on any install
missing one, which is every install. The one place a read is not the whole story is Bedrock, where
`prepare_request` reads `dimensions` for every model and then discards it below a version
threshold; `test_bedrock_dimensions_support_matches_caveat` pins that separately by building real
request bodies.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from pydantic_ai.embeddings.settings import EmbeddingSettings

pytestmark = pytest.mark.anyio

EMBEDDINGS_DIR = Path(__file__).parent.parent / 'pydantic_ai_slim' / 'pydantic_ai' / 'embeddings'

#: Bullet label in a `Supported by:` list -> the module implementing it.
PROVIDER_MODULES = {
    'OpenAI': 'openai',
    'Cohere': 'cohere',
    'Google': 'google',
    'Sentence Transformers': 'sentence_transformers',
    'Bedrock': 'bedrock',
    'VoyageAI': 'voyageai',
}


def _field_docstrings() -> dict[str, str]:
    """The docstring following each `EmbeddingSettings` field, keyed by field name."""
    source = (EMBEDDINGS_DIR / 'settings.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    class_def = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'EmbeddingSettings')

    docstrings: dict[str, str] = {}
    pending: str | None = None
    for node in class_def.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            pending = node.target.id
        elif (
            pending is not None
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            docstrings[pending] = node.value.value
            pending = None
    return docstrings


def _parse_bullets(docstring: str) -> set[str]:
    """The provider labels bulleted under `Supported by:`, with parenthetical caveats stripped."""
    _, _, block = docstring.partition('Supported by:')
    labels: set[str] = set()
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith('* '):
            labels.add(stripped[2:].split(' (', 1)[0].strip())
    return labels


class _SettingsKeyVisitor(ast.NodeVisitor):
    """Collect every settings key the module reads."""

    def __init__(self) -> None:
        self.keys: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == 'get' and node.args:
            if isinstance(arg := node.args[0], ast.Constant) and isinstance(arg.value, str):
                self.keys.add(arg.value)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(index := node.slice, ast.Constant) and isinstance(index.value, str):
            self.keys.add(index.value)
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # Only the leading operator matters: `'x' in settings` is a single comparison, and a chained
        # one (`'x' in a in b`) isn't a settings read.
        if (
            isinstance(node.ops[0], ast.In)
            and isinstance(left := node.left, ast.Constant)
            and isinstance(left.value, str)
        ):
            self.keys.add(left.value)
        self.generic_visit(node)


def _settings_keys_read(module: str) -> set[str]:
    visitor = _SettingsKeyVisitor()
    visitor.visit(ast.parse((EMBEDDINGS_DIR / f'{module}.py').read_text(encoding='utf-8')))
    return visitor.keys


GENERIC_FIELDS = sorted(EmbeddingSettings.__annotations__)


@pytest.mark.parametrize('field_name', GENERIC_FIELDS)
def test_supported_by_matches_the_models_that_read_it(field_name: str) -> None:
    """Every `Supported by:` bullet reads the setting, and every model that reads it is bulleted."""
    documented = _parse_bullets(_field_docstrings()[field_name])
    assert documented, f'`{field_name}` has no `Supported by:` list'

    actual = {label for label, module in PROVIDER_MODULES.items() if field_name in _settings_keys_read(module)}

    assert documented == actual, (
        f'`EmbeddingSettings.{field_name}` documents {sorted(documented)} '
        f'but {sorted(actual)} read it. Update the docstring or the implementation.'
    )


def test_every_documented_provider_has_a_module() -> None:
    """A bullet naming a provider this module can't resolve would silently pass the check above."""
    documented = {label for doc in _field_docstrings().values() for label in _parse_bullets(doc)}
    assert documented <= set(PROVIDER_MODULES), (
        f'unmapped provider labels: {sorted(documented - set(PROVIDER_MODULES))}'
    )


BEDROCK_DIMENSION_CASES = [
    ('amazon.titan-embed-text-v1', False),
    ('amazon.titan-embed-text-v2:0', True),
    ('cohere.embed-english-v3', False),
    ('cohere.embed-v4:0', True),
    ('amazon.nova-2-embed:0', True),
]


@pytest.mark.parametrize(('model_name', 'forwards'), BEDROCK_DIMENSION_CASES)
def test_bedrock_dimensions_support_matches_caveat(model_name: str, forwards: bool) -> None:
    """Bedrock forwards `dimensions` only above each family's version threshold.

    The AST check above can only see that `bedrock.py` reads `dimensions`; the version gate that
    discards it lives below the read, so the caveat on the `dimensions` bullet is pinned here by
    building the real request body instead.
    """
    pytest.importorskip('boto3')
    from pydantic_ai.embeddings.bedrock import _get_handler_for_model  # pyright: ignore[reportPrivateUsage]

    body = _get_handler_for_model(model_name).prepare_request(['hello'], 'document', {'dimensions': 256})

    assert (256 in _values(body)) is forwards


def _values(value: object) -> list[object]:
    """Every scalar reachable in a request body, so the assertion doesn't depend on the field name.

    Each Bedrock family spells the parameter differently (`dimensions`, `output_dimension`,
    `embeddingDimension`) and nests it at a different depth.
    """
    if isinstance(value, dict):
        return [scalar for item in value.values() for scalar in _values(item)]  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
    if isinstance(value, list):
        return [scalar for item in value for scalar in _values(item)]  # pyright: ignore[reportUnknownVariableType,reportUnknownArgumentType]
    return [value]


def test_caveat_is_documented_for_partially_supported_providers() -> None:
    """`dimensions` must keep naming which Bedrock families honor it, not just `Bedrock`."""
    dimensions_doc = _field_docstrings()['dimensions']
    bedrock_bullet = next(line.strip() for line in dimensions_doc.splitlines() if line.strip().startswith('* Bedrock'))
    assert re.search(r'^\* Bedrock \(.+\)$', bedrock_bullet), (
        f'`dimensions` documents flat Bedrock support, but some Bedrock models drop it: {bedrock_bullet!r}'
    )
