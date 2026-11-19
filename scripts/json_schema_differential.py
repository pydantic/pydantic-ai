"""Walk every `JsonSchemaTransformer` subclass over the JSON schema conformance corpus on two revisions, and compare.

Each step imports whichever `pydantic_ai` is on `sys.path`, so run it with `PYTHONPATH=<checkout>/pydantic_ai_slim`
pointing at the revision it reads, and with `PYTHONHASHSEED=0`: the kept recursive `$defs` follow set order
(https://github.com/pydantic/pydantic-ai/issues/13723), so two processes can otherwise order them differently.

1. `corpus TREE OUT` builds the inputs once, from a checkout whose `tests/test_json_schema_conformance.py` holds the
   corpus and whose `tests/json_schema_test_suite/` holds the pinned JSON-Schema-Test-Suite. Each case keeps its
   schema and, when it can be built, the schema with every `$defs` `$ref` under a keyword the walk never enters
   written out in place: the expected output of a fix that inlines those `$ref`s as written is the base revision's
   walk of that form.
2. `dump CORPUS OUT` walks every case, and its written-out form, with every transformer and `strict` unset, `True`
   and `False`, recording each output's `repr`, `is_strict_compatible` and the `$ref`s it leaves pointing at nothing,
   or the exception the walk raised.
3. `compare BASE HEAD` fails when HEAD changes an output BASE leaves no `$ref` dangling in, raises where BASE
   doesn't, or changes `is_strict_compatible` on an unchanged output. It checks each changed output against BASE's
   walk of the written-out form, and lists the cases HEAD still leaves dangling.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

NEVER_WALKED = frozenset(
    {
        'not',
        'if',
        'then',
        'else',
        'dependentSchemas',
        'dependencies',
        'unevaluatedProperties',
        'unevaluatedItems',
        'additionalItems',
        'contentSchema',
        'definitions',
        '$defs',
    }
)
"""Keywords no `JsonSchemaTransformer` walk enters on `main`, except the root `$defs`, which inlining drops."""

STRICT_VARIANTS: dict[str, dict[str, bool]] = {'unset': {}, 'true': {'strict': True}, 'false': {'strict': False}}

Shape = Literal['schema', 'schema_map', 'schema_list', 'schema_or_list']
Record = dict[str, object]


class _Cycle(Exception):
    """A `$ref` written out in place leads back to itself, so the schema has no written-out form."""


def _map_members(shape: Shape, value: object, fn: Callable[[object], object]) -> object:
    """`value` with `fn` applied to each member that is a schema, given its keyword's shape."""
    if shape == 'schema_map':
        if not isinstance(value, dict):
            return value
        return {key: fn(member) if isinstance(member, dict | bool) else member for key, member in value.items()}
    if shape != 'schema' and isinstance(value, list):
        return [fn(member) if isinstance(member, dict | bool) else member for member in value]
    if shape == 'schema_list' or not isinstance(value, dict | bool):
        return value
    return fn(value)


def _defs_target(root: Mapping[str, object], ref: str) -> tuple[bool, object]:
    """Whether `ref` points into the root `$defs`, and at what: a whole definition name, or a pointer at an object."""
    if not ref.startswith('#/$defs/'):
        return False, None
    defs = root.get('$defs')
    name = ref.removeprefix('#/$defs/').replace('~1', '/').replace('~0', '~')
    if isinstance(defs, dict) and name in defs:
        return True, defs[name]
    node: object = root
    for token in ref.split('/')[1:]:
        token = token.replace('~1', '/').replace('~0', '~')
        if isinstance(node, dict):
            node = node.get(token)
        elif isinstance(node, list) and token in map(str, range(len(node))):
            node = node[int(token)]
        else:
            return False, None
    return isinstance(node, dict), node


def _merged_over(target: object, node: Mapping[str, object]) -> dict[str, object]:
    """The definition `target` with the other keywords of the `$ref` node merged over it, as inlining merges them."""
    siblings = {key: value for key, value in node.items() if key != '$ref'}
    if target is False:
        return {**siblings, 'not': {}}
    if isinstance(target, dict):
        return {**deepcopy(target), **siblings}
    return siblings


def _written_out(
    node: object, root: Mapping[str, object], keywords: Mapping[str, Shape], under: bool, stack: tuple[str, ...]
) -> object:
    """`node` with each `$defs` `$ref` below a `NEVER_WALKED` keyword replaced by its definition as written."""
    if not isinstance(node, dict):
        return node
    ref = node.get('$ref')
    if under and isinstance(ref, str):
        found, target = _defs_target(root, ref)
        if found:
            if ref in stack:
                raise _Cycle
            return _written_out(_merged_over(target, node), root, keywords, under, (*stack, ref))
    out = dict(node)
    for key, shape in keywords.items():
        if key in node:
            child_under = under or (key in NEVER_WALKED and not (node is root and key == '$defs'))
            out[key] = _map_members(
                shape, node[key], lambda member: _written_out(member, root, keywords, child_under, stack)
            )
    return out


def _with_written_out(schema: dict[str, object], keywords: Mapping[str, Shape]) -> dict[str, object] | None:
    """`schema` with its `$ref`s below `NEVER_WALKED` keywords written out, or `None` if it has none or they cycle."""
    try:
        written = _written_out(schema, schema, keywords, False, ())
    except _Cycle:
        return None
    return written if isinstance(written, dict) and written != schema else None


def build_corpus(tree: Path) -> dict[str, object]:
    """The keyword table and the corpus and suite schemas of the checkout at `tree`, each with its written-out form."""
    spec = importlib.util.spec_from_file_location('_conformance', tree / 'tests' / 'test_json_schema_conformance.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    keywords: dict[str, Shape] = dict(module.SUBSCHEMA_KEYWORDS)
    schemas: dict[str, dict[str, object]] = {name: deepcopy(schema) for name, schema in module.CORPUS.items()}
    for path in sorted((tree / 'tests' / 'json_schema_test_suite').rglob('*.json')):
        for index, group in enumerate(json.loads(path.read_text(encoding='utf-8'))):
            if isinstance(schema := group['schema'], dict):
                schemas[f'suite/{path.parent.name}/{path.stem}/{index}'] = schema
    cases = {
        name: {'schema': schema, 'written_out': _with_written_out(schema, keywords)} for name, schema in schemas.items()
    }
    return {'keywords': keywords, 'cases': cases}


def _transformer_classes() -> dict[str, type]:
    """Every concrete `JsonSchemaTransformer` subclass the installed `pydantic_ai` package defines."""
    from pydantic_ai._json_schema import JsonSchemaTransformer

    package = Path(inspect.getfile(JsonSchemaTransformer)).parent
    for path in package.rglob('*.py'):
        source = path.read_text(encoding='utf-8')
        if 'SchemaTransformer' in source and any(
            isinstance(node, ast.ClassDef)
            and any(ast.unparse(base).endswith('SchemaTransformer') for base in node.bases)
            for node in ast.walk(ast.parse(source))
        ):
            try:
                importlib.import_module('.'.join(('pydantic_ai', *path.relative_to(package).with_suffix('').parts)))
            except ImportError:
                print(f'skipped {path}: its provider SDK is not installed', file=sys.stderr)
    found: dict[str, type] = {}
    pending: list[type] = [JsonSchemaTransformer]
    while pending:
        for subclass in pending.pop().__subclasses__():
            pending.append(subclass)
            if not inspect.isabstract(subclass) and subclass.__module__.startswith('pydantic_ai.'):
                found[f'{subclass.__module__}.{subclass.__qualname__}'] = subclass
    return found


def _refs(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            if key == '$ref' and isinstance(value, str):
                yield value
            else:
                yield from _refs(value)
    elif isinstance(node, list):
        for value in node:
            yield from _refs(value)


def _resolves(schema: object, ref: str) -> bool:
    node = schema
    for token in ref.removeprefix('#').split('/')[1:]:
        token = unquote(token).replace('~1', '/').replace('~0', '~')
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            return False
    return True


def _walk(cls: type, schema: dict[str, object], kwargs: dict[str, bool]) -> Record:
    try:
        transformer = cls(deepcopy(schema), **kwargs)
        output = transformer.walk()
    except Exception as error:
        return {'raises': f'{type(error).__module__}.{type(error).__qualname__}: {error}'}
    return {
        'output': repr(output),
        'strict_compatible': transformer.is_strict_compatible,
        # Only local refs: a relative URI like `./baseSchema` resolves against an `$id`, not the root.
        'dangling': [ref for ref in _refs(output) if ref.startswith('#') and not _resolves(output, ref)],
    }


def dump(corpus: Mapping[str, Mapping[str, object]]) -> dict[str, Record]:
    """Every walk of every case and written-out form, keyed `transformer|strict|case|form`."""
    records: dict[str, Record] = {}
    for name, cls in sorted(_transformer_classes().items()):
        for variant, kwargs in STRICT_VARIANTS.items():
            for case, inputs in corpus.items():
                for form in ('schema', 'written_out'):
                    if isinstance(schema := inputs[form], dict):
                        records[f'{name}|{variant}|{case}|{form}'] = _walk(cls, schema, kwargs)
    return records


def _classify(base: Record, head: Record, written_out_base: Record | None) -> str:
    """The outcome of one walk on HEAD against BASE; names in capitals fail the comparison."""
    if 'raises' in base:
        return 'raise unchanged' if head == base else 'raise changed'
    if 'raises' in head:
        return 'NEW RAISE'
    if head['output'] == base['output']:
        if head['strict_compatible'] != base['strict_compatible']:
            return 'STRICT FLAG CHANGED'
        return 'still dangling' if base['dangling'] else 'identical'
    if not base['dangling']:
        return 'CHANGED WITHOUT DANGLING'
    if head['dangling']:
        return 'changed, still dangling'
    if written_out_base is None:
        return 'fixed, no written-out form'
    return 'fixed as written' if head['output'] == written_out_base.get('output') else 'FIXED OTHERWISE'


def compare(base: Mapping[str, Record], head: Mapping[str, Record]) -> int:
    """Print each outcome of HEAD against BASE, and return 1 if any fails."""
    outcomes: dict[str, list[str]] = {}
    for key in sorted(base.keys() | head.keys()):
        if not key.endswith('|schema'):
            continue
        if key not in base or key not in head:
            outcomes.setdefault('ONLY ON ONE SIDE', []).append(key)
            continue
        written_out_base = base.get(key.removesuffix('|schema') + '|written_out')
        outcomes.setdefault(_classify(base[key], head[key], written_out_base), []).append(key)
    print(json.dumps(Counter({outcome: len(keys) for outcome, keys in outcomes.items()}), indent=2))
    failed = {outcome: keys for outcome, keys in outcomes.items() if outcome.isupper()}
    for outcome, keys in failed.items():
        print(f'\n{outcome}:', *keys[:40], *([f'... {len(keys) - 40} more'] if len(keys) > 40 else []), sep='\n  ')
    for outcome in ('still dangling', 'changed, still dangling'):
        # transformer|strict|case|form: one line per case, with how many transformer and `strict` walks it covers
        by_case = Counter(key.split('|')[2] for key in outcomes.get(outcome, []))
        print(f'\n{outcome}, walks by case:', *(f'{case}: {n}' for case, n in sorted(by_case.items())), sep='\n  ')
    return 1 if failed else 0


def main() -> int:
    """Run the step the command line names."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest='step', required=True)
    corpus_step = steps.add_parser('corpus')
    corpus_step.add_argument('tree', type=Path)
    corpus_step.add_argument('out', type=Path)
    dump_step = steps.add_parser('dump')
    dump_step.add_argument('corpus', type=Path)
    dump_step.add_argument('out', type=Path)
    compare_step = steps.add_parser('compare')
    compare_step.add_argument('base', type=Path)
    compare_step.add_argument('head', type=Path)
    args = parser.parse_args()

    if args.step == 'corpus':
        args.out.write_text(json.dumps(build_corpus(args.tree)), encoding='utf-8')
        return 0
    if args.step == 'dump':
        corpus = json.loads(args.corpus.read_text(encoding='utf-8'))['cases']
        args.out.write_text(json.dumps(dump(corpus)), encoding='utf-8')
        return 0
    return compare(json.loads(args.base.read_text(encoding='utf-8')), json.loads(args.head.read_text(encoding='utf-8')))


if __name__ == '__main__':
    sys.exit(main())
