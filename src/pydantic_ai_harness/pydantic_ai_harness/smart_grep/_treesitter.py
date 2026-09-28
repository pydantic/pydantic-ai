"""tree-sitter `Syntax` for fifteen languages beyond Python.

JavaScript/TypeScript, Go, Rust, Java, C, C++, C#, Ruby, PHP, Kotlin, Swift, Scala, Bash and Lua.

Grammars are pinned, offline wheels (one package per language, installed by the `smart-grep` extra);
nothing is downloaded at runtime. They are optional at runtime: if tree-sitter or a grammar cannot be
imported (not installed, or an unsupported platform) or the installed tree-sitter is a known-bad version,
`treesitter_ranges` returns `None` and the caller falls back to line windows. A file with parse errors
(macros, newer syntax than the grammar knows) keeps its structure where the tree is clean; only the broken
regions fall back to line windows.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Iterable, Sequence
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from pydantic_ai_harness.smart_grep._structure import Range, structure_ranges

if TYPE_CHECKING:
    from tree_sitter import Node, Parser

GrammarLoader = Callable[[], object]
"""Returns a grammar's language pointer, as `tree_sitter.Language` accepts it."""


def _grammar(module: str, attr: str = 'language') -> GrammarLoader:
    """A lazy loader for one grammar (the wheel is imported on first use)."""

    def load() -> object:
        language: object = getattr(importlib.import_module(module), attr)
        if not callable(language):  # pragma: no cover - every pinned grammar exposes a function
            raise TypeError(f'{module}.{attr} is not callable')
        return language()

    load.__qualname__ = f'load<{module}.{attr}>'
    return load


_JAVASCRIPT = ('javascript', _grammar('tree_sitter_javascript'))
_TYPESCRIPT = ('typescript', _grammar('tree_sitter_typescript', 'language_typescript'))
_C = ('c', _grammar('tree_sitter_c'))
_CPP = ('cpp', _grammar('tree_sitter_cpp'))

LANGUAGES: dict[str, tuple[str, GrammarLoader]] = {
    **dict.fromkeys(('.js', '.jsx', '.mjs', '.cjs'), _JAVASCRIPT),
    **dict.fromkeys(('.ts', '.mts', '.cts'), _TYPESCRIPT),
    '.tsx': ('tsx', _grammar('tree_sitter_typescript', 'language_tsx')),
    '.go': ('go', _grammar('tree_sitter_go')),
    '.rs': ('rust', _grammar('tree_sitter_rust')),
    '.java': ('java', _grammar('tree_sitter_java')),
    '.c': _C,
    # .h is C or C++: parse as C++ (a near-superset), retry as C (`_RETRY`).
    **dict.fromkeys(
        ('.h', '.cc', '.cpp', '.cxx', '.c++', '.hh', '.hpp', '.hxx', '.h++', '.ipp', '.tpp', '.inl'),
        _CPP,
    ),
    '.cs': ('csharp', _grammar('tree_sitter_c_sharp')),
    **dict.fromkeys(('.rb', '.rake', '.gemspec'), ('ruby', _grammar('tree_sitter_ruby'))),
    '.php': ('php', _grammar('tree_sitter_php', 'language_php')),
    **dict.fromkeys(('.kt', '.kts'), ('kotlin', _grammar('tree_sitter_kotlin'))),
    '.swift': ('swift', _grammar('tree_sitter_swift')),
    **dict.fromkeys(('.scala', '.sc'), ('scala', _grammar('tree_sitter_scala'))),
    **dict.fromkeys(('.sh', '.bash'), ('bash', _grammar('tree_sitter_bash'))),
    '.lua': ('lua', _grammar('tree_sitter_lua')),
}
"""File extension -> (parser name, grammar loader)."""

_RETRY: dict[str, tuple[tuple[str, GrammarLoader], ...]] = {'.h': (_C,)}
"""Grammars tried in order when the primary grammar leaves parse errors."""

_CONTAINERS = frozenset(
    {
        'class_declaration',  # Java, C#, PHP, Kotlin, Swift (class/struct/extension)
        'abstract_class_declaration',
        'class',  # JS class expressions, Ruby
        'enum_declaration',
        'record_declaration',
        'struct_declaration',  # C#
        'impl_item',  # Rust
        'trait_item',
        'mod_item',
        'class_specifier',  # C++ (C structs are data and stay whole)
        'module',  # Ruby
        'trait_declaration',  # PHP
        'object_declaration',  # Kotlin
        'companion_object',
        'class_definition',  # Scala
        'object_definition',
        'trait_definition',
    }
)
"""Declarations split into a header plus one range per member."""

_BLOCKS = frozenset(
    {
        'statement_block',
        'block',
        'class_body',
        'declaration_list',
        'enum_body',
        'switch_body',
        'compound_statement',  # C, C++, PHP, Bash
        'body_statement',  # Ruby
        'function_body',  # Kotlin, Swift
        'template_body',  # Scala
        'enum_class_body',  # Kotlin
    }
)
"""Bodies: of functions (statements) and of containers (members)."""

_STATEMENT_WRAPPERS = frozenset({'statement_list', 'statements', 'block', 'body_statement'})
"""A body that holds its statements in one wrapper node (Go, Swift, Kotlin)."""

_TRANSPARENT = frozenset(
    {
        'preproc_ifdef',
        'preproc_if',
        'preproc_else',
        'preproc_elif',
        'preproc_elifdef',
        'namespace_definition',  # C++, PHP (braced)
        'namespace_declaration',  # C#
        'file_scoped_namespace_declaration',  # C# `namespace X;`
        'linkage_specification',  # extern "C" { ... }
        'declaration_list',  # the body of the three above
    }
)
"""Unwrapped wherever they appear: they scope code rather than being code.

A namespace or include guard often wraps a whole file; kept, it would make the file one range.
"""

_TRANSPARENT_FIELDS = ('name', 'condition', 'value')
"""The transparent wrapper's own label fields."""

_UNNAMED = frozenset(
    {
        'import_declaration',
        'package_declaration',
        'use_declaration',
        'preproc_include',
        'using_directive',  # C#
        'namespace_use_declaration',  # PHP
        'import',  # Kotlin
        'package_header',
        'package_clause',  # Scala
        'command',  # Bash: a call site, not a declaration
    }
)
"""Never named: an imported name would earn BM25's symbol boost for a line that is almost never the answer
(Python's `ast` leaves imports unnamed too)."""

_DECLARATORS = frozenset({'lexical_declaration', 'variable_declaration'})

_DECLARATOR_NAMES = frozenset(
    {
        'identifier',
        'field_identifier',
        'type_identifier',
        'qualified_identifier',
        'destructor_name',
        'operator_name',
    }
)
"""Where a C/C++ declarator chain ends in a name."""

_QUALIFIED = frozenset({'qualified_identifier', 'method_index_expression'})
_SKIPPED = frozenset({'access_specifier'})  # C++ `public:` labels
_BODY_SEARCH_DEPTH = 4
_DECLARATOR_DEPTH = 8
_MAX_ABSORBED_TAIL = 3
"""Lines: a closing brace or `#endif`, not a comment block."""

_BROKEN_TREE_SITTER = ((0, 26),)
"""py-tree-sitter 0.26.0 segfaults walking ordinary trees (reproduced on CPython 3.13 and 3.14; 0.25.2 is
clean on the same input). A native crash would kill the whole process, so refuse it even if something pins it.
The `smart-grep` extra caps `tree-sitter<0.26` for the same reason."""


def _tree_sitter_is_safe() -> bool:
    try:
        major, minor = (int(p) for p in version('tree-sitter').split('.')[:2])
    except (PackageNotFoundError, ValueError):
        return False
    return (major, minor) not in _BROKEN_TREE_SITTER


def _code_children(node: Node) -> list[Node]:
    """Named children minus comments and labels.

    A skipped leading comment is absorbed into the declaration after it (as Python's `ast`, which has no
    comment nodes, already behaves).
    """
    return [c for c in node.named_children if 'comment' not in c.type and c.type not in _SKIPPED]


def _flatten(nodes: Iterable[Node]) -> list[Node]:
    """Replace namespaces, include guards and `extern "C"` blocks with the declarations inside them.

    Their own lines are absorbed by the neighbouring ranges, so line coverage is unchanged.
    """
    out: list[Node] = []
    for node in nodes:
        if node.type not in _TRANSPARENT:
            out.append(node)
            continue
        labels = {
            (c.start_byte, c.end_byte)
            for field in _TRANSPARENT_FIELDS
            if (c := node.child_by_field_name(field)) is not None
        }
        inner = [c for c in _code_children(node) if (c.start_byte, c.end_byte) not in labels]
        out.extend(_flatten(inner))
    return out


class TreeSitterSyntax:
    """`Syntax` over any tree-sitter grammar.

    Node text is sliced from the `source` bytes that were parsed, by byte offset, so names come from exactly
    the input we hold.
    """

    def __init__(self, source: bytes) -> None:
        self._source = source

    def _text(self, node: Node) -> str:
        return self._source[node.start_byte : node.end_byte].decode('utf-8', errors='replace')

    @staticmethod
    def _inner(node: Node) -> Node:
        """The declaration an `export` or `template<...>` wraps (the wrapper keeps the span)."""
        if node.type == 'export_statement':
            inner = node.child_by_field_name('declaration')
            if inner is not None:
                return inner
        elif node.type == 'template_declaration':
            # The declaration is the last child. A broken tree may end at the parameter list, which has no
            # name, members or body either, so returning it is as good as returning the wrapper.
            return next(reversed(_code_children(node)), node)
        return node

    def span(self, node: Node) -> tuple[int, int]:
        start = node.start_point.row + 1
        end = node.end_point.row + (1 if node.end_point.column else 0)
        return start, max(start, end)

    def _name(self, node: Node) -> str:
        text = self._text(node)
        if node.type in _QUALIFIED:  # C++ Server::port, Lua Account:deposit
            text = text.replace('::', '.').replace(':', '.')
        return text

    def symbol(self, node: Node) -> str | None:
        node = self._inner(node)
        if node.type in _UNNAMED:
            return None
        if node.type == 'impl_item':  # `impl Cache` / `impl Trait for Cache`
            target = node.child_by_field_name('type')
            return self._text(target) if target is not None else None
        name = node.child_by_field_name('name')
        if name is None:
            declared = self._declarator_name(node)  # C/C++: int *make_buf(...)
            if declared:
                return declared
            if node.type in _DECLARATORS or node.type.endswith('_declaration'):
                return self._declared_names(node)  # const a, b / Go type / Java field
            return None
        receiver = node.child_by_field_name('receiver')  # Go: (s *Server) Serve
        if receiver is not None:
            owner = self._receiver_type(receiver)
            if owner:
                return f'{owner}.{self._text(name)}'
        return self._name(name)

    def _declarator_name(self, node: Node) -> str | None:
        """Follow a C/C++ declarator chain (pointer -> function -> name)."""
        current = node.child_by_field_name('declarator')
        for _ in range(_DECLARATOR_DEPTH):
            if current is None:
                return None
            if current.type in _DECLARATOR_NAMES:
                return self._name(current)
            children = current.named_children
            current = current.child_by_field_name('declarator') or (children[0] if children else None)
        return None  # pragma: no cover - no real declarator chain is this deep

    def _declared_names(self, node: Node) -> str | None:
        """Names one level down: `const a, b`, Go `type_spec`, Java fields."""
        candidates = [node.child_by_field_name('declarator'), *node.named_children]
        names = dict.fromkeys(
            self._text(name)
            for child in candidates
            if child is not None and (name := child.child_by_field_name('name')) is not None
        )
        return ', '.join(names) or None

    def _receiver_type(self, receiver: Node) -> str | None:
        for param in receiver.named_children:
            kind = param.child_by_field_name('type')
            if kind is not None:
                return self._text(kind).lstrip('*&').split('[', 1)[0] or None
        return None  # pragma: no cover - a Go receiver always has a type

    def members(self, node: Node) -> Sequence[Node] | None:
        node = self._inner(node)
        if node.type not in _CONTAINERS:
            return None
        body = node.child_by_field_name('body') or next((c for c in node.named_children if c.type in _BLOCKS), None)
        members = _flatten(_code_children(body)) if body is not None else []
        # Only one-line members (fields, enum constants, prototypes, interface signatures): keep the
        # declaration whole. A range per line would cost a judgment each and hold no behaviour.
        if all(self.span(m)[1] - self.span(m)[0] < 1 for m in members):
            return None
        return members

    def broken(self, node: Node) -> bool:
        return node.type == 'ERROR' or node.has_error

    def statements(self, node: Node) -> Sequence[Node] | None:
        block = _body_block(self._inner(node))
        if block is None:
            return None
        children = _code_children(block)
        for _ in range(_BODY_SEARCH_DEPTH):  # function_body -> block -> statements
            if len(children) != 1 or children[0].type not in _STATEMENT_WRAPPERS:
                break
            children = _code_children(children[0])
        return children or None


def _body_block(node: Node) -> Node | None:
    """The nearest block under `node`: its own body, or one inside a wrapper.

    For example `const f = () => {...}` or `app.get("/x", (req, res) => {...})`.
    """
    if node.type in _BLOCKS:
        return node
    body = node.child_by_field_name('body')
    if body is not None and body.type in _BLOCKS:
        return body
    frontier = list(node.named_children)
    for _ in range(_BODY_SEARCH_DEPTH):
        for child in frontier:
            if child.type in _BLOCKS:
                return child
        frontier = [grand for child in frontier for grand in child.named_children]
    return None


@cache
def _parser(loader: GrammarLoader) -> Parser | None:
    if not _tree_sitter_is_safe():
        return None
    try:
        from tree_sitter import Language, Parser

        return Parser(Language(loader()))
    except Exception:  # unavailable grammar => line windows
        return None


def treesitter_ranges(text: str, path: str) -> tuple[list[Range], str] | None:
    """`(ranges, parser name)`, or `None` when this file should use line windows."""
    ext = os.path.splitext(path)[1].lower()
    entry = LANGUAGES.get(ext)
    if entry is None:
        return None
    source = text.encode('utf-8')
    syntax = TreeSitterSyntax(source)
    # First clean parse wins; otherwise the tree with the fewest broken lines.
    best: tuple[int, str, Node, list[Node]] | None = None
    for name, loader in (entry, *_RETRY.get(ext, ())):
        parser = _parser(loader)
        if parser is None:
            continue
        root = parser.parse(source).root_node
        nodes = _flatten(_code_children(root))
        broken = sum(syntax.span(n)[1] - syntax.span(n)[0] + 1 for n in nodes if syntax.broken(n))
        if best is None or broken < best[0]:
            best = (broken, name, root, nodes)
        if not broken:
            break
    if best is None:
        return None
    broken, name, root, nodes = best
    ranges = structure_ranges(nodes, syntax)
    if not ranges:
        return None
    # Close the last range over a short tail (a namespace's closing brace, an include guard's #endif)
    # instead of leaving it to become a judgment-costing snippet of its own. Longer tails stay separate.
    tail = syntax.span(root)[1]
    if 0 < tail - ranges[-1].end <= _MAX_ABSORBED_TAIL:
        ranges[-1].end = tail
    return ranges, f'{name}-partial' if broken else name
