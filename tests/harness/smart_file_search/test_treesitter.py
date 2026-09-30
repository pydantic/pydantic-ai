"""`SmartFileSearch` syntax-aware chunking for fifteen languages beyond Python (tree-sitter).

Ported from Code Puppy's `code_puppy_core_plugins/jev_grep` tests.
"""

from __future__ import annotations

import textwrap

import pytest

from pydantic_ai_harness.smart_file_search import _treesitter as treesitter
from pydantic_ai_harness.smart_file_search._chunks import source_chunks

Ranges = list[tuple[int, int, str | None]]


def _ranges(text: str, path: str) -> tuple[Ranges, str]:
    chunks, parser = source_chunks(text, path)
    return [(c.line, c.end_line, c.symbol) for c in chunks], parser


def _assert_exact_coverage(text: str, path: str) -> None:
    chunks, _ = source_chunks(text, path)
    covered = {n for c in chunks for n in range(c.line, c.end_line + 1)}
    for number, line in enumerate(text.splitlines(), start=1):
        assert not line.strip() or number in covered, f'{path}:{number} uncovered'


GO = textwrap.dedent(
    """\
    package server

    import "net/http"

    type Server struct {
    \tmux *http.ServeMux
    }

    // Serve rejects expired sessions before routing.
    func (s *Server) Serve(w http.ResponseWriter, r *http.Request) {
    \tif expired(r) {
    \t\thttp.Error(w, "expired", http.StatusUnauthorized)
    \t\treturn
    \t}
    \ts.mux.ServeHTTP(w, r)
    }

    func expired(r *http.Request) bool {
    \treturn r.Header.Get("X-Expired") != ""
    }
    """
)

RUST = textwrap.dedent(
    """\
    use std::collections::HashMap;

    pub struct Cache {
        map: HashMap<String, u64>,
    }

    impl Cache {
        pub fn get(&self, key: &str) -> Option<&u64> {
            self.map.get(key)
        }

        pub fn evict_expired(&mut self, now: u64) {
            self.map.retain(|_, deadline| *deadline > now);
        }
    }
    """
)

JAVA = textwrap.dedent(
    """\
    package app;

    import java.util.Optional;

    @Service
    public class UserService {
        private final Repo repo;

        public UserService(Repo repo) {
            this.repo = repo;
        }

        public Optional<User> find(long id) {
            if (id < 0) {
                throw new IllegalArgumentException("negative id");
            }
            return repo.get(id);
        }
    }
    """
)

TS = textwrap.dedent(
    """\
    import { db } from "./db";

    /** Fetch one user or fail loudly. */
    export async function fetchUser(id: string): Promise<User> {
      if (!id) {
        throw new Error("missing id");
      }
      return db.users.get(id);
    }

    export const retry = async (fn: () => Promise<void>, attempts = 3) => {
      for (let i = 0; i < attempts; i++) {
        try {
          return await fn();
        } catch (err) {
          console.warn(err);
        }
      }
    };

    @Injectable()
    export class SessionService {
      constructor(private readonly store: Store) {}

      isExpired(session: Session): boolean {
        return session.expiresAt <= Date.now();
      }
    }
    """
)


JS = textwrap.dedent(
    """\
    const express = require("express");

    async function fetchUser(id) {
      if (!id) {
        throw new Error("missing id");
      }
      return db.users.get(id);
    }

    class Cache {
      get(key) {
        return this.map.get(key);
      }
    }

    module.exports = { fetchUser, Cache };
    """
)


def test_javascript_functions_and_classes() -> None:
    ranges, parser = _ranges(JS, 'api.js')
    assert parser == 'javascript'
    assert {'fetchUser', 'Cache.get'} <= {symbol for _, _, symbol in ranges}


def test_go_methods_are_named_by_receiver() -> None:
    ranges, parser = _ranges(GO, 'server.go')
    assert parser == 'go'
    symbols = [symbol for _, _, symbol in ranges]
    assert 'Server' in symbols  # type declaration named via its type_spec
    assert 'Server.Serve' in symbols
    assert 'expired' in symbols
    serve = next(r for r in ranges if r[2] == 'Server.Serve')
    assert serve[0] <= 9 and serve[1] == 16  # leading comment absorbed


def test_rust_impl_blocks_split_into_methods() -> None:
    ranges, parser = _ranges(RUST, 'cache.rs')
    assert parser == 'rust'
    symbols = [symbol for _, _, symbol in ranges]
    assert {'Cache.get', 'Cache.evict_expired'} <= set(symbols)
    assert 'HashMap' not in symbols  # imports never earn a symbol boost


def test_java_class_members_and_annotations() -> None:
    ranges, parser = _ranges(JAVA, 'UserService.java')
    assert parser == 'java'
    by_symbol = {symbol: (start, end) for start, end, symbol in ranges}
    header = by_symbol['UserService']
    assert header[0] <= 5 <= header[1]  # @Service annotation belongs to the class
    find = by_symbol['UserService.find']
    assert find[0] <= 13 and find[1] >= 18  # whole method, plus absorbed gaps
    assert 'UserService.UserService' in by_symbol  # constructor
    assert 'Optional' not in by_symbol


def test_typescript_exports_arrows_and_classes() -> None:
    ranges, parser = _ranges(TS, 'api.ts')
    assert parser == 'typescript'
    symbols = [symbol for _, _, symbol in ranges]
    assert {'fetchUser', 'retry', 'SessionService.isExpired'} <= set(symbols)
    assert 'SessionService.constructor' in symbols


@pytest.mark.parametrize(
    'text, path, parser',
    [
        (TS, 'api.ts', 'typescript'),
        (TS.replace('@Injectable()\n', ''), 'api.tsx', 'tsx'),
        (JS, 'api.js', 'javascript'),
        (GO, 'server.go', 'go'),
        (RUST, 'cache.rs', 'rust'),
        (JAVA, 'UserService.java', 'java'),
    ],
)
def test_every_language_covers_every_line(text: str, path: str, parser: str) -> None:
    assert source_chunks(text, path)[1] == parser
    _assert_exact_coverage(text, path)


def test_long_function_splits_into_blocks_with_full_coverage() -> None:
    body = '\n'.join(
        f'  if (x === {i}) {{\n    a = {i};\n    b = {i};\n    c = {i};\n    d = {i};\n  }}' for i in range(6)
    )
    text = f'export function route(x) {{\n{body}\n}}\n'
    ranges, _ = _ranges(text, 'router.js')
    assert len(ranges) > 1
    assert {symbol for _, _, symbol in ranges} == {'route'}
    _assert_exact_coverage(text, 'router.js')


@pytest.mark.parametrize('text, path', [(TS, 'api.ts'), (GO, 'server.go')])
def test_windows_line_endings_give_identical_snippets(text: str, path: str) -> None:
    assert _ranges(text, path) == _ranges(text.replace('\n', '\r\n'), path)


def test_unparsable_region_uses_line_windows() -> None:
    ranges, parser = _ranges('export function oops( {\n  return 1;\n', 'broken.ts')
    assert parser == 'typescript-partial'
    assert ranges == [(1, 2, None)]


def test_parse_error_costs_its_region_not_the_whole_file() -> None:
    """C macros commonly defeat the grammar; the clean functions around one
    must keep their structure and names."""
    text = textwrap.dedent(
        """\
        static int ok(int a) {
          return a + 1;
        }

        FOREACH(item, list) {
          use(item)
        }

        int also_ok(void) {
          return 2;
        }
        """
    )
    ranges, parser = _ranges(text, 'macro.c')
    assert parser == 'c-partial'
    symbols = {symbol for _, _, symbol in ranges}
    assert {'ok', 'also_ok'} <= symbols
    parsed = treesitter.treesitter_ranges(text, 'macro.c')
    assert parsed is not None
    broken = [r for r in parsed[0] if r.window]
    assert broken and all(5 <= r.start <= 7 for r in broken)
    _assert_exact_coverage(text, 'macro.c')


def test_header_falls_back_to_c_grammar() -> None:
    """K&R definitions are C, not C++: .h retries the C grammar."""
    text = 'int add(a, b)\nint a;\nint b;\n{\n  return a + b;\n}\n'
    ranges, parser = _ranges(text, 'old.h')
    assert parser == 'c'
    assert ranges[0][2] == 'add'


def test_container_of_one_liners_stays_whole() -> None:
    text = 'class Point {\n  int x;\n  int y;\n}\n'
    assert _ranges(text, 'Point.java')[0] == [(1, 4, 'Point')]


def test_short_tail_joins_last_range_long_tail_does_not() -> None:
    code = 'export function f() {\n  return 1;\n}\n'
    assert _ranges(code + '// done\n', 'a.ts')[0] == [(1, 4, 'f')]
    long_tail = code + ''.join(f'// note {i}\n' for i in range(10))
    assert _ranges(long_tail, 'a.ts')[0] == [(1, 3, 'f'), (4, 13, None)]


# Realistic code per newly supported language: (path, parser, source, symbols
# that must be found). Every sample must also cover every line exactly.
MORE_LANGUAGES: list[tuple[str, str, str, set[str]]] = [
    (
        'server.c',
        'c',
        """\
#include <stdlib.h>

static int *make_buf(size_t n) {
    int *buf = calloc(n, sizeof(int));
    return buf;
}

int main(void) {
    return make_buf(4) != NULL;
}
""",
        {'make_buf', 'main'},
    ),
    (
        'server.cpp',
        'cpp',
        """\
#include <string>

namespace net {

class Server {
 public:
  explicit Server(int port) : port_(port) {}

  void serve() {
    while (running_) {
      accept();
    }
  }

 private:
  int port_;
};

int Server::port() const {
  return port_;
}

}  // namespace net
""",
        {'Server', 'Server.serve', 'Server.port'},
    ),
    (
        'UserService.cs',
        'csharp',
        """\
using System;

namespace App.Services
{
    public class UserService
    {
        private readonly Repo _repo;

        public UserService(Repo repo)
        {
            _repo = repo;
        }

        public User Find(long id)
        {
            return _repo.Get(id);
        }
    }
}
""",
        {'UserService', 'UserService.UserService', 'UserService.Find'},
    ),
    (
        'invoice.rb',
        'ruby',
        """\
require 'json'

module Billing
  class Invoice
    def initialize(total)
      @total = total
    end

    def paid?
      status == :paid
    end
  end
end
""",
        {'Billing.Invoice.initialize', 'Billing.Invoice.paid?'},
    ),
    (
        'UserController.php',
        'php',
        """\
<?php
namespace App;

class UserController
{
    public function show(int $id)
    {
        return $this->repo->find($id);
    }
}

function helper($x)
{
    return $x;
}
""",
        {'UserController.show', 'helper'},
    ),
    (
        'Cache.kt',
        'kotlin',
        """\
package app

class Cache(private val size: Int) {
    fun get(key: String): Int? {
        return map[key]
    }
}

fun topLevel(x: Int): Int {
    return x + 1
}
""",
        {'Cache.get', 'topLevel'},
    ),
    (
        'Session.swift',
        'swift',
        """\
import Foundation

class Session {
    var expiresAt: Date

    func isExpired() -> Bool {
        return expiresAt < Date()
    }
}
""",
        {'Session', 'Session.isExpired'},
    ),
    (
        'Cache.scala',
        'scala',
        """\
package app

class Cache(size: Int) {
  def get(key: String): Option[Int] = {
    map.get(key)
  }
}
""",
        {'Cache', 'Cache.get'},
    ),
    (
        'deploy.sh',
        'bash',
        """\
#!/usr/bin/env bash
set -euo pipefail

deploy() {
  echo "deploying"
  rsync -a . host:/srv
}

deploy
""",
        {'deploy'},
    ),
    (
        'account.lua',
        'lua',
        """\
local M = {}

function M.greet(name)
  return "hi " .. name
end

function Account:deposit(v)
  self.balance = self.balance + v
end

return M
""",
        {'M.greet', 'Account.deposit'},
    ),
]


@pytest.mark.parametrize('path, parser, text, symbols', MORE_LANGUAGES)
def test_more_languages_are_structured(path: str, parser: str, text: str, symbols: set[str]) -> None:
    ranges, used = _ranges(text, path)
    assert used == parser
    assert symbols <= {symbol for _, _, symbol in ranges}
    _assert_exact_coverage(text, path)


def test_call_sites_and_namespaces_are_not_symbols() -> None:
    bash = next(t for p, _, t, _ in MORE_LANGUAGES if p == 'deploy.sh')
    assert 'set' not in {s for _, _, s in _ranges(bash, 'deploy.sh')[0]}
    cpp = next(t for p, _, t, _ in MORE_LANGUAGES if p == 'server.cpp')
    assert not any(s and s.startswith('net') for _, _, s in _ranges(cpp, 'server.cpp')[0])


def test_unsupported_extension_uses_windows() -> None:
    assert source_chunks('# Title\n\nprose\n', 'notes.md')[1] == 'overlapping-lines'


def test_known_bad_tree_sitter_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """0.26.x segfaults natively; it must degrade to windows, never be used."""
    treesitter._parser.cache_clear()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(treesitter, 'version', lambda name: '0.26.0')  # pyright: ignore[reportUnknownLambdaType,reportUnknownArgumentType]
    try:
        assert treesitter._tree_sitter_is_safe() is False  # pyright: ignore[reportPrivateUsage]
        assert source_chunks(GO, 'server.go')[1] == 'overlapping-lines'
    finally:
        treesitter._parser.cache_clear()  # pyright: ignore[reportPrivateUsage]


def test_missing_grammar_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_loader() -> object:
        raise ImportError('grammar wheel not available on this platform')

    treesitter._parser.cache_clear()  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setitem(treesitter.LANGUAGES, '.go', ('go', broken_loader))
    try:
        assert source_chunks(GO, 'server.go')[1] == 'overlapping-lines'
    finally:
        treesitter._parser.cache_clear()  # pyright: ignore[reportPrivateUsage]


def test_unreadable_version_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> str:
        raise treesitter.PackageNotFoundError(name)

    monkeypatch.setattr(treesitter, 'version', missing)
    assert treesitter._tree_sitter_is_safe() is False  # pyright: ignore[reportPrivateUsage]


NESTED_GO = (
    'package main\n\nfunc deep() {\n\t{\n\t\t{\n\t\t\t{\n\t\t\t\t{\n'
    + ''.join(f'\t\t\t\t\tx{i}()\n' for i in range(30))
    + '\t\t\t\t}\n\t\t\t}\n\t\t}\n\t}\n}\n'
)


@pytest.mark.parametrize(
    ('path', 'text', 'parser', 'symbols'),
    [
        pytest.param(
            'exports.js', 'export { helper };\nfunction helper() {\n  return 1;\n}\n', 'javascript', {'helper'}
        ),
        pytest.param('max.cpp', 'template<typename T>\nT biggest(T a, T b) {\n  return a;\n}\n', 'cpp', {'biggest'}),
        pytest.param('partial.cpp', 'template<typename T>\n', 'cpp-partial', {None}),
        pytest.param('receiver.go', 'package main\n\nfunc (/* c */ s *Server) Serve() {\n}\n', 'go', {'Server.Serve'}),
        pytest.param('bare.go', 'package main\n\nfunc () Serve() {\n}\n', 'go', {'Serve'}),
        pytest.param('block.js', '{\n  let x = 1;\n}\n', 'javascript', {None}),
        pytest.param('deep.go', NESTED_GO, 'go', {'deep'}),
        pytest.param('broken.h', 'int f( {\n', 'cpp-partial', {None}),
    ],
)
def test_tree_sitter_edge_cases(path: str, text: str, parser: str, symbols: set[str | None]) -> None:
    chunks, used = source_chunks(text, path)
    assert used == parser
    assert symbols <= {c.symbol for c in chunks}


def test_file_without_declarations_uses_windows() -> None:
    assert source_chunks('// just a comment\n', 'only.go')[1] == 'overlapping-lines'


def test_nesting_deeper_than_the_stack_falls_back_to_windows() -> None:
    text = 'namespace a {\n' * 500 + 'int f() { return 1; }\n' + '}\n' * 500
    assert source_chunks(text, 'deep.cpp')[1] == 'overlapping-lines'
