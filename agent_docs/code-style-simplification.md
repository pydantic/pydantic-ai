# Code Style & Simplification

> Rules for writing clean, maintainable code including comprehensions, removing redundant code, consolidating conditionals, eliminating intermediate variables, and using walrus operator. Covers code duplication, default parameters, and boolean expression patterns.

**When to check**: When writing loops, conditionals, variable assignments, or refactoring code for clarity

## Rules

<!-- rule:255 -->
- Use comprehensions over loop+append when building collections — more readable and idiomatic Python — Comprehensions (list/dict/set) are clearer, more concise, and signal intent better than initializing empty collections and populating with loops
<!-- rule:122 -->
- Flatten nested `if` statements without intervening code into `if condition1 and condition2:` — Improves readability and reduces indentation depth, making control flow easier to follow
<!-- rule:341 -->
- Delete commented-out code, unused definitions, and obsolete implementations — rely on git history — Dead code obscures actual control flow, increases maintenance burden, and risks bugs if wrong implementation is used
<!-- rule:-1 -->
- Use tuple syntax for `isinstance()` checks, not `|` union — tuples are faster at runtime — Tuple syntax `isinstance(obj, (type1, type2))` has better runtime performance than union syntax `isinstance(obj, type1 | type2)`
<!-- rule:559 -->
- Consolidate identical logic in conditional branches — combine conditions with `or`, extract differing values to variables, or pull shared code outside the `if`/`elif` block — Reduces duplication, makes changes safer (single edit point), and improves readability by surfacing actual differences between branches
<!-- rule:166 -->
- Avoid single-use intermediate variables — return or reassign directly instead of creating `_filtered`, `_to_format`, or pass-through copies — Reduces cognitive load and line noise by eliminating variables that don't represent meaningful state transformations
<!-- rule:2 -->
- Extract duplicated logic into shared helpers after 2+ occurrences — reduces maintenance burden and prevents inconsistencies — Co-locating mapping dictionaries with transformation helpers creates a single source of truth, making updates safer and easier to verify across the codebase
<!-- rule:85 -->
- Omit parameters that match default values — reduces noise and makes non-default config explicit — Highlighting only non-default values makes the code's intent clearer and reduces maintenance burden when defaults change
<!-- rule:74 -->
- Use walrus operator (`:=`) to combine assignment with conditionals — avoids redundant access, repeated computations, and separate assignment statements — Reduces bugs from inconsistent get/check/delete patterns (especially with `dict.pop()`), improves readability for nested optional attributes and fallback chains, and eliminates redundant operations
<!-- rule:330 -->
- Replace for-loops that check conditions with `any()` or `all()` — more Pythonic and eliminates manual flag variables — Reduces boilerplate, prevents flag management bugs, and improves readability through idiomatic Python patterns
<!-- rule:1001 -->
- Define `TypeAdapter` instances at module level as constants — avoids repeated construction overhead — Creating `TypeAdapter` instances repeatedly inside functions or loops incurs unnecessary performance costs; module-level constants are constructed once and reused.
<!-- rule:41 -->
- Order required fields before optional fields in dataclasses — Python dataclasses raise a syntax error when required fields (no defaults) follow optional fields (with defaults)
<!-- rule:677 -->
- Use `@cached_property` for expensive computed attributes instead of eager computation in `__init__` — Enables lazy evaluation that defers costly operations until needed and caches results, improving initialization performance and reducing unnecessary computation
<!-- rule:358 -->
- Extract validation logic into shared helpers when patterns repeat 2-3+ times — Prevents validation logic from diverging across locations, reducing bugs from inconsistent checks and making updates easier
<!-- rule:3 -->
- Use `x or default` for simple fallbacks instead of if/else or ternary — Reduces boilerplate and improves readability for common fallback patterns, but only when falsy values (0, '', [], etc.) shouldn't be preserved
<!-- rule:661 -->
- Remove null/None checks for attributes guaranteed by type signatures, constructors, or data models — Defensive checks for impossible conditions create noise and obscure actual invariants; if a field can't be None, don't test for it
<!-- rule:224 -->
- Use built-in serialization methods (`args_as_json_str()`, `.model_dump()`, `.dict()`, `.to_dict()`) instead of manual `isinstance()` checks or ad-hoc `str()`/`json.dumps()` patterns — Prevents type-checking bugs, ensures consistent serialization behavior, and leverages validated serialization logic already in the codebase
<!-- rule:499 -->
- Compile static regex patterns at module level as constants — prevents recompilation overhead on repeated function calls — Compiling regex patterns once at module load time instead of inside functions improves performance by avoiding redundant compilation on every invocation
