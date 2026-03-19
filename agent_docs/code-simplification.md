# Code Simplification & Idioms

> Rules for simplifying code using Python idioms, comprehensions, operators, and eliminating unnecessary complexity

**When to check**: When refactoring code for clarity or looking to simplify complex patterns

## Rules

<!-- rule:255 -->
- Use list comprehensions instead of for-loop-with-append patterns — More concise, readable, and often faster for transforming/filtering iterables into lists
<!-- rule:85 -->
- Omit parameters that match default values in function/constructor calls — Reduces noise, prevents maintenance burden when defaults change, and makes non-default configuration more visible
<!-- rule:166 -->
- Eliminate single-use intermediate variables — reassign or return directly instead of creating `_filtered`, `_copy`, etc. — Reduces noise and indirection, making data flow clearer and eliminating unnecessary names that don't add semantic value.
<!-- rule:122 -->
- Flatten nested `if` statements with no intervening code into `if condition1 and condition2:` — Reduces nesting depth and improves readability without changing logic
<!-- rule:-1 -->
- Use tuple syntax for `isinstance()` checks, not `|` union — tuples are faster at runtime — Runtime performance optimization: tuple syntax avoids the overhead of union type creation
<!-- rule:34 -->
- Link to official provider/project docs instead of duplicating model lists, features, or setup details — prevents stale documentation and reduces maintenance burden — Exhaustive inline lists become outdated quickly; authoritative external sources stay current and reduce maintenance
<!-- rule:519 -->
- Use dict comprehensions instead of empty dict + loop — more concise and idiomatic Python — Reduces boilerplate, improves readability, and signals intent more clearly for simple mappings and filtered sequences
<!-- rule:1001 -->
- Define `TypeAdapter` instances at module level as constants — avoids repeated initialization overhead — Creating TypeAdapters repeatedly inside functions or loops wastes CPU cycles on redundant schema construction that only needs to happen once.
<!-- rule:330 -->
- Use `any()` instead of for-loops with boolean flags when checking if any element matches a condition — More concise and Pythonic; eliminates manual flag management and break statements, reducing potential for logic errors
<!-- rule:677 -->
- Use `@cached_property` for expensive computed attributes — defers computation until first access and caches the result — Improves startup performance by avoiding unnecessary computation and reduces memory overhead when attributes may not be used
<!-- rule:3 -->
- Use `x or default` for fallback values instead of verbose if-else blocks — more concise and idiomatic — This pattern is shorter and clearer for typical fallback logic, but avoid it when falsy values (0, '', [], None) are semantically valid and shouldn't trigger the default
<!-- rule:661 -->
- Remove redundant null/None checks for guaranteed-present values — simplifies code and makes type invariants clearer — Unnecessary defensive checks obscure actual invariants, add maintenance burden, and suggest false uncertainty about what the type system already guarantees
