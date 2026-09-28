from __future__ import annotations

import importlib.util

# Syntax-aware chunking beyond Python needs the `smart-grep` extra; without it those files use line windows.
collect_ignore = ['test_treesitter.py'] if importlib.util.find_spec('tree_sitter') is None else []
