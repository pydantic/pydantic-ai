# Testing conventions

## general rules

- prefer using `snapshot()` instead of line-by-line assertions
- unless the snapshot is too big and you only need to check specific values
- kwargs are big no no in this codebase because they're untyped
```python
    # unacceptable
    def _make_agent(model: AnthropicModel, **agent_kwargs: Any) -> Agent:
```

### about static typing

- other codebases don't use types in their test files
- but this codebase is fully typed with static types
- proper types are required and the pre-commit hook sstrictly checks for types and won't allow commits with type errors
- so you're required to use proper types in test files as well
- refer to `tests/models/anthropic/conftest.py` for examples of typing in test files

## for testing filepaths

- define your function with a parameter `tmp_path: Path`

## examples

### inline vs snapshot
```python
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    assert 'tools' in completion_kwargs
    tools = completion_kwargs['tools']
    # By default, tools should be strict-compatible
    assert any(tool.get('strict') is True for tool in tools)
    # Should include structured-outputs beta
    assert 'structured-outputs-2025-11-13' in completion_kwargs.get('betas', [])
```

can be simplified to

```python
    completion_kwargs = get_mock_chat_completion_kwargs(mock_client)[0]
    tools = completion_kwargs['tools']
    betas = completion_kwargs['betas']
    assert tools = snapshot()
    assert betas = snapshot()
```

- it's preferable to use the snapshot, run the test and check what comes out
- if the snapshot is too large in comparison with the equivalent inline assertions, it's ok to keep the inline assertions
- confirm with the user what they prefer in cases that don't have a clear preference
