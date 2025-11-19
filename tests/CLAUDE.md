# Testing conventions

## general rules

- prefer using `snapshot()` instead of line-by-line assertions
- unless the snapshot is too big and you only need to check specific values

## for testing filepaths

- define your function with a parameter `tmp_path: Path`