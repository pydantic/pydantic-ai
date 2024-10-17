.DEFAULT_GOAL := all
sources = pydantic_ai tests

.PHONY: .uv  # Check that uv is installed
.uv:
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit  # Check that pre-commit is installed
.pre-commit:
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install  # Install the package, dependencies, and pre-commit for local development
install: .uv .pre-commit
	uv sync --frozen
	pre-commit install --install-hooks

.PHONY: format  # Format the code
format:
	uv run --frozen ruff format $(sources)
	uv run --frozen ruff check --fix --fix-only $(sources)

.PHONY: lint  # Lint the code
lint:
	uv run --frozen ruff format --check $(sources)
	uv run --frozen ruff check $(sources)

.PHONY: typecheck-pyright
typecheck-pyright:
	uv run --frozen pyright

.PHONY: typecheck-mypy
typecheck-mypy:
	uv run --frozen mypy --strict tests/typed_agent.py

.PHONY: typecheck  # Run static type checking
typecheck: typecheck-pyright

.PHONY: test  # Run tests and collect coverage data
test:
	uv run --frozen coverage run -m pytest
	@uv run --frozen coverage report

.PHONY: testcov  # Run tests and generate a coverage report
testcov: test
	@echo "building coverage html"
	@uv run --frozen coverage html

.PHONY: all
all: format lint typecheck testcov
