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
	uv run ruff format $(sources)
	uv run ruff check --fix --fix-only $(sources)

.PHONY: lint  # Lint the code
lint:
	uv run ruff format --check $(sources)
	uv run ruff check $(sources)

.PHONY: typecheck-pyright
typecheck-pyright:
	uv run pyright

.PHONY: typecheck-mypy
typecheck-mypy:
	uv run mypy --strict tests/typed_agent.py

.PHONY: typecheck  # Run static type checking
typecheck: typecheck-pyright

.PHONY: test  # Run tests and collect coverage data
test:
	uv run coverage run -m pytest

.PHONY: testcov  # Run tests and generate a coverage report
testcov: test
	@echo "building coverage html"
	@uv run coverage html --show-contexts
	@uv run coverage report

.PHONY: all
all: format lint typecheck test
