.DEFAULT_GOAL := all
sources = pydantic_ai

.PHONY: .rye  # Check that Rye is installed
.rye:
	@rye --version || echo 'Please install Rye: https://rye-up.com/guide/installation/'

.PHONY: .pre-commit  # Check that pre-commit is installed
.pre-commit:
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install  # Install the package, dependencies, and pre-commit for local development
install: .rye .pre-commit
	rye show
	rye sync --no-lock
	pre-commit install --install-hooks

.PHONY: format  # Format the code
format:
	rye format
	rye lint --fix -- --fix-only

.PHONY: lint  # Lint the code
lint:
	rye lint
	rye format --check

.PHONY: typecheck  # Run static type checking
typecheck:
	rye run pyright

.PHONY: test  # Run tests and collect coverage data
test:
	rye run coverage run -m pytest

.PHONY: testcov  # Run tests and generate a coverage report
testcov: test
	@echo "building coverage html"
	@rye run coverage html --show-contexts

.PHONY: all
all: format lint typecheck test
