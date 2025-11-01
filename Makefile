.DEFAULT_GOAL := all

# Detect OS
ifeq ($(OS),Windows_NT)
	DETECTED_OS := Windows
else
	DETECTED_OS := $(shell uname -s)
endif

.PHONY: .uv
.uv: ## Check that uv is installed
	@uv --version || echo 'Please install uv: https://docs.astral.sh/uv/getting-started/installation/'

.PHONY: .pre-commit
.pre-commit: ## Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install
install: .uv .pre-commit ## Install the package, dependencies, and pre-commit for local development
	uv sync --frozen --all-extras --all-packages --group lint --group docs
	pre-commit install --install-hooks

.PHONY: install-all-python
install-all-python: ## Install and synchronize an interpreter for every python version
ifeq ($(DETECTED_OS),Windows)
	@set UV_PROJECT_ENVIRONMENT=.venv310 & uv sync --python 3.10 --frozen --all-extras --all-packages --group lint --group docs
	@set UV_PROJECT_ENVIRONMENT=.venv311 & uv sync --python 3.11 --frozen --all-extras --all-packages --group lint --group docs
	@set UV_PROJECT_ENVIRONMENT=.venv312 & uv sync --python 3.12 --frozen --all-extras --all-packages --group lint --group docs
	@set UV_PROJECT_ENVIRONMENT=.venv313 & uv sync --python 3.13 --frozen --all-extras --all-packages --group lint --group docs
else
	UV_PROJECT_ENVIRONMENT=.venv310 uv sync --python 3.10 --frozen --all-extras --all-packages --group lint --group docs
	UV_PROJECT_ENVIRONMENT=.venv311 uv sync --python 3.11 --frozen --all-extras --all-packages --group lint --group docs
	UV_PROJECT_ENVIRONMENT=.venv312 uv sync --python 3.12 --frozen --all-extras --all-packages --group lint --group docs
	UV_PROJECT_ENVIRONMENT=.venv313 uv sync --python 3.13 --frozen --all-extras --all-packages --group lint --group docs
endif

.PHONY: sync
sync: .uv ## Update local packages and uv.lock
	uv sync --all-extras --all-packages --group lint --group docs

.PHONY: format
format: ## Format the code
	uv run ruff format
	uv run ruff check --fix --fix-only

.PHONY: lint
lint: ## Lint the code
	uv run ruff format --check
	uv run ruff check

.PHONY: typecheck-pyright
typecheck-pyright:
	@# To typecheck for a specific version of python, run 'make install-all-python' then set environment variable PYRIGHT_PYTHON=3.10 or similar
	@# PYRIGHT_PYTHON_IGNORE_WARNINGS avoids the overhead of making a request to github on every invocation
	PYRIGHT_PYTHON_IGNORE_WARNINGS=1 uv run pyright $(if $(PYRIGHT_PYTHON),--pythonversion $(PYRIGHT_PYTHON))

.PHONY: typecheck-mypy
typecheck-mypy:
	uv run mypy

.PHONY: typecheck
typecheck: typecheck-pyright ## Run static type checking

.PHONY: typecheck-both  ## Run static type checking with both Pyright and Mypy
typecheck-both: typecheck-pyright typecheck-mypy

.PHONY: test
test: ## Run tests and collect coverage data
	@# To test using a specific version of python, run 'make install-all-python' then set environment variable PYTEST_PYTHON=3.10 or similar
	$(if $(PYTEST_PYTHON),UV_PROJECT_ENVIRONMENT=.venv$(subst .,,$(PYTEST_PYTHON))) uv run $(if $(PYTEST_PYTHON),--python $(PYTEST_PYTHON)) coverage run -m pytest -n auto --dist=loadgroup --durations=20
	@uv run coverage combine
	@uv run coverage report

.PHONY: test-all-python
test-all-python: ## Run tests on Python 3.10 to 3.13
ifeq ($(DETECTED_OS),Windows)
	@set UV_PROJECT_ENVIRONMENT=.venv310 & uv run --python 3.10 --all-extras --all-packages coverage run -p -m pytest
	@set UV_PROJECT_ENVIRONMENT=.venv311 & uv run --python 3.11 --all-extras --all-packages coverage run -p -m pytest
	@set UV_PROJECT_ENVIRONMENT=.venv312 & uv run --python 3.12 --all-extras --all-packages coverage run -p -m pytest
	@set UV_PROJECT_ENVIRONMENT=.venv313 & uv run --python 3.13 --all-extras --all-packages coverage run -p -m pytest
else
	UV_PROJECT_ENVIRONMENT=.venv310 uv run --python 3.10 --all-extras --all-packages coverage run -p -m pytest
	UV_PROJECT_ENVIRONMENT=.venv311 uv run --python 3.11 --all-extras --all-packages coverage run -p -m pytest
	UV_PROJECT_ENVIRONMENT=.venv312 uv run --python 3.12 --all-extras --all-packages coverage run -p -m pytest
	UV_PROJECT_ENVIRONMENT=.venv313 uv run --python 3.13 --all-extras --all-packages coverage run -p -m pytest
endif
	@uv run coverage combine
	@uv run coverage report

.PHONY: testcov
testcov: test ## Run tests and generate an HTML coverage report
	@echo "building coverage html"
	@uv run coverage html

.PHONY: update-examples
update-examples: ## Update documentation examples
	uv run -m pytest --update-examples tests/test_examples.py

.PHONY: update-vcr-tests
update-vcr-tests: ## Update tests using VCR that hit LLM APIs; note you'll need to set API keys as appropriate
	uv run -m pytest --record-mode=rewrite tests

# `--no-strict` so you can build the docs without insiders packages
.PHONY: docs
docs: ## Build the documentation
	uv run mkdocs build --no-strict

# `--no-strict` so you can build the docs without insiders packages
.PHONY: docs-serve
docs-serve: ## Build and serve the documentation
	uv run mkdocs serve --no-strict

.PHONY: .docs-insiders-install
.docs-insiders-install: ## Install insiders packages for docs if necessary
ifeq ($(DETECTED_OS),Windows)
	@powershell -NoProfile -Command " \
		$$material = uv pip show mkdocs-material 2>$$null; \
		if ($$material -match 'insiders') { \
			Write-Host 'insiders packages already installed'; \
		} elseif ('$(PPPR_TOKEN)' -eq '') { \
			Write-Host 'Error: PPPR_TOKEN is not set, cannot install insiders packages'; \
			exit 1; \
		} else { \
			Write-Host 'installing insiders packages...'; \
			uv pip install --reinstall --no-deps --extra-index-url https://pydantic:$(PPPR_TOKEN)@pppr.pydantic.dev/simple/ mkdocs-material mkdocstrings-python; \
		} \
	"
else
	ifeq ($(shell uv pip show mkdocs-material | grep -q insiders && echo 'installed'), installed)
		@echo 'insiders packages already installed'
	else ifeq ($(PPPR_TOKEN),)
		@echo "Error: PPPR_TOKEN is not set, can't install insiders packages"
		@exit 1
	else
		@echo 'installing insiders packages...'
		@uv pip install --reinstall --no-deps \
			--extra-index-url https://pydantic:${PPPR_TOKEN}@pppr.pydantic.dev/simple/ \
			mkdocs-material mkdocstrings-python
	endif
endif

.PHONY: docs-insiders
docs-insiders: .docs-insiders-install ## Build the documentation using insiders packages
	uv run --no-sync mkdocs build -f mkdocs.insiders.yml

.PHONY: docs-serve-insiders
docs-serve-insiders: .docs-insiders-install ## Build and serve the documentation using insiders packages
	uv run --no-sync mkdocs serve -f mkdocs.insiders.yml

.PHONY: cf-pages-build
cf-pages-build: ## Install uv, install dependencies and build the docs, used on CloudFlare Pages
	curl -LsSf https://astral.sh/uv/install.sh | sh
	uv python install 3.12
	uv sync --python 3.12 --frozen --group docs
	uv pip install --reinstall --no-deps \
		--extra-index-url https://pydantic:${PPPR_TOKEN}@pppr.pydantic.dev/simple/ \
		mkdocs-material mkdocstrings-python
	uv pip freeze
	uv run --no-sync mkdocs build -f mkdocs.insiders.yml

.PHONY: all
all: format lint typecheck testcov ## Run code formatting, linting, static type checks, and tests with coverage report generation

.PHONY: help
help: ## Show this help (usage: make help)
ifeq ($(DETECTED_OS),Windows)
	@echo Usage: make [recipe]
	@echo Recipes:
	@uv run python -c "import re; [print(f'  {m[0]:<20} {m[1]}') for m in re.findall(r'^([a-zA-Z0-9_-]+):.*?## (.*)$$', open('$(MAKEFILE_LIST)').read(), re.MULTILINE)]" || powershell -NoProfile -Command "Get-Content '$(MAKEFILE_LIST)' | Select-String -Pattern '^([a-zA-Z0-9_-]+):.*?## (.*)$$' | ForEach-Object { Write-Host ('  {0,-20} {1}' -f $$_.Matches[0].Groups[1].Value, $$_.Matches[0].Groups[2].Value) }"
else
	@echo "Usage: make [recipe]"
	@echo "Recipes:"
	@awk '/^[a-zA-Z0-9_-]+:.*?##/ { \
		helpMessage = match($$0, /## (.*)/); \
		if (helpMessage) { \
			recipe = $$1; \
			sub(/:/, "", recipe); \
			printf "  \033[36m%-20s\033[0m %s\n", recipe, substr($$0, RSTART + 3, RLENGTH); \
		} \
	}' $(MAKEFILE_LIST)
endif
