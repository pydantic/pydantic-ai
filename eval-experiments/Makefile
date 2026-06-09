# easy_evals -- dev tasks, run against the pydantic-ai workspace venv.
#
# The hosted integration's *live* path and the strict typecheck of hosted.py need
# logfire>=4.35 (hosted datasets + managed variables). The workspace lockfile pins
# 4.16, so install a newer logfire into the venv (`make setup`) and run with
# `--no-sync` (otherwise `uv run` resyncs and reverts it). Lint/typecheck use the
# upstream pydantic-ai configs so the gate matches CI.

PROJ := ..
VENV := $(PROJ)/.venv
UV   := uv run --no-sync --project $(PROJ)
GATED := easy_evals fake_models.py demo.py tests conftest.py

.PHONY: setup typecheck lint test demo samples check
setup:  ## install logfire>=4.35 into the venv (local only; does not touch the lockfile)
	uv pip install --python $(VENV) --no-deps 'logfire>=4.35'

typecheck:  ## pyright strict, upstream flags (pyrightconfig.json)
	$(UV) pyright $(GATED)

lint:  ## ruff with the upstream pydantic-ai config (matches CI rules)
	$(UV) ruff check --config $(PROJ)/pyproject.toml $(GATED)

test:
	$(UV) python -m pytest . -q

demo:
	PYTHONPATH=$(CURDIR) $(UV) python demo.py

samples:  ## run the offline-runnable before/after samples
	@for f in samples/*_today.py samples/*_easy.py; do \
		case $$f in *pytest*|*hosted_today*) continue;; esac; \
		echo "--- $$f ---"; PYTHONPATH=$(CURDIR) $(UV) python $$f >/dev/null && echo ok || echo FAIL; \
	done

check: lint typecheck test  ## the full gate
