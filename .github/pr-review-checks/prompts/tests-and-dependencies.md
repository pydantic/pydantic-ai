## This check

Review the changed tests, cassettes, dependency manifests and lockfiles against the repository's testing and dependency rules. This is not a vulnerability review, apart from the supply-chain items below.

Read these rules at the root before you review: `tests/AGENTS.md`, `tests/harness/AGENTS.md` for files under `tests/harness/`, and the testing and type-system sections of `AGENTS.md` and `agent_docs/index.md`. When a changed test covers a package with its own `AGENTS.md`, such as `src/pydantic_ai_harness/AGENTS.md` or `src/pydantic_clai2/AGENTS.md`, read its testing section too.

CI already fails `tests/cost_guards.py` violations, orphan or partly played cassettes, coverage below 100%, and a stale runner script lock. Do not report them.

### `high`: a hard rule broken

- A changed line breaks a rule that the root instruction files state with "must", "never" or "do not". Examples: a new `pytest.mark.anyio`; a subprocess or a real sleep or timeout in a test of logic that can run in-process; a line number in a test docstring or comment; a whole-file `# pyright: ignore`; a rewritten historical compatibility fixture under `tests/clai2/`.
- A new `Any` or `cast` in a test that the code does not need. Under `tests/harness/`, any new `Any` or `cast`.
- A cassette that its own test did not record. A cassette at `tests/<dir>/cassettes/<file_stem>/<test>.yaml` belongs to the test `<test>` in `tests/<dir>/<file_stem>.py`. `<test>` is the function name, or `<Class>.<method>`, followed by `[<param ids>]` for a parametrized test. A cassette shared through `@pytest.mark.vcr('<name>.yaml')`, with a docstring that says why, is allowed.
- A secret in a cassette: an authorization or API-key header that is not a placeholder, an organization, account or project identifier, or an internal hostname. Name the field and never quote the value. A module-level `vcr_config` that replaces the root one, instead of extending it with `{**vcr_config, ...}`, records cassettes without the scrubbing filters.
- A requirement changed in a `pyproject.toml` without the same change in `uv.lock`: the member's `requires-dist`, or the `[manifest]` `constraints` and `overrides`, with the same name, marker and specifier.
- A dependency floor raised with no reason in a comment next to the requirement.
- A dependency change in `src/pydantic_ai_harness/pyproject.toml` when the pull request body in `pr-details.json` links no issue.
- Lockfile content written by hand: conflict markers, a version bump whose artifact URLs, hashes and `upload-time` did not change, or a wheel filename that disagrees with its `version`.

### `medium`: everything else this check covers

- **Coverage.** The changed tests do not exercise the behavior the pull request changes, including both sides of a changed branch and each side of a profile flag. Compare the production change under `pr-head/` with the root. A test whose assertions do not match its name or docstring.
- **Snapshots.** A whole-object `snapshot()` replaced with `len()`, `IsInstance` or field assertions. A snapshot change that contradicts the test's name or the intended change. A recorded fact, such as a resolved model identifier, pinned only in the cassette and not in the test body.
- **Test shape.** A unit test that does not say why it is not a VCR test. A test of a private method where a public API reaches the behavior. Many small single-branch tests of one private entry point that one integration test would cover. A feature test appended to a provider's test file instead of `tests/test_<feature>.py`.
- **Cassettes.** Signs of a copied or hand-edited cassette: a response id or `date` header that another cassette also has, body text changed while ids and dates stayed the same, or a request whose model, prompt, tools or settings do not match the test. A cassette whose response names a different model than its request. A newer dated identifier of the same model is allowed. A request model that differs from the model the test configures. An error response in a test that asserts success. A re-recorded request body that changed with no matching code or test change. A `[param-id]` cassette whose id matches no current parameter.
- **Dependencies.** An upper cap added with no migration reason. A new dependency that nothing under `pr-head/` imports. A new optional extra in `pydantic_ai_slim` that `docs/install.md` does not mention. Lock changes that the manifest changes do not explain, in a pull request that is not a lock bump. A floor raised for a security fix, which belongs in `[tool.uv] constraint-dependencies`.
- **Supply chain.** A package index other than `https://pypi.org/simple`, or an artifact URL outside `files.pythonhosted.org`. A new `git`, `url` or `path` source, unless it pins a full commit SHA of a Pydantic repository. The global `exclude-newer` removed or raised, an `exclude-newer-package` entry set to `false` for a package Pydantic does not publish, or a new cutoff with no reason comment. In an npm lockfile, a `resolved` URL outside `registry.npmjs.org`, a missing `integrity`, or a `git+`, `file:` or tarball specifier. A changed `.github/aw/actions-lock.json` with no workflow change, or a pin that is not a full commit SHA.

### Do not flag

- A lock-only bump within the declared ranges, from PyPI.
- A snapshot or fixture change that follows an intended behavior change.
- Harness tests that use `TestModel` or `FunctionModel` and no cassette.
- A heavy optional import inside a test or fixture.
- What the files cannot show: download counts, yanked releases, or the record mode a cassette used.

Use only `high` and `medium`, as defined above.
