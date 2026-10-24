# CLAI 2 web UI development

The Python server ships the committed bundle in
`../pydantic_clai2/builtin_plugins/web/static`. Node is a build dependency.
Use the Node version in `.nvmrc` when rebuilding (`nvm use` or `fnm use`).

From this directory:

```bash
npm ci --ignore-scripts --no-audit --no-fund
npm run build
```

`build` typechecks before bundling. Commit generated assets with source and
lockfile changes, including dependency updates. CI rejects a rebuilt bundle
that differs from the committed assets. Dependabot monitors frontend advisories
and opens security updates; live advisory data does not gate builds or releases.

From the repository root, install Chromium and run the real-browser regressions:

```bash
uv sync --locked --package pydantic-clai2 --group dev --group clai2-browser
uv run --no-sync playwright install --with-deps chromium
CLAI2_BROWSER_TESTS=1 uv run --no-sync pytest tests/clai2/browser
```

The browser suite serves the actual committed bundle through the Python app.
It uses local test models and shares its server and Chromium across the cases.
Other test jobs leave this suite disabled. The dedicated job also requires full
line and branch coverage of the request `Guard`; repository-wide coverage scope
is unchanged.

CI checks installed wheel and sdist assets against the validated bundle.
`clai2-bleeding` runs the same web validation before publishing its checked-out
commit and checks the installed sdists again before replacing release assets.
