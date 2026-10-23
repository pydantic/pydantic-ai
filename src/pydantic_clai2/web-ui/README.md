# CLAI 2 web UI development

The Python server ships the committed bundle in
`../pydantic_clai2/builtin_plugins/web/static`. Node is a build dependency.
Use the Node version in `.github/workflows/clai2-web.yml` when rebuilding.

From this directory:

```bash
npm ci --ignore-scripts --no-audit --no-fund
npm audit --audit-level=low
npm run build
```

`build` typechecks before bundling. Commit generated assets with source and
lockfile changes, including dependency updates. CI rejects a rebuilt bundle
that differs from the committed assets and audits development dependencies too.

From the repository root, install Chromium and run the real-browser regressions:

```bash
uv run --with playwright==1.63.0 playwright install --with-deps chromium
CLAI2_BROWSER_TESTS=1 uv run --with playwright==1.63.0 pytest tests/clai2/browser
```

The browser suite serves the actual committed bundle through the Python app.
It uses local test models and shares its server and Chromium across the cases.
Other test jobs leave this suite disabled. The dedicated job also requires full
line and branch coverage of the request `Guard`; repository-wide coverage scope
is unchanged.

CI checks installed wheel and sdist assets against the validated bundle.
`clai2-bleeding` runs the same web validation before publishing its checked-out
commit and checks the installed sdists again before replacing release assets.
