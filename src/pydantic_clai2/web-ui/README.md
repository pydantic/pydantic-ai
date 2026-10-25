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
that differs from the committed assets. When updating dependencies, including on
a Dependabot security PR, check out that branch, run the commands above, and
commit the rebuilt assets before merging.

After `make install`, run from the repository root:

```bash
uv run --group clai2-browser playwright install --with-deps chromium
CLAI2_BROWSER_TESTS=1 uv run --group clai2-browser pytest tests/clai2/browser
```

The browser suite serves the actual committed bundle through the Python app.
It uses local test models and shares its server and Chromium across the cases.
