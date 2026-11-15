# CLAI 2 web UI development

The Python server serves the bundle in
`../pydantic_clai2/builtin_plugins/web/static`. Node is a build dependency.
Use the Node version in `.nvmrc` when rebuilding (`nvm use` or `fnm use`).

From this directory:

```bash
npm ci --ignore-scripts --no-audit --no-fund
npm run build
```

`build` typechecks before bundling and writes the bundle there, so `clai2 --web`
from the checkout serves your changes.

Do not commit the rebuilt bundle: a pull request changes only the sources and
lockfile, so two web UI pull requests never conflict on the minified output.
Restore it before committing:

```bash
git restore ../pydantic_clai2/builtin_plugins/web/static
```

CI builds the bundle from source for the browser tests, the packages, and
releases, and fails a pull request that commits it alongside other changes.
After a web UI change lands on `main`, the `CLAI 2 web bundle` workflow
commits the rebuilt bundle through a pull request of its own, so installs from
a Git ref or commit archive get a working web UI without Node.

After `make install`, run from the repository root:

```bash
uv run --group clai2-browser playwright install --with-deps chromium
CLAI2_BROWSER_TESTS=1 uv run --group clai2-browser pytest tests/clai2/browser
```

The browser suite serves the bundle you built through the Python app.
It uses local test models and shares its server and Chromium across the cases.
