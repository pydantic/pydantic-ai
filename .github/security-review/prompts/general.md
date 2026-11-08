## This check

Apply the classes above to every listed file. Also check each boundary in `THREAT_MODEL.md` "Core boundaries" and "Harness and CLI execution boundaries" that a listed file crosses.

For files under `.github/workflows/` and `.github/actions/`:

- Flag a `pull_request_target` or `workflow_run` job that executes pull-request content while it holds a secret or a write token.
- Flag a job that gives pull-request content to a model that holds a write tool, a shell, or a token that can write.
- Flag an attacker-controlled event field, such as a title, body or branch name, expanded with `${{ }}` inside a `run:` script.
