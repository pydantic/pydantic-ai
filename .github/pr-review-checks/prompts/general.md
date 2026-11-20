## This check

Apply the classes above to every listed file. Also check each boundary in `THREAT_MODEL.md` "Core boundaries" and "Harness and CLI execution boundaries" that a listed file crosses.

For files under `.github/workflows/` and `.github/actions/`, hold each job to `THREAT_MODEL.md` "Repository CI and agentic workflows". Flag a change that lets code or text from a pull request act with a secret, a write token or a model tool that the job holds.
