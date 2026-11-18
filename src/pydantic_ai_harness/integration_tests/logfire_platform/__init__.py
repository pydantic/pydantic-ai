"""Live Agent Control tests against a running Logfire platform. See `README.md`.

A package rather than a bare directory because these tests share an agent, a platform client, and a
Logfire configuration, and relative imports keep those private to the suite. The directory is named
`logfire_platform` and not `logfire` for the same reason `tests/logfire_variables` is: a
`logfire/` directory on the import path would shadow the third-party `logfire` package that every
module here imports.
"""
