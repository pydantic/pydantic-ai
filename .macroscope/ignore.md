# Repository-wide Macroscope ignore (code review + any check-run agents).
#
# One glob per line; `#` starts a comment; `**` spans directories. Files listed
# here are dropped from the billed review diff, so Macroscope does not spend
# credits reviewing generated, recorded, or vendored files that no human reads.

# Recorded VCR cassettes: LLM/HTTP interaction recordings, regenerated in bulk
# (~1000 files, ~200 MB), never hand-reviewed.
**/cassettes/**

# Lock files.
**/*.lock
**/package-lock.json

# Compiled gh-aw workflows (generated from their `.md` sources).
.github/workflows/*.lock.yml

# Documentation images (binary, not reviewable as a diff).
docs/img/**
