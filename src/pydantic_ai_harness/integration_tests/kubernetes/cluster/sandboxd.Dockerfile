# The sandbox image the cluster tests claim: sandboxd built from an agent-sandbox release tag, in a
# Debian userland with `sh`, coreutils, findutils and Python. agent-sandbox doesn't publish sandboxd
# to `registry.k8s.io`, so it is built here.
FROM golang:1.26-bookworm AS sandboxd
ARG AGENT_SANDBOX_VERSION
RUN CGO_ENABLED=0 GOTOOLCHAIN=auto GOBIN=/out \
    go install "sigs.k8s.io/agent-sandbox/packages/sandboxd/cmd/sandboxd@${AGENT_SANDBOX_VERSION}"

FROM python:3.12-slim-bookworm
COPY --from=sandboxd /out/sandboxd /usr/local/bin/sandboxd
RUN mkdir /workspace && chown 1000:1000 /workspace
USER 1000
ENV HOME=/workspace
WORKDIR /workspace
# Ports and root are set by each `SandboxTemplate`, so one image serves every pool.
ENTRYPOINT ["/usr/local/bin/sandboxd"]
