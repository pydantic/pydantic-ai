# The sandbox image the live tier claims: sandboxd from agent-sandbox's staging image, in a Debian
# userland with `sh`, coreutils, findutils and Python. agent-sandbox doesn't publish sandboxd to
# `registry.k8s.io`; its staging registry rebuilds `latest-main` from every commit to `main`.
ARG SANDBOXD_IMAGE=us-central1-docker.pkg.dev/k8s-staging-images/agent-sandbox/sandboxd:latest-main
FROM ${SANDBOXD_IMAGE} AS sandboxd

FROM python:3.12-slim-bookworm
COPY --from=sandboxd /usr/local/bin/sandboxd /usr/local/bin/sandboxd
RUN mkdir /workspace && chown 1000:1000 /workspace
USER 1000
ENV HOME=/workspace
WORKDIR /workspace
# Ports and root are set by each `SandboxTemplate`, so one image serves every pool.
ENTRYPOINT ["/usr/local/bin/sandboxd"]
