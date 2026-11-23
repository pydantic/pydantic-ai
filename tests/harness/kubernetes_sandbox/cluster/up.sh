#!/usr/bin/env bash
# Creates, or reuses, the kind cluster `test_kubernetes_live.py` runs against: the agent-sandbox
# controller from the newest release, sandboxd from agent-sandbox's newest staging build, and the
# warm pools in `pools.yaml`. `AGENT_SANDBOX_VERSION` (a release tag) and `SANDBOXD_IMAGE` pin
# either one to reproduce an older run. Its credentials go only to the file `KUBECONFIG` names, so
# the current kubectl context is untouched. Delete the cluster with
# `kind delete cluster --name pydantic-ai-live`.
set -euo pipefail

: "${KUBECONFIG:?set KUBECONFIG to the file this script writes the cluster credentials to}"
export KUBECONFIG
mkdir -p "$(dirname "$KUBECONFIG")"
here="$(cd "$(dirname "$0")" && pwd)"
staging=us-central1-docker.pkg.dev/k8s-staging-images/agent-sandbox/sandboxd
sandboxd_image="${SANDBOXD_IMAGE:-${staging}:latest-main}"
releases=https://github.com/kubernetes-sigs/agent-sandbox/releases
# `releases/latest` redirects to the newest release's tag page.
version="${AGENT_SANDBOX_VERSION:-$(curl -fsS -o /dev/null -w '%{redirect_url}' "$releases/latest")}"
version="${version##*/}"
manifests="$releases/download/${version}/sandbox-with-extensions.yaml"
name=pydantic-ai-live
image=pydantic-ai-live/sandboxd:live

if kind get clusters | grep -qx "$name"; then
  kind export kubeconfig --name "$name" --kubeconfig "$KUBECONFIG"
else
  kind create cluster --name "$name" --kubeconfig "$KUBECONFIG" --wait 120s
fi

# `--pull` so a reused cluster still gets today's `latest-main`, not the copy Docker cached.
docker build --quiet --pull --tag "$image" --build-arg "SANDBOXD_IMAGE=${sandboxd_image}" \
  --file "$here/sandboxd.Dockerfile" "$here"
kind load docker-image --name "$name" "$image"

# A moving tag says nothing in a failed run's log; the digest and release name what ran.
echo "sandboxd: $(docker buildx imagetools inspect "$sandboxd_image" --format '{{.Name}}@{{.Manifest.Digest}}')"
echo "controller: ${version}"
kubectl apply --server-side -f "$manifests"
kubectl -n agent-sandbox-system rollout status deploy --timeout=180s
kubectl wait --for=condition=Established crd --all --timeout=60s

sed "s|SANDBOXD_IMAGE|${image}|" "$here/pools.yaml" | kubectl apply -f -
kubectl wait swp/python-pool --for=jsonpath='{.status.readyReplicas}'=2 --timeout=180s
kubectl wait swp/sandboxd-other-ports --for=jsonpath='{.status.readyReplicas}'=1 --timeout=180s
kubectl wait swp/sandboxd-token --for=jsonpath='{.status.readyReplicas}'=1 --timeout=180s
