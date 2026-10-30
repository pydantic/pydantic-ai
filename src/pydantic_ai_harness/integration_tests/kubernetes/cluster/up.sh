#!/usr/bin/env bash
# Creates, or reuses, the kind cluster `test_live_cluster.py` runs against: the agent-sandbox
# controller from a release, a sandboxd image built from the same tag, and the warm pools in
# `pools.yaml`. Its credentials go only to the file `KUBECONFIG` names, so the current kubectl
# context is untouched. Delete the cluster with `kind delete cluster --name pydantic-ai-live`.
set -euo pipefail

: "${KUBECONFIG:?set KUBECONFIG to the file this script writes the cluster credentials to}"
export KUBECONFIG
mkdir -p "$(dirname "$KUBECONFIG")"
here="$(cd "$(dirname "$0")" && pwd)"
version="${AGENT_SANDBOX_VERSION:-v1.0.7}"
name=pydantic-ai-live
image="pydantic-ai-live/sandboxd:${version}"

if kind get clusters | grep -qx "$name"; then
  kind export kubeconfig --name "$name" --kubeconfig "$KUBECONFIG"
else
  kind create cluster --name "$name" --kubeconfig "$KUBECONFIG" --wait 120s
fi

docker build --quiet --tag "$image" --build-arg "AGENT_SANDBOX_VERSION=${version}" \
  --file "$here/sandboxd.Dockerfile" "$here"
kind load docker-image --name "$name" "$image"

kubectl apply --server-side \
  -f "https://github.com/kubernetes-sigs/agent-sandbox/releases/download/${version}/sandbox-with-extensions.yaml"
kubectl -n agent-sandbox-system rollout status deploy --timeout=180s
kubectl wait --for=condition=Established crd --all --timeout=60s

sed "s|SANDBOXD_IMAGE|${image}|" "$here/pools.yaml" | kubectl apply -f -
kubectl wait swp/python-pool --for=jsonpath='{.status.readyReplicas}'=2 --timeout=180s
kubectl wait swp/sandboxd-other-ports --for=jsonpath='{.status.readyReplicas}'=1 --timeout=180s
