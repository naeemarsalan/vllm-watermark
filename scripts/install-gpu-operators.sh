#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export KUBECONFIG="${KUBECONFIG:-${repo_dir}/cluster/auth/kubeconfig}"

wait_for_csv() {
  local namespace=$1
  local subscription=$2
  local csv=""

  for _ in $(seq 1 60); do
    csv=$(oc -n "$namespace" get subscription "$subscription" -o jsonpath='{.status.installedCSV}' 2>/dev/null || true)
    [[ -n "$csv" ]] && break
    sleep 5
  done
  if [[ -z "$csv" ]]; then
    echo "Timed out waiting for $subscription to select a CSV" >&2
    return 1
  fi

  oc -n "$namespace" wait --for=jsonpath='{.status.phase}'=Succeeded "csv/$csv" --timeout=10m >&2
  printf '%s' "$csv"
}

oc apply -f "${repo_dir}/gpu/nfd-operator.yaml"
nfd_csv=$(wait_for_csv openshift-nfd nfd)
oc apply -f "${repo_dir}/gpu/nfd-instance.yaml"
echo "Installed $nfd_csv and created NodeFeatureDiscovery/nfd-instance"

oc apply -f "${repo_dir}/gpu/nvidia-gpu-operator.yaml"
gpu_csv=$(wait_for_csv nvidia-gpu-operator gpu-operator-certified)

oc -n nvidia-gpu-operator get "csv/$gpu_csv" -o jsonpath='{.metadata.annotations.alm-examples}' \
  | jq -e 'map(select(.kind == "ClusterPolicy")) | .[0]' \
  | oc apply -f -

echo "Installed $gpu_csv and created the catalog-provided ClusterPolicy"
