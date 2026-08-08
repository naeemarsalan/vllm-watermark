#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 REPLICAS" >&2
  exit 2
fi

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export KUBECONFIG="${KUBECONFIG:-${repo_dir}/cluster/auth/kubeconfig}"

mapfile -t gpu_machinesets < <(oc -n openshift-machine-api get machineset -o json \
  | jq -r '.items[] | select(.spec.template.metadata.labels["node-role.kubernetes.io/gpu"] == "") | .metadata.name')

if [[ ${#gpu_machinesets[@]} -ne 1 ]]; then
  echo "Expected exactly one GPU MachineSet; found ${#gpu_machinesets[@]}" >&2
  exit 1
fi

oc -n openshift-machine-api scale machineset "${gpu_machinesets[0]}" --replicas="$1"

