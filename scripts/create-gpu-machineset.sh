#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export KUBECONFIG="${KUBECONFIG:-${repo_dir}/cluster/auth/kubeconfig}"
gpu_instance_type="${GPU_INSTANCE_TYPE:-g5.xlarge}"
gpu_replicas="${GPU_REPLICAS:-0}"

if ! [[ "$gpu_replicas" =~ ^[0-9]+$ ]]; then
  echo "GPU_REPLICAS must be a non-negative integer" >&2
  exit 2
fi

source_machineset=$(oc -n openshift-machine-api get machineset -o json \
  | jq -r '.items | map(select(.spec.template.spec.providerSpec.value.instanceType | startswith("g") | not)) | .[0].metadata.name')

if [[ -z "$source_machineset" || "$source_machineset" == "null" ]]; then
  echo "No worker MachineSet was found" >&2
  exit 1
fi

availability_zone=$(oc -n openshift-machine-api get machineset "$source_machineset" -o jsonpath='{.spec.template.spec.providerSpec.value.placement.availabilityZone}')
infra_id=$(oc get infrastructure cluster -o jsonpath='{.status.infrastructureName}')
gpu_machineset="${infra_id}-gpu-${availability_zone}"

oc -n openshift-machine-api get machineset "$source_machineset" -o json \
  | jq \
      --arg name "$gpu_machineset" \
      --arg instance_type "$gpu_instance_type" \
      --argjson replicas "$gpu_replicas" '
      del(
        .metadata.annotations,
        .metadata.creationTimestamp,
        .metadata.generation,
        .metadata.managedFields,
        .metadata.resourceVersion,
        .metadata.uid,
        .status
      )
      | .metadata.name = $name
      | .spec.replicas = $replicas
      | .spec.selector.matchLabels["machine.openshift.io/cluster-api-machineset"] = $name
      | .spec.template.metadata.labels["machine.openshift.io/cluster-api-machineset"] = $name
      | .spec.template.metadata.labels["node-role.kubernetes.io/gpu"] = ""
      | .spec.template.spec.providerSpec.value.instanceType = $instance_type
      | .spec.template.spec.providerSpec.value.tags = ((.spec.template.spec.providerSpec.value.tags // []) + [{"name":"openshift-ai-node","value":"gpu"}] | unique_by(.name))
    ' \
  | oc apply -f -

echo "GPU MachineSet $gpu_machineset is configured with replicas=$gpu_replicas and instanceType=$gpu_instance_type"

