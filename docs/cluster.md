# Cluster environment

## OpenShift on AWS with optional GPU capacity

The repository's provisioning assets target an OpenShift 4.20 cluster named
`ocp-ai`; Phase 0 recorded that cluster and its optional A10G GPU node
(`EXECUTED`; [infrastructure run](../EXPERIMENTS.md#2026-08-08--phase-0-infrastructure-bring-up-ocp-ai-cluster)).
Generated installer assets and credentials are excluded from Git as `aws` and
`cluster/` (`STATIC`; [repository rules](../AGENTS.md#3-secrets-and-safety)).
Do not inspect, print, or commit either location.

## GPU node

Add the GPU node when needed:

```bash
./scripts/scale-gpu.sh 1
```

Remove the billable GPU node when it is no longer needed:

```bash
./scripts/scale-gpu.sh 0
```

The repository provisions Node Feature Discovery and the NVIDIA GPU Operator
through `scripts/install-gpu-operators.sh` (`STATIC`; [provisioning script](../scripts/install-gpu-operators.sh)).
The recorded Phase 0 run observed both operators and a completed CUDA validator
on the GPU node (`EXECUTED`; [infrastructure run](../EXPERIMENTS.md#2026-08-08--phase-0-infrastructure-bring-up-ocp-ai-cluster)).

## Access

- `KUBECONFIG=cluster/auth/kubeconfig` (gitignored, local only)
- The GPU node is billable — **scale it to 0 whenever it is not actively in use.**
- Historical scale-down checks are `EXECUTED` only at their recorded timestamps;
  they are not claims about current live state ([run log](../EXPERIMENTS.md)).
- The `aws` credentials file is gitignored; never print or commit it
  (`STATIC`; [repository rules](../AGENTS.md#3-secrets-and-safety)).
