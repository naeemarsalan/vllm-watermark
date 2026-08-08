# Cluster environment

## OpenShift on AWS with optional GPU capacity

This workspace provisions an OpenShift 4.20 cluster named `ocp-ai` in `us-east-1`
under `sandbox1392.opentlc.com`. Generated installer assets and credentials are
excluded from Git (`aws`, `cluster/` — see `.gitignore`; never commit them).

The initial cluster uses three `m6i.xlarge` control-plane nodes and three
`m6i.xlarge` workers. A separate `g5.xlarge` GPU MachineSet is created at zero
replicas because the AWS account has a four-vCPU on-demand G-instance quota.

## GPU node

Add the GPU node when needed:

```bash
./scripts/scale-gpu.sh 1
```

Remove the billable GPU node when it is no longer needed:

```bash
./scripts/scale-gpu.sh 0
```

OpenShift AI also requires Node Feature Discovery and the NVIDIA GPU Operator
before workloads can consume `nvidia.com/gpu` resources. They are installed by
`scripts/install-gpu-operators.sh`, together with the required
`NodeFeatureDiscovery` and catalog-provided `ClusterPolicy` resources.

## Access

- `KUBECONFIG=cluster/auth/kubeconfig` (gitignored, local only)
- The GPU node is billable — **scale it to 0 whenever it is not actively in use.**
- The sandbox account is monitored and charged back; leaked credentials mean
  environment deletion. The `aws` credentials file is gitignored — keep it that way.
