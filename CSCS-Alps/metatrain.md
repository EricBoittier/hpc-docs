# Using `metatrain` on Alps

Last update 2026-11-10.

> [!warning]
> Efficient distributed training on Alps requires [NCCL](https://docs.cscs.ch/software/communication/nccl/) over the high-speed Slingshot 11 network. Therefore, careful setup is required.

## On top of uenv

uenv are user environment providing applications and libraries on Alps. `metatrain` can be built on top of the `pytorch` uenv. To list the available `pytorch` uenv you can use

```bash
uenv image find pytorch
```

> [!note]
> As of 2026-11, `pytorch/v2.6.0`, `v2.8.0` and `v2.9.1` are deployed on both `clariden` and `daint`. The examples below use `v2.9.1:v2`; check `uenv image find pytorch` for the current list. A plain venv on the shared filesystem scales badly to many ranks; see the CSCS guide on squashing Python virtual environments with uenv.

Load the `pytorch` uenv, create a virtual environment, and install `metatrain`:
```bash
uenv start --view=default pytorch/v2.9.1:v2        # Load the PyTorch uenv
python -m venv --system-site-packages ./myenv    # Create a venv
. myenv/bin/activate                             # Activate venv
pip install metatrain
```

If a development version is needed, one can simply use `pip install -e .` in the root directory of the `metatrain` repository.

NCCL requires specific environment variables on Alps, to ensure that the high-speed network is used. See the [CSCS NCCL documentation](https://docs.cscs.ch/software/communication/nccl/) for details. The following is an example of `sbatch` script for running `metatrain` on top of the PyTorch uenv:

```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=<NODES>
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=<ACCOUNT>
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -x
cat "$0"

export MASTER_PORT=25678
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=$SLURM_NTASKS

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_HOME="/dev/shm/"
export OMP_NUM_THREADS=64

# Disable JIT caching to distributed filesystem
export CUDA_CACHE_DISABLE=1

# Disable GPU support in MPICH which can lead to deadlocks with NCCL
export MPICH_GPU_SUPPORT_ENABLED=0

# Setup NCCL to use the libfabric plugin and GPU direct
export NCCL_NET="AWS Libfabric"
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
# libfabric-related variables for best performance on Alps
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
export FI_MR_CACHE_MONITOR=userfaultfd
# NCCL >= 2.27 enables the LL128 protocol by default, which performs worse on Slingshot
export NCCL_PROTO=^LL128

srun -c ${SLURM_CPUS_PER_TASK} --cpu-bind=socket bash -c "
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID
. ./myenv/bin/activate
mtt train ${PWD}/options.yaml
"
```

The `RANK` and `LOCAL_RANK` variables are set for each rank within the bash script run with `srun`.
The combination of `--ntasks-per-node=4` and `--gpus-per-task=1` ensures that `CUDA_VISIBLE_DEVICES` is set appropriately for each rank: each of the 4 ranks sees only one of the 4 GPUs available on one node.

> [!warning]
> Always remember to activate the uenv when running. Most environment variables are required in order to obtain good scaling and good performance.

## On top of a container

> [!note]
> `tox` is often used for the tests. This makes it difficult to tap into `pytorch` installed in the NVidia NGC PyTorch container. For this reason, in CI, a base CUDA container is used instead and the following index is added to `pip` for ARM64 + CUDA wheels: `https://download.pytorch.org/whl/cu128`.

The container needs to be defined. Here we use the Nvidia NGC PyTorch container as base, so that we don't have to install PyTorch and other dependencies:
```Dockerfile
FROM docker://nvcr.io/nvidia/pytorch:25.04-py3
RUN python3 -m pip install metatrain
```
To build the container, one can use `podman`:
```
podman build -f <PATH_TO_DOCKERFILE> -t <TAG>
```
Once the container is built, one needs to use `enroot` to create a SquashFS file for the [Container Engine](https://docs.cscs.ch/software/container-engine/)
```bash
enroot import -x mount -o <SQUASHFS_IMAGE_NAME>.sqsh podman://<TAG>
```
To use the container on Alps, an environment definition file (EDF) is required. An example is given below:
```toml
image = "/path/to/<SQUASHFS_IMAGE_NAME>.sqsh"
mounts = ["/capstor:/capstor", "/iopsstor:/iopsstor"]
workdir = <WORKDIR>

[annotations]
com.hooks.aws_ofi_nccl.enabled = "true"
com.hooks.aws_ofi_nccl.variant = "cuda-dl"

[env]
# NCCL_DEBUG = "INFO"
CUDA_CACHE_DISABLE = "1"
TORCH_NCCL_ASYNC_ERROR_HANDLING = "1"
MPICH_GPU_SUPPORT_ENABLED = "0"
```
The `aws_ofi_nccl` annotation enables the injection of the AWS libfabric plugin in the container, thus enabling NCCL to use the high-speed network. This also sets libfabric-related environment variables to obtain good performance on Alps.
> [!note]
> Many environment variables are set by the container engine in order to obtain best performance with NCCL. With uenv, however, this is not the case and one has to set such environment variables manually.

The following submission script allows to run `metatrain` installed within the container:
```bash
#!/bin/bash
#SBATCH --job-name=test
#SBATCH --nodes=<NODES>
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --account=<ACCOUNT>

set -x
cat "$0"

export MASTER_PORT=25678
export MASTER_ADDR=$(hostname)
export WORLD_SIZE=$SLURM_NTASKS
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_HOME="/dev/shm/"

export OMP_NUM_THREADS=64

export CUDA_CACHE_DISABLE=1
export MPICH_GPU_SUPPORT_ENABLED=0

env="/path/to/<EDF_FILE>.toml"
srun --environment=${env} -c ${SLURM_CPUS_PER_TASK} --cpu-bind=socket bash -c "
export RANK=\$SLURM_PROCID
export LOCAL_RANK=\$SLURM_LOCALID
mtt train ${PWD}/options.yaml
"
```
The EDF file needs to be passed directly to `srun`, not as a `#SBATCH` directive.
