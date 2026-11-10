# Running molecular dynamics simulations on Alps with `metatomic` and `i-pi`

Last update 2026-11-10.

Containers are powerful tools that give you almost full control over the environment of your application and ensure the reproducibility of the environment.

CSCS is equipped with [the Container Engine (CE) toolset](https://docs.cscs.ch/software/container-engine/). This toolset is tightly integrated into the Slurm workload manager, allowing running the tasks inside the containers with little extra effort.

Container images are generally available online. Specifically for CSCS, there are [images](https://docs.cscs.ch/software/alps-extended-images/) modified to fully leverage the ability of `daint.alps` based on the [NGC containers](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch?version=26.02-py3). Other containers should work too, but you should be careful about the network stack. For example, the NGC containers mentioned above uses OpenMPI 4, which is incompatible with CSCS network stack, and leads to some complexities when MPI is needed.

## Selecting and Building a Base Container

To start, first pick a version of the container at this [Alps Extended Images](https://docs.cscs.ch/software/alps-extended-images/) page and copy the image URL. Then, create a new folder, say `MD-container` for the following steps.

```bash
mkdir MD-container
cd MD-container
```

> [!note]
> Building the image consumes lots of computational resources, so it is recommended to do the following steps on an interactive computational node. To do so, use this command `srun -A <ACCOUNT> -p debug -t 01:00:00 --pty bash`.

Inside `MD-container`, create a file `Containerfile`. Later on, we will create a custom container base on the container that you chose and the custom commands written in the `Containerfile`. The `Containerfile` for now looks like:

```Dockerfile
FROM <image_path:you_chose>
```

Before building the container, set the container storage to `/dev/shm` by creating `~/.config/containers/storage.conf` (or `$XDG_CONFIG_HOME/containers/storage.conf` if `XDG_CONFIG_HOME` is set) with the content below. `/dev/shm` is wiped at the end of the allocation, so run `enroot import` in the same allocation as the build.
```bash
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
```

Now we can choose a name for the container and build it:

```bash
IMAGE_NAME=<image_name:you_like>
podman build -t ${IMAGE_NAME} .
```

At this stage, the container is identical to the upstream image. You can enter this container with `srun`, but we need to:

1. save the container image;
2. write down the configuration of this container for Container Engine.

To save:

```bash
SQSH_IMAGE_FILE=/path/to/container.sqsh
enroot import -x mount  -o ${SQSH_IMAGE_FILE} "podman://${IMAGE_NAME}"
```

To configure, we need to write the following contents in to `$HOME/.edf/<container_name>.toml`

```toml
image = "/path/to/container.sqsh"
# Make your folders accessible inside the container, format: "/path/on/host:/path/in/container"
mounts = ["${SCRATCH}:${SCRATCH}", "/your/workdir:/your/workdir"]
workdir = "/your/workdir"

[annotations]
com.hooks.cxi.enabled = "false"
```

> [!note]
> For a detailed explanation of the EDF file, please refer to [this page](https://docs.cscs.ch/software/container-engine/edf/#mounts).

Then, enter the container with

```bash
srun --environment=<container_name> --pty bash
```

This opens an interactive node and loads the container specified in `$HOME/.edf/<container_name>.toml` for you.

> [!warning]
> This only helps you get familiar with the container, and after you exit this container, all the modifications that you have done would be permanently lost.

## Making Persistent Modifications

After entering the container, you can play with it and configure your simulation environment as usual. However, be sure that every command is recorded, because your modification to this loaded container is *temporary and will be lost* after you exit.

To do permanent modifications, you should write down the commands in the `Containerfile`. The way to do so is to add them with the `RUN` keyword:

```Dockerfile
FROM <image_path:you_chose>

RUN command_0 \
    && command_1 \
    && ...
```

It is also possible to setup the environmental variables in the `Containerfile` too, with:

```Dockerfile
ENV VARIABLE_NAME=value
```

Rebuild the container:

```bash
podman build -t ${IMAGE_NAME} .
```

You can save the modified container, but you need to first remove the previous image file:

```bash
rm ${SQSH_IMAGE_FILE}
enroot import -x mount  -o ${SQSH_IMAGE_FILE} "podman://${IMAGE_NAME}"
```

Next time, you can use the container by first loading it with:

```bash
srun --environment=<container_name> --pty bash
```

## Example Containerfile for `metatomic + i-Pi`

Below is an example `Containerfile`

```Dockerfile
# The image path of the Alps Extended container that you want to use. You can find the latest version here: https://docs.cscs.ch/software/alps-extended-images/
FROM jfrog.svc.cscs.ch/docker-group-csstaff/alps-images/ngc-pytorch:26.02-py3-alps6

# Build the torch extensions from source against the container's PyTorch FIRST:
# if metatrain is installed first, pip pulls binary wheels and the later
# --no-binary installs become no-ops ("Requirement already satisfied").
RUN pip install --no-cache-dir --no-build-isolation \
        --no-binary=metatensor-torch,metatomic-torch,vesin-torch \
        "metatensor[torch]" "metatomic[torch]" "vesin[torch]" \
    && pip install --no-cache-dir \
        ase \
        ipi \
        "metatrain[mace]" \
        sphericart-torch \
        torch-spex \
        torch-ema

WORKDIR /workspace
```

## Example SLURM Submission Script

The following is an example job submission script, in which we launch four tasks, each on a unique GPU.

```bash
#!/bin/bash
#SBATCH -o job.%j.out
#SBATCH -J example_job
#SBATCH --time=12:00:00   # clariden max 12 h; daint allows 24 h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --exclusive

run_job () {
    local system=$1
    # --exclusive + an explicit --mem make the steps run concurrently on distinct GPUs;
    # without them the steps were serialized on clariden (all on GPU 0)
    srun --exclusive --ntasks=1 --gpus-per-task=1 --cpus-per-task=4 --mem=50G \
        --environment=<container_name> bash -lc "
        cd ${SCRATCH}/results
        mkdir -p $system
        cd $system
        i-pi /your/workdir/${system}.xml
    " &
}

mkdir -p ${SCRATCH}/results
run_job water_0
run_job water_1
run_job water_2
run_job water_3

wait

exit 0
```

The i-PI input must evaluate the model in-process with `ffdirect` and the `metatomic` PES. Set `device: cuda`, since the default is `cpu`:

```xml
<ffdirect name='mta' pbc='true'>
  <pes> metatomic </pes>
  <parameters> {template:init.extxyz, model:model.pt, device:cuda} </parameters>
</ffdirect>
```

## Test Cases

For testing if the whole workflow works or not, you can refer to [this recipe](https://atomistic-cookbook.org/examples/water-model/water-model.html#path-integral-molecular-dynamics-with-i-pi) from atomistic cookbook.
