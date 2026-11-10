# Using the metatensor ecosystem on SCITAS (EPFL) clusters


Last update 2026-11-10.

SCITAS runs several clusters. This page covers:

- **kuma**: the GPU cluster, with NVIDIA H100 and L40S nodes, for training and MD.
- **jed**: the CPU cluster.

Both use Slurm. Accounts are per lab; use `-A <ACCOUNT>` if you have more than one. Official documentation: <https://scitas-doc.epfl.ch/supercomputers/kuma/> and <https://scitas-doc.epfl.ch/supercomputers/jed/>.

## kuma (GPU)

### Hardware, partitions and QOS

| partition | nodes | per node | max cores per GPU |
|---|---|---|---|
| `h100` | 74 | 4× H100 94 GB, 64 cores (2 sockets), 380 GB RAM | 16 |
| `l40s` | 20 | 8× L40S 48 GB, 64 cores (2 sockets), 380 GB RAM | 8 |
| `mig12gb`, `mig24gb` | 2 and 8 | H100 MIG slices, for debugging and small jobs | 2 and 5 per MIG |

> [!warning]
> **There is no default partition on kuma.** Every `sbatch`, `srun` or `Sinteract` must give `-p h100`, `-p l40s` or a MIG partition.

| QOS | limit |
|---|---|
| `normal` (default) | ≤ 8 nodes, 3 days |
| `long` | ≤ 8 nodes, 7 days |
| `debug` | ≤ 2 nodes, 1 hour, high priority |
| `build` | 1 node, ≤ 16 cores, no GPU, 4 hours |

Request GPUs with `--gres=gpu:N` (or `-g gpu:N` with `Sinteract`).

**Memory.** Slurm gives each core 5900 MB, and you cannot ask for more than that per core, even with `--mem`. Memory is enforced (`ConstrainRAMSpace=yes`): a job that goes over is killed. To get more RAM, ask for more cores.

### Software environment

The modules come from the SCITAS Spack stack. The login nodes default to the H100 stack; switch to the L40S stack with `slmodules -e kuma-l40s` (listed by `slmodules -h`; not tested here). You need `gcc` loaded before `cuda` appears:

```bash
module load gcc cuda      # gcc/13.2.0 + cuda/12.4.1 on the h100 stack
```

The module system has `py-torch/2.3.1`. That is the minimum torch metatensor-torch supports, and the rest of the stack is not provided as modules. The simplest route is a self-contained conda + pip environment. The recipe below has been tested with micromamba. Keep the environment on `/work` (your group's space) or `/scratch`, not in `$HOME`, which has a small quota:

```bash
# once, on a login node (compute nodes may have no internet access)
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$HOME" bin/micromamba
export MAMBA_ROOT_PREFIX=/work/<group>/$USER/micromamba   # or /scratch/$USER/micromamba
eval "$($HOME/bin/micromamba shell hook --shell bash)"

micromamba create -y -n mta -c conda-forge python=3.12 pip
micromamba activate mta
export PIP_CACHE_DIR=$MAMBA_ROOT_PREFIX/pip-cache          # keep the pip cache off $HOME

# The cu126 index pins a CUDA version the kuma driver supports (driver 560, CUDA <= 12.6)
pip install --extra-index-url https://download.pytorch.org/whl/cu126 torch==2.12.1
pip install metatrain==2026.4.1 metatomic-torch==0.1.19 "vesin[torch]" ipi ase
```

The tested combination was torch 2.12.1+cu126, metatrain 2026.4.1, metatomic-torch 0.1.19 and vesin 0.6.1, with a development branch of i-PI. With the release `ipi` (3.3.1), which ships the same `metatomic` PES, it is expected but not yet tested. The binary wheels of metatensor-torch and metatomic-torch work with this torch, so no source build is needed.

In job scripts, activate the same environment:

```bash
export MAMBA_ROOT_PREFIX=/work/<group>/$USER/micromamba
eval "$($HOME/bin/micromamba shell hook --shell bash)"
micromamba activate mta
```

### Running MD with i-PI and a metatomic model

With i-PI, a metatomic model runs *inside* the i-PI process through `ffdirect`, so no separate driver is needed. Set `device:cuda`: the default is `cpu`.

```xml
<ffdirect name='mta' pbc='false'>
  <pes> metatomic </pes>
  <parameters> {template:init.extxyz, model:model.pt, device:cuda} </parameters>
</ffdirect>
```

Batch job on one H100:

```bash
#!/bin/bash
#SBATCH --job-name=ipi-mta
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

export MAMBA_ROOT_PREFIX=/work/<group>/$USER/micromamba
eval "$($HOME/bin/micromamba shell hook --shell bash)"
micromamba activate mta
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

i-pi input.xml > ipi.log 2>&1
```

Several small simulations can share one GPU. For example, four 288-atom boxes in one job:

1. Request `--cpus-per-task=16`.
2. Set `OMP_NUM_THREADS=4`.
3. Launch four `(cd run_i && i-pi input.xml > ipi.log 2>&1) &`.
4. End the script with `wait`.

For small systems the GPU is mostly idle between kernels, so this costs little.

### Interactive sessions

```bash
Sinteract -p h100 -q debug -g gpu:1 -c 8 -m 32G -t 00:30:00
```

> [!warning]
> `Sinteract` passes **`--mem=4G` by default** unless you give `-m`, and kuma enforces it. An i-PI + PET-MAD (torch) session peaked at 2.0–2.5 GB, but the JAX variant peaked at 4.3 GB, and larger runs and analysis scripts (4.5–4.9 GB) were OOM-killed at 4G. Always pass `-m` (≤ 5900 MB × cores; `-m 32G` with `-c 8` is accepted by the `debug` QOS). Batch jobs are not affected: they get 5900 MB per core.

### Practical notes

- **Login nodes.** `kuma.hpc.epfl.ch` round-robins between two login nodes, so after a dropped connection `tmux attach` may land on the wrong one. Note `hostname` and reconnect to that node.
- **Harmless warnings.** `hwloc_set_cpubind() failed` in job logs is harmless. So are metatomic deprecation warnings about `quantity`/`per_atom` or the `non_conservative_forces` output name.
- **Hugging Face downloads.** Download models (e.g. PET-MAD via `upet`) on the login node. In jobs, set `HF_HUB_OFFLINE=1` and point `HF_HOME` at a shared directory.
- **Caches.** Point `XDG_CACHE_HOME` and `TRITON_CACHE_DIR` away from `$HOME` if you compile or JIT a lot.

## jed (CPU)

> [!note]
> The facts below were checked on the cluster, but the metatensor stack itself has **not yet been tested on jed**. The kuma recipe without `--extra-index-url` (CPU torch wheels: `--index-url https://download.pytorch.org/whl/cpu`) is the expected starting point. Contributions welcome.

| partition | nodes | per node | memory per core (default and max) |
|---|---|---|---|
| `standard` (default) | 258 | 72 cores (2 sockets), ~500 GB | 7000 MB |
| `bigmem` | 42 | 72 cores | 14000 MB |
| `hugemem` | 2 | 72 cores | 28000 MB |
| `academic` | 92 | students only (`--partition=academic`) | 7000 MB |

| QOS | limit |
|---|---|
| `serial` (default) | 1 node (72 cores), 7 days |
| `parallel` | > 1 node, 15 days |
| `debug` | 2 hours |
| `free` | 72 cores, 6 hours, lowest priority (not attached to every account: check `sacctmgr show assoc user=$USER format=Account,QOS`) |

Useful modules: `gcc/13.2.0`, `openmpi/5.0.3`, `python/3.11.7` (`module load gcc python`), and the Intel oneAPI compilers, MKL and MPI. jed is a good fit for generating CPU reference data and for CPU-only MD. A full-node MPI job uses `--nodes=1 --ntasks-per-node=72` and `srun --cpu-bind=cores`.
