# uenv


Software on Alps is provided via [uenv] (user environments).

## Spack

uenv is based on the [Spack] package manager, 
therefore all the software in an uenv needs to be available in Spack.

If a software is not available in Spack, you need to create a Spack package for it,
following the [Spack packaging guide].

## Recipes

A [uenv recipe] is a collection of YAML files describing the uenv (compilers, libraries, applications, etc.).
An uenv can be built from a recipe using [`stackinator`], or using the [uenv build service].

Uenv recipes for the metatensor ecosystem are available under

```
CSCS-Alps/uenv/
```

### Updating a package version

To update a package version in an existing recipe, you need to do the following:

1. [Optional] Update the `spack: packages: commit` in the `config.yaml` file,
2. Update the version of the package in the `environments.yaml` file.

The first step is only required if the version of interest has been added to Spack
after the version/commit of `spack-packages` specified in the `config.yaml` file.

### Custom Spack packages

Custom Spack packages, overwriting the ones in Spack, can be added to

```
<PATH_TO_RECIPE>/repo/packages/
```

## Examples

### Build uenv with build service

Build an uenv from a recipe using the [uenv build service]:

```bash
git clone https://github.com/metatensor/hpc-docs
cd hpc-docs
uenv build CSCS-Alps/uenv/lammps-metatomic/ lmp/mta@daint%gh200
```

This will build an uenv with the `lammps-metatomic` recipe, targeting the `daint` cluster and the `gh200` architecture.
A tag will be automatically generated for the uenv; look at the output of `uenv build` for the exact name of the uenv,
and for the link to the build logs.

Once the build is complete, the uenv will be available in the `service::` namespace.

```bash
uenv image pull service::lmp/mta:<CI_TAG>@daint%gh200
```

Once you pull an uenv from the `service::` namespace, you can use it as you would use any other uenv,
without the need to specify the `service::` namespace.

### Build uenv with stackinator

[Install `stackinator`] and clone the [Alps cluster configuration] repository.

```bash
git clone https://github.com/eth-cscs/stackinator.git
cd stackinator
uv tool install --editable .
cd ..
```

```bash
git clone https://github.com/eth-cscs/alps-cluster-config.git
```

Build an uenv from a recipe using `stackinator`:

```bash
git clone https://github.com/metatensor/hpc-docs
cd hpc-docs
stack-config --build /dev/shm/$USER/lmp-mta --recipe CSCS-Alps/uenv/lammps-metatomic -s $PWD/../alps-cluster-config/daint
```

It is recommended to build under `/dev/shm/$USER/`.
The path to the cluster configuration should point to the correct cluster directory in the `alps-cluster-config` repository.

If you are doing repeated builds (e.g., for testing different dependencies or development), 
you can [setup a local build cache]. `uenv build` automatically uses a build cache.

## Available recipes

### lammps-metatomic

Recipe for building an uenv with LAMMPS and the `metatomic` package.
The LAMMPS Spack package is a custom package, which uses [metatensor/LAMMPS],
since the `metatomic` package is not yet available in the official LAMMPS distribution (as of 2026-11).

The main changes are the following:
1. Changed `git` attribute to use `metatensor/LAMMPS`
2. Added `+metatomic` variant to enable `metatomic` (`lammps+metatomic`)
3. Added dependencies `libmetatensor-torch`, `libmetatomic-torch`, `py-torch` (when `+metatomic`)
4. Enabled `PKG_ML-METATOMIC` when `+metatomic`, and enforced usage of Spack-installed `metatensor` and `metatomic`
5. Removed `^[virtuals=mpi] cray-mpich`-specific handling, it's already accounted for in Alps' custom `cray-mpich` package 

The version of PyTorch provided by this uenv is non-distributed.

[uenv]: https://docs.cscs.ch/software/uenv/
[Spack]: https://spack.readthedocs.io/en/latest/
[Spack packaging guide]: https://spack.readthedocs.io/en/latest/packaging_guide_creation.html#
[`stackinator`]: https://eth-cscs.github.io/stackinator/
[uenv build service]: https://docs.cscs.ch/software/uenv/build/#uenv-build-service
[uenv recipe]: https://eth-cscs.github.io/stackinator/recipes/
[Install `stackinator`]: https://eth-cscs.github.io/stackinator/#getting-stackinator
[Alps cluster configuration]: https://github.com/eth-cscs/alps-cluster-config
[setup a local build cache]: https://eth-cscs.github.io/stackinator/mirrors/
