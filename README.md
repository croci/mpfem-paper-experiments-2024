# Mixed-precision finite element kernels

## Documentation

This software generates the numerical results of the manuscript *Mixed-precision finite element kernels and assembly: Rounding error analysis and hardware acceleration* by M. Croci and G. N. Wells.
See techreport in the [Zenodo upload](https://doi.org/10.5281/zenodo.13941629) for more information.

## Important

These codes can only be run on a machine with an Intel Sapphire Rapids CPU!!!

## Paper experiments

The code is in `src/local_kernel`.

Before running the kernel scripts one first has to generate the FFCX kernels and the meshes. These require
[FEniCSx](https://fenicsproject.org/) and [gmsh](https://gmsh.info/) to be installed, but only have to be generated once.
The easiest way is to use one of the fenicsproject docker containers.

**ALTERNATIVE:** Directly download the dataset and software from the [Zenodo upload](https://doi.org/10.5281/zenodo.13941629) to
directly skip to Step 3. The original software is also part of the Zenodo upload and can be used instead of the GitHub version.
However, the Zenodo version won't be kept up to date.

1- Generate the FFCX kernels by running the `ffcx_codegen.py` script in `src/ffcx`, i.e.,

```bash
cd src/ffcx
python3 ffcx_codegen.py
```

2- Generate the mesh files by running

```bash
cd src/local_kernel/meshes
gmsh -3 tet.geo && gmsh -3 cube.geo && python3 get_mesh_coordinates.py
```

3- Check that all dependencies have been installed (see "Dependencies" below):

```bash
cd src/local_kernel
make
./kernel.run
```

4- Run all the experiments (it takes some time):

```bash
python3 run_all_kernels.py
```

5- Plot results (requires matplotlib):

```bash
python3 plot_results.py
```

## Dependencies

Hardware dependency: These codes can only be run on a machine with an Intel Sapphire Rapids CPU!!!

Software dependencies: `numpy`, `gcc@14.2.0+binutils` and `intel-oneapi-mkl` (optional). You can use [spack](https://spack.io/) to install these with 

```bash
spack install gcc@14.2.0+binutils
spack install intel-oneapi-mkl
```

**NOTES**: 
* `gcc@14.2.0` can be replaced with `llvm@18.1.2~libomptarget` or `intel-oneapi-compilers@2024.2.1`, but the code is untested for these compilers.
* `intel-oneapi-mkl` is optional and only needed for the gemm tests.
