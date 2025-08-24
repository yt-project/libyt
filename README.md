# libyt
[![build test](https://github.com/yt-project/libyt/actions/workflows/cmake-build-test.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/cmake-build-test.yml)
[![amr example](https://github.com/yt-project/libyt/actions/workflows/example-test-run.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/example-test-run.yml)
[![unit test](https://github.com/yt-project/libyt/actions/workflows/unit-test.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/unit-test.yml)
[![memory profile](https://github.com/yt-project/libyt/actions/workflows/memory-profile.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/memory-profile.yml)
[![Documentation Status](https://readthedocs.org/projects/libyt/badge/?version=latest)](https://libyt.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/yt-project/libyt/graph/badge.svg?token=NRYLAipewN)](https://codecov.io/gh/yt-project/libyt)

`libyt` is an open-source C library for simulation that enables researchers to analyze and visualize data using [`yt`](https://yt-project.org/) or other Python packages in parallel during simulation runtime. 

We can skip the intermediate step of writing data to a hard disk before doing analysis using Python. 
This significantly reduces the disk usage and increases the temporal resolution. 
Converting the post-processing analysis Python script using `yt` to an inline Python script is a two-line change.

`libyt` also provides a Python interface (Python prompt, file-based prompt, and JupyterLab frontend) to access simulation data running on an HPC cluster, enabling real-time visualization and sampling of the data.

- **Documents**: https://libyt.readthedocs.io/

## Related Projects

- [`yt`](https://github.com/yt-project/yt): a Python package for analyzing and visualizing volumetric data. It is the core method that `libyt` uses to provide data analysis pipeline for simulations.
- [`yt_libyt`](https://github.com/data-exp-lab/yt_libyt): a Python package that provides a `yt` frontend. It makes converting post-processing scripts using `yt` into inline scripts a two-line change.
- [`jupyter_libyt`](https://github.com/yt-project/jupyter_libyt): a JupyterLab frontend for `libyt`. It provides methods to connect to simulations.

## Installation

More details in [how to install](https://libyt.readthedocs.io/en/latest/how-to-install/how-to-install.html).

#### Serial 

```bash
cmake -S . -B build -DSERIAL_MODE=ON \
                    -DINTERACTIVE_MODE=ON \
                    -DJUPYTER_KERNEL=ON \
                    -DPYTHON_PATH=<your-python-prefix>
```

#### Parallel with MPI

```bash
cmake -S . -B build -DINTERACTIVE_MODE=ON \
                    -DJUPYTER_KERNEL=ON \
                    -DPYTHON_PATH=<your-python-prefix> \
                    -DMPI_PATH=<your-mpi-prefix>
```


## Develop and Contributing

This project is currently in an active development stage. Some of its cores and architectures are subject to change to make it more efficient, extendable, and easy to use.

We encourage and welcome the submission of issues, feature requests, and suggestions. 
Such contributions are invaluable to the continued improvement of this project, and we appreciate the time and effort taken to help us grow and better serve the community.



## Code of Conduct

We are committed to fostering a respectful, inclusive, and harassment-free environment for everyone. 
All participants are expected to treat one another with kindness, regardless of their background, identity, or experience. 
Harassment, discrimination, personal attacks, or any other disruptive behavior will not be tolerated. 
By participating in this community, you agree to uphold these standards and contribute to creating a welcoming space.

## License

BSD 3-Clause License
