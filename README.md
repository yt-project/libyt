# libyt
[![build test](https://github.com/yt-project/libyt/actions/workflows/cmake-build-test.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/cmake-build-test.yml)
[![amr example](https://github.com/yt-project/libyt/actions/workflows/example-test-run.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/example-test-run.yml)
[![unit test](https://github.com/yt-project/libyt/actions/workflows/unit-test.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/unit-test.yml)
[![memory profile](https://github.com/yt-project/libyt/actions/workflows/memory-profile.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/memory-profile.yml)
[![Documentation Status](https://readthedocs.org/projects/libyt/badge/?version=latest)](https://libyt.readthedocs.io/en/latest/?badge=latest)

`libyt` is an open source C library for simulation, that allows researchers to analyze and visualize data using [`yt`](https://yt-project.org/) or other Python packages in parallel during simulation runtime. 

We can skip the intermediate step of writing data to a hard disk before doing analysis using Python. This greatly reduce the disk usage, and increase the temporal resolution. Converting post-processing analysis Python script using `yt` to inline Python script is a two-line change.

`libyt` also provides a Python interface (Python prompt, file-base prompt, and JupyterLab frontend) to access the data in simulations running in an HPC cluster, which can be used to visualize and debug the data in real-time.

- **Documents**: https://libyt.readthedocs.io/

### Related Projects

- [`yt_libyt`](https://github.com/data-exp-lab/yt_libyt): a Python package that provides a `yt` frontend. It makes converting post-processing scripts using `yt` into inline scripts a two-line change.
- [`jupyter_libyt`](https://github.com/yt-project/jupyter_libyt): a JupyterLab frontend for `libyt`. It provides methods to connect to simulations.

### Installation

See the [how to install](https://libyt.readthedocs.io/en/latest/how-to-install/how-to-install.html).

### Contributing

We welcome contributions of all kinds! Whether you're fixing a bug, adding a feature, improving documentation, or reporting an issue -- thank you for helping improve this project.

Please follow the coding style when committing to the git history using `pre-commit` hooks. 

### Code of Conduct

We are committed to fostering a welcoming, respectful, and inclusive environment for everyone involved in this project.

### License

BSD 3-Clause License
