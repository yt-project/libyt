# libyt
[![build test](https://github.com/yt-project/libyt/actions/workflows/cmake-build-test.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/cmake-build-test.yml)
[![amr example](https://github.com/yt-project/libyt/actions/workflows/example-test-run.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/example-test-run.yml)
[![unit test](https://github.com/yt-project/libyt/actions/workflows/unit-test.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/unit-test.yml)
[![memory profile](https://github.com/yt-project/libyt/actions/workflows/memory-profile.yml/badge.svg?branch=main)](https://github.com/yt-project/libyt/actions/workflows/memory-profile.yml)
[![Documentation Status](https://readthedocs.org/projects/libyt/badge/?version=latest)](https://libyt.readthedocs.io/en/latest/?badge=latest)

`libyt` is an open source C library for simulation, that allows researchers to analyze and visualize data using [`yt`](https://yt-project.org/) or other Python packages in parallel during simulation runtime. In this way, we can skip the step of writing data to local disk before doing analysis using Python. This greatly reduce the disk usage, and increase the temporal resolution.

- **Documents**: https://libyt.readthedocs.io/
