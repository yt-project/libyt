# libyt
[![build-test](https://github.com/cindytsai/libyt/actions/workflows/build-test.yml/badge.svg?branch=master)](https://github.com/cindytsai/libyt/actions/workflows/build-test.yml)

`libyt` is a C library for C/C++ simulation to use [`yt`](https://yt-project.org/) and Python to do in situ analysis, while the simulation is still running. In this way, we can skip the step of writing data to local disk before any analysis using Python can happen. This greatly reduce the disk usage, and increase the temporal resolution.

- **Implement `libyt` into your code** :arrow_right:
  - [Installation](#installation)
  - [User Guide](#user-guide)
  - [Example](#example)
- **See `libyt` working progress** :arrow_right: [Project Board](https://github.com/calab-ntu/libyt/projects/1)
- **See how `libyt` is developed** :arrow_right: [`libyt` Milestone](https://hackmd.io/@Viukb0eMS-aeoZQudVyJ2w/ryCYwu0xF)
- **Supported `yt` functionalities**:

  |      `yt` Function       | Supported | Notes                                    |
  |:------------------------:|:---------:|------------------------------------------|
  |        `find_max`        |     V     |                                          |
  |     `ProjectionPlot`     |     V     |                                          |
  | `OffAxisProjectionPlot`  |     V     |                                          |
  |       `SlicePlot`        |     V     |                                          |
  |    `OffAxisSlicePlot`    |     V     |                                          |
  |     `covering_grid`      |     V     |                                          |
  |   1D `create_profile`    |     V     |                                          |
  |   2D `create_profile`    |     V     |                                          |
  |      `ProfilePlot`       |     V     |                                          |
  |       `PhasePlot`        |     V     |                                          |
  |        `LinePlot`        |     V     |                                          |
  |      Halo Analysis       |           | Not test yet.                            |
  |       Isocontours        |     V     |                                          |
  |     `volume_render`      |     V     | Only when MPI size is even will it work. |
  |      `ParticlePlot`      |     V     |                                          |
  | `ParticleProjectionPlot` |     V     |                                          |


## User Guide
This guide will walk you through how to implement `libyt` into your code. And how you can convert your everyday used `yt` script to do inline-analysis. All the user guide are in [`doc`](./doc) folder.


- Implement `libyt` to your code step by step
  - [Initialize - `yt_initialize`](doc/libytAPI/Initialize.md#initialize)
  - [Set `yt` Parameter - `yt_set_Parameters`](doc/libytAPI/SetYTParameter.md#set-yt-parameter)
  - [Set Code Specific Parameter - `yt_set_UserParameter*`](doc/libytAPI/SetCodeSpecificParameter.md#set-code-or-user-specific-parameter)
  - [Set Fields Information - `yt_get_FieldsPtr`](doc/libytAPI/FieldInfo/SetFieldsInformation.md#set-fields-information)
  - [Set Particles Information - `yt_get_ParticlesPtr`](doc/libytAPI/SetParticlesInformation.md#set-particles-information)
  - [Set Local Grids Information - `yt_get_GridsPtr`](doc/libytAPI/SetLocalGridsInformation.md#set-local-grids-information)
  - [Commit Your Settings - `yt_commit`](doc/libytAPI/CommitYourSettings.md#commit-your-settings)
  - [Perform Inline-Analysis - `yt_run_Function` and `yt_run_FunctionArguments`](doc/libytAPI/PerformInlineAnalysis.md#perform-inline-analysis)
  - [Activate Interactive Mode - `yt_run_InteractiveMode`](doc/libytAPI/ActivateInteractiveMode.md#activate-interactive-mode) (Only available in interactive mode)
  - [Free Resource - `yt_free`](doc/libytAPI/FreeResource.md#free-resource)
  - [Finalize - `yt_finalize`](doc/libytAPI/Finalize.md#finalize)
- [Inline Python Script](doc/InSituPythonAnalysis/InlinePythonScript.md#inline-python-script)

## Example
- A simple demo in [`libyt/example.cpp`](./example/example.cpp)
- [`gamer`](https://github.com/gamer-project/gamer/tree/master/src/YT)
