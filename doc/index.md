---
hide-toc: true
---

# libyt Documents
{bdg-primary}`version 0.1.0`

`libyt` is an open source C library for simulation. 
It is an in situ analysis tool that allows researchers to analyze and visualize data using [`yt`](https://yt-project.org/) or other Python packages in parallel during simulation runtime.

```{toctree}
:hidden:

how-to-install
how-it-works
example
libyt-api/index
in-situ-python-analysis/index
debug-and-profiling/index
```

#### Contents
- [**How to Install**](./how-to-install.md#how-to-install): how to get `libyt`?
- [**How it Works**](./how-it-works.md#how-it-works): what is happening behind the scene?
- [**Example**](./example.md#example): this demonstrates how to implement `libyt` in simulation step by step.
- [**libyt API**](./libyt-api/index.md#libyt-api): this is for simulation developers that wish to implement `libyt`.
- [**In Situ Python Analysis**](./in-situ-python-analysis/index.md#in-situ-python-analysis): this is for users that are using `libyt` to do in situ analysis.
- [**Debug and Time Profiling**](./debug-and-profiling/index.md#debug-and-time-profiling): how to check inputs and run time profiling in `libyt`.

#### Label Meanings
- {octicon}`pencil;1em;sd-text-warning;` means `libyt` only borrows the variable and does not make a copy.
- {octicon}`info;1em;sd-text-info;` means more information on this topic.
- {octicon}`alert;1em;sd-text-danger;` means things we should be careful.
- {octicon}`calendar;1em;sd-text-secondary;` means things we are trying to improve.
