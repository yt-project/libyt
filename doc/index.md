---
hide-toc: true
---

# libyt Documents
{bdg-primary}`version 0.3.0`

`libyt` is an open source C library for simulation. 
It is an in situ analysis tool that allows researchers to analyze and visualize data using [`yt`](https://yt-project.org/) or other Python packages in parallel during simulation runtime.

```{toctree}
:hidden:
:caption: For users

quick-start
example
how-to-install/how-to-install
how-it-works
in-situ-python-analysis/index
FAQs
```

```{toctree}
:hidden:
:caption: For developers

libyt-api/index
debug-and-profiling/index
api-doc/index
```

#### Contents

**For users:**
- [**Quick Start**](./quick-start.md)
- [**Example**](./example.md): this demonstrates how to implement `libyt` in an amr simulation step by step.
- [**How to Install**](./how-to-install/how-to-install.md#how-to-install): how to get `libyt`?
- [**How it Works**](./how-it-works.md): what is happening behind the scene?
- [**In Situ Python Analysis**](./in-situ-python-analysis/index.md): this is for users that are using `libyt` to do in situ analysis.
- [**FAQs**](./FAQs.md)

**For simulation developers:**
- [**libyt API**](./libyt-api/index.md): this is for simulation developers that wish to implement `libyt`.
- [**Debug and Time Profiling**](./debug-and-profiling/index.md): how to check inputs and run time profiling in `libyt`.
- [**Index**](./api-doc/index.md)
