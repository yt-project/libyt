---
hide-toc: true
---

# libyt Documents

`libyt` is an open source C library for simulation. 
It is an in situ analysis tool that allows researchers to analyze and visualize data using [`yt`](https://yt-project.org/) or other Python packages in parallel during simulation runtime.

```{toctree}
:hidden:

HowToInstall
HowItWorks
Example
libytAPI/index
InSituPythonAnalysis/index
DebugAndTimeProfiling/index
```

## Contents
- [**How to Install**](./HowToInstall.md#how-to-install): how to get `libyt`?
- [**How it Works**](./HowItWorks.md#how-it-works): what is happening behind the scene?
- [**Example**](./Example.md#example): this demonstrates how to implement `libyt` in simulation step by step.
- [**libyt API**](./libytAPI/index.md#libyt-api): this is for simulation developers that wish to implement `libyt`.
- [**In Situ Python Analysis**](./InSituPythonAnalysis/index.md#in-situ-python-analysis): this is for users that are using `libyt` to do in situ analysis.
- [**Debug and Time Profiling**](./DebugAndTimeProfiling/index.md#debug-and-time-profiling): how to check inputs and run time profiling in `libyt`.

## Label Meanings
- ✏️ means `libyt` only borrows the variable and does not make a copy.
- ℹ️ means more information on this topic.
- ⚠️ means things we should be careful.
- 🦎 means things we are trying to improve.
