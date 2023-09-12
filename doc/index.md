---
layout: home
title: Home
nav_order: 1
---

# libyt Documents (Dev)
Version 0.1.0
{: .label }

`libyt` is an open source C library for simulation. 
It is an in situ analysis tool that allows researchers to analyze and visualize data using [`yt`](https://yt-project.org/) or other Python packages in parallel during simulation runtime.
{: .fs-6 .fw-300 }


## Content
- [**How to Install**]({% link HowToInstall.md %}#how-to-install): how to get `libyt`?
- [**How it Works**]({% link HowItWorks.md %}#how-it-works): what is happening behind the scene?
- [**Example**]({% link Example.md %}#example): this demonstrates how to implement `libyt` in simulation step by step.
- [**libyt API**]({% link libytAPI/index.md %}#libyt-api): this is for simulation developers that wish to implement `libyt`.
- [**In Situ Python Analysis**]({% link InSituPythonAnalysis/index.md %}#in-situ-python-analysis): this is for users that are using `libyt` in simulations.

## Label Meanings

- :pencil2: means be careful about the lifetime of this variable. `libyt` only borrows it, and does not make a copy.
- :information_source: means more information on this topic.
- :warning: means things we should be careful.
- :lizard: means things we are trying to improve.