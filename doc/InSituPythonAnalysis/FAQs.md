---
layout: default
title: FAQs
parent: In Situ Python Analysis
nav_order: 5
---
# Frequently Asked Questions
{: .no_toc }
<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
- TOC
{:toc}
</details>
---

## Why does my program hang and how do I solve it?
This is probably due to only some MPI processes are accessing the data, and not all of them.

Please do:
1. Check if there is an if statements that makes MPI processes non-symmetric. For example, only root process runs the statement:
  ```python
  def func():
      if yt.is_root():
          ...  # <-- This statement only executes in MPI root rank
  ```
2. Move the statement out of `if yt.is_root()` (for the case here).

> :lizard: When accessing simulation data, `libyt` requires every process to participate in this.
> We are working on this in both `yt` and `libyt`.
