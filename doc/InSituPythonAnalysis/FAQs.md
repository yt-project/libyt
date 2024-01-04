---
layout: default
title: FAQs
parent: In Situ Python Analysis
nav_order: 7
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

## Why Does my Program Hang and How Do I Solve It?
Though `libyt` can execute any Python module, when it comes to reading simulation data, it requires every MPI process to participate
The program hanging problem is due to only some MPI processes are accessing the data, but not all of them.

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
