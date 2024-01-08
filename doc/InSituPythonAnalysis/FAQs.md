---
layout: default
title: FAQs
parent: In Situ Python Analysis
nav_order: 8
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

## How Does libyt Run Python Script?
`libyt` runs Python script synchronously, which means every MPI process runs the same piece of Python code. 
They do the job together under the process space of MPI tasks using [`mpi4py`](https://mpi4py.readthedocs.io/en/stable/index.html). 


