---
layout: default
title: libyt API
nav_order: 5
has_children: true
permalink: libytAPI
has_toc: false
---

# libyt API
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

## Procedure
Currently, `libyt` only supports loading simulation data with adaptive mesh refinement grid structure (AMR grid) to Python.[^1]

<table>
  <thead>
    <tr>
      <th>Stage</th>
      <th>libyt API</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=1><strong>Initialization</strong></td>
      <td><code>yt_initialize</code></td>
      <td>Initialize embedded Python and import inline Python script.</td>
    </tr>
    <tr>
      <td rowspan=6><strong>Iteration</strong></td>
      <td><code>yt_set_Parameters</code>, <code>yt_set_UserParameter*</code></td>
      <td>Set yt parameters and user specific parameters.</td>
    </tr>
    <tr>
      <td><code>yt_get_FieldsPtr</code>, <code>yt_get_ParticlesPtr</code>, <code>yt_get_GridsPtr</code></td>
      <td>Get fields, particles, and grids information array (ptr), and write corresponding data in.</td>
    </tr>
    <tr>
      <td><code>yt_commit</code></td>
      <td>Tell libyt you're done.</td>
    </tr>
    <tr>
      <td><code>yt_run_Function</code>, <code>yt_run_FunctionArguments</code></td>
      <td>Run Python functions.</td>
    </tr>
    <tr>
      <td><code>yt_run_InteractiveMode</code></td>
      <td>Activate interactive prompt. This is only available in interactive mode.</td>
    </tr>
    <tr>
      <td><code>yt_free</code></td>
      <td>Free resources for in situ analysis.</td>
    </tr>
    <tr>
      <td rowspan=1><strong>Finalization</strong></td>
      <td><code>yt_finalize</code></td>
      <td>Finalize libyt.</td>
    </tr>
  </tbody>
</table>

## Label Meanings
- :pencil2: means be careful about the lifetime of this variable. `libyt` only borrows it, and does not make a copy.

----
[^1] Even though we can still activate Python prompt and run in situ Python script with no data loaded ahead in non-AMR grid simulation with just `yt_initialize`, `yt_run_Function`, `yt_run_FunctionArguments`, `yt_run_InteractiveMode`, and `yt_finalize`. We are trying to make `libyt` works with more data structure.


