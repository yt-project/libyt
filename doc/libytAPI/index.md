# libyt API

```{toctree}
:hidden:

Initialize
SetYTParameter
SetCodeSpecificParameter
FieldInfo/SetFieldsInformation
SetParticlesInformation
SetLocalGridsInformation
LookUpPassedInData
CommitYourSettings
PerformInlineAnalysis
ActivateInteractiveMode
ActivateReloadingScript
ActivateJupyterKernel
FreeResource
Finalize
DataType
```

## Procedure
It can break down into five stages: 
  - initialization, 
  - loading simulation data into Python[^1], 
  - do in situ analysis, 
  - reset, 
  - and finalization.

## libyt API

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
      <td rowspan=3><strong>Loading data</strong></td>
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
      <td rowspan=4><strong>In situ analysis</strong></td>
      <td><code>yt_run_Function</code>, <code>yt_run_FunctionArguments</code></td>
      <td>Run Python functions.</td>
    </tr>
    <tr>
      <td><code>yt_run_InteractiveMode</code></td>
      <td>Activate interactive prompt. This is only available in interactive mode.</td>
    </tr>
    <tr>
      <td><code>yt_run_ReloadScript</code></td>
      <td>Enter reloading script phase. This is only available in interactive mode.</td>
    </tr>
    <tr>
      <td><code>yt_run_JupyterKernel</code></td>
      <td>Activate interactive prompt. This is only available in Jupyter kernel mode.</td>
    </tr>
    <tr>
      <td rowspan=1><strong>Reset</strong></td>
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


[^1]: :lizard: Currently, `libyt` only supports loading simulation data with adaptive mesh refinement grid structure (AMR grid) to Python. We are trying to make `libyt` works with more data structure.

