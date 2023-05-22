# User Guide
This guide will walk you through how to implement `libyt` into your code. And how you can convert your everyday used `yt` script to do inline-analysis. 

- Implement `libyt` to your code step by step
  - [Initialize - `yt_initialize`](libytAPI/Initialize.md#initialize)
  - [Set `yt` Parameter - `yt_set_Parameters`](libytAPI/SetYTParameter.md#set-yt-parameter)
  - [Set Code Specific Parameter - `yt_set_UserParameter*`](libytAPI/SetCodeSpecificParameter.md#set-code-or-user-specific-parameter)
  - [Set Fields Information - `yt_get_FieldsPtr`](libytAPI/FieldInfo/SetFieldsInformation.md#set-fields-information)
  - [Set Particles Information - `yt_get_ParticlesPtr`](libytAPI/SetParticlesInformation.md#set-particles-information)
  - [Set Local Grids Information - `yt_get_GridsPtr`](libytAPI/SetLocalGridsInformation.md#set-local-grids-information)
  - [Commit Your Settings - `yt_commit`](libytAPI/CommitYourSettings.md#commit-your-settings)
  - [Perform Inline-Analysis - `yt_run_Function` and `yt_run_FunctionArguments`](libytAPI/PerformInlineAnalysis.md#perform-inline-analysis)
  - [Activate Interactive Mode - `yt_run_InteractiveMode`](libytAPI/ActivateInteractiveMode.md#activate-interactive-mode) (Only available in interactive mode)
  - [Free Resource - `yt_free`](libytAPI/FreeResource.md#free-resource)
  - [Finalize - `yt_finalize`](libytAPI/Finalize.md#finalize)
- [Inline Python Script](InSituPythonAnalysis/InlinePythonScript.md#inline-python-script)

## Example
- [`libyt/example`](../example/example.cpp)
- [`gamer`](https://github.com/gamer-project/gamer/tree/master/src/YT)

## Overview
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
      <td rowspan=1>Initialization</td>
      <td>yt_initialize</td>
      <td>Initialize embedded Python and import inline Python script.</td>
    </tr>
    <tr>
      <td rowspan=6>Iteration</td>
      <td>yt_set_Parameters, yt_set_UserParameter*</td>
      <td>Set parameters.</td>
    </tr>
    <tr>
      <td>yt_get_FieldsPtr, yt_get_ParticlesPtr, yt_get_GridsPtr</td>
      <td>Get fields, particles, and grids information array (ptr), and write in it.</td>
    </tr>
    <tr>
      <td>yt_commit</td>
      <td>Tell libyt you're done.</td>
    </tr>
    <tr>
      <td>yt_run_Function, yt_run_FunctionArguments</td>
      <td>Run Python functions.</td>
    </tr>
    <tr>
      <td>yt_run_InteractiveMode (Only available in interactive mode)</td>
      <td>Activate interactive prompt.</td>
    </tr>
    <tr>
      <td>yt_free</td>
      <td>Free resources.</td>
    </tr>
    <tr>
      <td rowspan=1>Finalization</td>
      <td>yt_finalize</td>
      <td>Finalize embedded Python.</td>
    </tr>
  </tbody>
</table>
