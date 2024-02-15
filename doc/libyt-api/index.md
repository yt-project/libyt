# libyt API

```{toctree}
:hidden:

yt_initialize
yt_set_parameters
yt_set_userparameter
field/yt_get_fieldsptr
yt_get_particlesptr
yt_get_gridsptr
yt_getgridinfo
yt_commit
run-python-function
yt_run_interactivemode
yt_run_reloadscript
yt_run_jupyterkernel
yt_free
yt_finalize
data-type
```

## Procedure

It can break down into five stages:

- initialization,
- loading simulation data into Python[^1],
- do in situ analysis,
- reset,
- and finalization.

## libyt API

<div class="table-wrapper colwidths-auto docutils container">
<table class="docutils align-default">
  <thead>
    <tr class="row-odd"><th class="head"><p>Stage</p></th>
      <th class="head"><p>libyt API</p></th>
      <th class="head"><p>Description</p></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=1><strong>Initialization</strong></td>
      <td><p><a class="reference internal" href="yt_initialize.html#yt-initialize-initialize"><code class="docutils literal notranslate"><span class="pre">yt_initialize</span></code></a></p></td>
      <td>Initialize embedded Python and import inline Python script.</td>
    </tr>
    <tr>
      <td rowspan=3><strong>Loading data</strong></td>
      <td><p><a class="reference internal" href="yt_set_parameters.html#yt-set-parameters-set-yt-parameters"><code class="docutils literal notranslate"><span class="pre">yt_set_Parameters</span></code></a>, <a class="reference internal" href="yt_set_userparameter.html#yt-set-userparameter-set-other-parameters"><code class="docutils literal notranslate"><span class="pre">yt_set_UserParameter*</span></code></a></p></td>
      <td>Set yt parameters and user specific parameters.</td>
    </tr>
    <tr>
      <td><p><a class="reference internal" href="field/yt_get_fieldsptr.html#yt-get-fieldsptr-set-field-information"><code class="docutils literal notranslate"><span class="pre">yt_get_FieldsPtr</span></code></a>, <a class="reference internal" href="yt_get_particlesptr.html#yt-get-particlesptr-set-particle-information"><code class="docutils literal notranslate"><span class="pre">yt_get_ParticlesPtr</span></code></a>, <a class="reference internal" href="yt_get_gridsptr.html#yt-get-gridsptr-set-local-grids-information"><code class="docutils literal notranslate"><span class="pre">yt_get_GridsPtr</span></code></a></p></td>
      <td>Get fields, particles, and grids information array (ptr), and write corresponding data in.</td>
    </tr>
    <tr>
      <td><p><a class="reference internal" href="yt_commit.html#yt-commit-commit-your-settings"><code class="docutils literal notranslate"><span class="pre">yt_commit</span></code></a></p></td>
      <td>Tell libyt you're done.</td>
    </tr>
    <tr>
      <td rowspan=4><strong>In situ analysis</strong></td>
      <td><p><a class="reference internal" href="run-python-function.html#yt-run-function"><code class="docutils literal notranslate"><span class="pre">yt_run_Function</span></code></a>, <a class="reference internal" href="run-python-function.html#yt-run-functionarguments"><code class="docutils literal notranslate"><span class="pre">yt_run_FunctionArguments</span></code></a></p></td>
      <td>Run Python functions.</td>
    </tr>
    <tr>
      <td><p><a class="reference internal" href="yt_run_interactivemode.html#yt-run-interactivemode-activate-interactive-python-prompt"><code class="docutils literal notranslate"><span class="pre">yt_run_InteractiveMode</span></code></a></p></td>
      <td>Activate interactive prompt. This is only available in interactive mode.</td>
    </tr>
    <tr>
      <td><p><a class="reference internal" href="yt_run_reloadscript.html#yt-run-reloadscript-reload-script"><code class="docutils literal notranslate"><span class="pre">yt_run_ReloadScript</span></code></a></p></td>
      <td>Enter reloading script phase. This is only available in interactive mode.</td>
    </tr>
    <tr>
      <td><p><a class="reference internal" href="yt_run_jupyterkernel.html#yt-run-jupyterkernel-activate-jupyter-kernel"><code class="docutils literal notranslate"><span class="pre">yt_run_JupyterKernel</span></code></a></p></td>
      <td>Activate interactive prompt. This is only available in Jupyter kernel mode.</td>
    </tr>
    <tr>
      <td rowspan=1><strong>Reset</strong></td>
      <td><p><a class="reference internal" href="yt_free.html#yt-free-free-libyt-resource"><code class="docutils literal notranslate"><span class="pre">yt_free</span></code></a></p></td>
      <td>Free resources for in situ analysis.</td>
    </tr>
    <tr>
      <td rowspan=1><strong>Finalization</strong></td>
      <td><p><a class="reference internal" href="yt_finalize.html#yt-finalize-finalize"><code class="docutils literal notranslate"><span class="pre">yt_finalize</span></code></a></p></td>
      <td>Finalize libyt.</td>
    </tr>
  </tbody>
</table>
</div>


[^1]: {octicon}`calendar;1em;sd-text-secondary;` Currently, `libyt` only supports loading simulation data with adaptive mesh refinement grid structure (
AMR grid) to Python. We are trying to make `libyt` works with more data structure.

