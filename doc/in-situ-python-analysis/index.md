# In Situ Python Analysis

```{toctree}
:hidden:

inline-python-script
using-yt
libyt-python-module
libyt-defined-command
interactive-python-prompt
reloading-script
jupyter-notebook/jupyter-notebook-access
limitation
faq
```

## Using Python for In Situ Analysis in libyt

- [**Inline Python Script**](./inline-python-script.md#inline-python-script): how inline Python script works?
- [**Using yt**](./using-yt.md#using-yt): how to use `yt` for in situ analysis?
- [**libyt Python Module**](./libyt-python-module.md#libyt-python-module): the Python module where libyt API binds simulation data to.

## User Interface

- [**libyt Defined Commands**](./libyt-defined-command.md#libyt-defined-commands): the commands allow users to load script, export history, set function to run or idle, etc.
- [**Interactive Python Prompt**](./interactive-python-prompt.md#interactive-python-prompt): the prompt can process Python statements and give feedback instantly. ([`-DINTERACTIVE_MODE=ON`](../how-to-install/how-to-install.md#-dinteractive_mode-off))
- [**Reloading Script**](./reloading-script.md#reloading-script): this allows reloading Python script. ([`-DINTERACTIVE_MODE=ON`](../how-to-install/how-to-install.md#-dinteractive_mode-off))
- [**Jupyter Notebook Access**](./jupyter-notebook/jupyter-notebook-access.md#jupyter-notebook-access): this allows accessing and analyzing simulation data through JupyterLab/Jupyter Notebook. ([`-DJUPYTER_KERNEL=ON`](../how-to-install/how-to-install.md#-djupyter_kernel-off))

## Limitations

- [**Limitations in MPI Related Python Tasks**](./limitation.md#limitations-in-mpi-related-python-tasks)

## FAQs

- [**How Does libyt Run Python Script?**](./faq.md#how-does-libyt-run-python-script)
