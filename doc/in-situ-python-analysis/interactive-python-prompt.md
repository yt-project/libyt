# Interactive Python Prompt

## Requirements

- Compile `libyt` in **interactive mode** ([`-DINTERACTIVE_MODE=ON`](../how-to-install/details.md#-dinteractive_mode-off)).
- Call libyt API [`yt_run_InteractiveMode`](../libyt-api/yt_run_interactivemode.md#yt_run_interactivemode).

> {octicon}`info;1em;sd-text-info;` Interactive mode only works on local machine or submit the job to HPC platform using interactive jobs like `qsub -I` on PBS scheduler.
> The reason is that the prompt is exposed through the terminal. Use [Reloading Script](./reloading-script.md#reloading-script) or [Jupyter Notebook Access](./jupyter-notebook/jupyter-notebook-access.md#jupyter-notebook-access) to avoid this.

## Using Interactive Python Prompt

### Status Board
At the start of interactive mode, `libyt` prints the [status board](./libyt-defined-command.md#status-board) to show the current execution state of each inline function.

### Python Prompt and libyt Defined Commands
```
>>> 
```
It can run any Python statements and [libyt Defined Commands](./libyt-defined-command.md#libyt-defined-commands) here. This includes importing Python modules.
This is like using normal Python prompt, except that there are functions, objects, and simulation data already defined in it.

> {octicon}`info;1em;sd-text-info;` Python statement runs in inline [`script`](../libyt-api/yt_initialize.md#yt_param_libyt)'s namespace, and it will update and make changes to it. The Python objects defined in it will be brought to next in situ analysis. 

> {octicon}`info;1em;sd-text-info;` To make interactive prompt more smoothly, set lower [`YT_VERBOSE`](../libyt-api/yt_initialize.md#yt_param_libyt).

:::
##### How does the prompt execute Python statements and libyt defined commands?
:::
1. MPI root process is the user interface, which takes user inputs.
2. MPI root process broadcasts the inputs to other processes.
3. Every other MPI process executes the same piece of input synchronously.
4. Print feedbacks to the terminal.

> {octicon}`alert;1em;sd-text-danger;` Changes is kept and maintained in user's inline [`script`](../libyt-api/yt_initialize.md#yt_param_libyt)'s namespace. The Python objects and function definition will be brought to the following round of in situ analysis.

### Exit Interactive Python Prompt
Use [`%libyt exit`](./libyt-defined-command.md#exit) to exit the prompt.
Then it will exit the API [`yt_run_InteractiveMode`](../libyt-api/yt_run_interactivemode.md#yt_run_interactivemode) and return back to the simulation process.

## Known Limitations
- Cannot use arrow keys or `Ctrl+D` in interactive mode.
- See [Limitations in MPI Related Python Tasks](./limitation.md#limitations-in-mpi-related-python-tasks).

## FAQs

- [Why Can't I Find the Prompt `>>>`?](../FAQs.md#why-cant-i-find-the-prompt-)
- [Where Can I Use Interactive Mode?](../FAQs.md#where-can-i-use-interactive-mode)