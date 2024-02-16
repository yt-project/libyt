# Interactive Python Prompt

## Requirements

- Compile `libyt` in [**interactive mode**](../how-to-install.md#dinteractive-mode).
- Call libyt API [`yt_run_InteractiveMode`]({% link libytAPI/ActivateInteractiveMode.md %}#activate-interactive-mode).

> {octicon}`info;1em;sd-text-info;` Interactive mode only works on local machine or submit the job to HPC platforms through interactive queue like `qsub -I`.
> The reason is that the prompt is exposed through the terminal. Use [Reloading Script](./reloading-script.md#reloading-script) or [Jupyter Notebook Access](./jupyter-notebook/jupyter-notebook-access.md#jupyter-notebook-access) to avoid this.

## Using Interactive Python Prompt

### Status Board
At the start of interactive mode, `libyt` prints the [status board](./libyt-defined-command.md#status-board) to show the current execution state of each inline function.

### Python Prompt and libyt Defined Commands
```
>>> 
```
It can run any Python statements or [**libyt defined commands**](./libyt-defined-command.md#libyt-defined-commands) here, including importing Python modules.
This is like using normal Python prompt, except that there are functions, objects, and simulation data already defined in it.
These statements run in inline script's namespace, and will update and make changes to it. 

> {octicon}`info;1em;sd-text-info;` To make interactive prompt more smoothly, set lower [YT_VERBOSE](../libyt-api/yt_initialize.md#yt-param-libyt).

:::
##### How does the prompt execute Python statements and libyt defined commands?
:::
1. MPI root process is the user interface, which takes user inputs.
2. MPI root process broadcasts the inputs to other processes.
3. Every other MPI process executes the same piece of input synchronously.
4. Print feedbacks to the terminal.

> {octicon}`alert;1em;sd-text-danger;` Changes is kept and maintain in user's inline script's namespace in situ analysis in the following round.

### Exit Interactive Python Prompt
Use [`%libyt exit`](./libyt-defined-command.md#exit) to exit the prompt.
Then it will return back to the simulation process.

## Known Limitations
- Cannot use arrow keys or `Crtl+D` in interactive mode.
- See [Limitations in MPI Related Python Tasks](./limitation.md#limitations-in-mpi-related-python-tasks).

## FAQs

### Why Can't I Find the Prompt `>>>`?
`>>> `  is probably immersed inside the output. 
We can hit enter again, which is to provide an empty statement, and it will come out. 

We can make prompt more smoothly by setting [YT_VERBOSE](../libyt-api/yt_initialize.md#yt-param-libyt) to YT_VERBOSE_INFO.

### Where Can I Use Interactive Mode?
`libyt` interactive Python prompt only works on local machine or submit the job to HPC platforms through interactive queue like `qsub -I`. 
The reason is that the user interface is exposed in the terminal.
