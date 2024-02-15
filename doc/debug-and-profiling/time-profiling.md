# Time Profiling

## How to Configure

Compile `libyt` with `-DSUPPORT_TIMER` option. (See [How to Install]({% link HowToInstall.md %}#options))

## Visualizing the Profile -- Chrome Tracing
1. Since each process dumps its profile `libytTimeProfile_MPI*.json` separately, we run the following to concatenate all of them:
   ```bash
   cat `ls libytTimeProfile_MPI*` >> TimeProfile.json
   ```
2. Add `]}` at the end of `TimeProfile.json`.
   > :information_source: This is optional, and we only need to add `]}` if [Perfetto](https://ui.perfetto.dev/) doesn't recognize the file.
3. Open Google Chrome and enter `chrome://tracing`, or go [Perfetto](https://ui.perfetto.dev/).
4. Load the time profile `TimeProfile.json`.
   
   ![](../_static/img/TracingTimeProfile.png)
