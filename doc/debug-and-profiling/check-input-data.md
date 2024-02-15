# Checking Input Data

## How to Configure
Set `check_data` to `true` when initializing `libyt` (see [Initialize]({% link libytAPI/Initialize.md %}#yt_initialize)):
```c++
yt_param_libyt param_libyt;
param_libyt.verbose = YT_VERBOSE_INFO;
param_libyt.script  = "inline_script";
param_libyt.check_data = true;

/* Initialize libyt, should only be called once. */
if( yt_initialize( argc, argv, &param_libyt ) != YT_SUCCESS ){
    // error message
    exit( EXIT_FAILURE );
}
```

## Data Checked by libyt

> :information_source: The test is specific for adaptive mesh refinement data structure.

If they all pass, then `libyt` is guarantee to work:
- Check if number of grids ([`num_grids`]({% link libytAPI/SetYTParameter.md %}#yt_param_yt)) matches the aggregated sum of ([`num_grids_local`]({% link libytAPI/SetYTParameter.md %}#yt_param_yt)).
- Check if field list is set properly. (See [Set Fields Information]({% link libytAPI/FieldInfo/SetFieldsInformation.md %}#set-fields-information))
  - Field name, type and data type are set.
  - User-defined derived function is set if field type is `"derived_func"`.
  - Ghost cell is larger than or equal to 0.
  - Each field name is unique.
- Check if particle list is set properly. (See [Set Particles Information]({% link libytAPI/SetParticlesInformation.md %}#set-particles-information))
  - Particle type, attribute list, and coordinate name are set.
  - Each attribute name in a particle type is unique.
  - Each particle type name is unique, and it is not the same as frontend name.
- Check if local grids information is set properly. (See [Set Local Grids Information]({% link libytAPI/SetLocalGridsInformation.md %}#set-local-grids-information))
  - Grid left/right edge, dimensions, id, parent id, level are properly set, with root level starts at 0, dimensions are all greater than 0, and all ids are unique.
  - Parent id is negative if it is at root level. And parent grid has child grid level minus 1.
  - Child grid edge is inside parent grid edge.
  - Grid left/right edge are in domain edge, and grid left edge is smaller than or equal to grid right edge.
  - Grid data is properly set.
