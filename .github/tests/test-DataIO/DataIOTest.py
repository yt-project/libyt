import yt

yt.enable_parallelism()
step = 0

def yt_inline_ProjectionPlot( fields ):
    # Load the data, just like using yt.load()
    ds = yt.frontends.libyt.libytDataset()

    # Do yt operation
    prjz = yt.ProjectionPlot(ds, 'z', fields)

    # Include this line, otherwise yt will save one copy in each rank.
    if yt.is_root():
        prjz.save()

def yt_inline_ProfilePlot():
    ds = yt.frontends.libyt.libytDataset()
    profile = yt.ProfilePlot(ds, "x", ["density"])
    if yt.is_root():
        profile.save()

def yt_inline_ParticlePlot():
    ds = yt.frontends.libyt.libytDataset()

    ## ParticleProjectionPlot
    #==========================
    # par = yt.ParticleProjectionPlot(ds, "z")

    ## ParticlePlot
    #==========================
    par = yt.ParticlePlot(ds, "particle_position_x", "particle_position_y", "Level", center = 'c')
    if yt.is_root():
        par.save()

def yt_derived_field_demo():
    ds = yt.frontends.libyt.libytDataset()
    slc1 = yt.SlicePlot(ds, "z", ("gamer", "level_derived_func"))
    slc2 = yt.SlicePlot(ds, "z", ("gamer", "level_derived_func_with_name"))

    if yt.is_root():
        slc1.save()
        slc2.save()

def test_function():
    import pandas as pd
    import numpy as np
    import libyt
    from mpi4py import MPI

    global step
    comm = MPI.COMM_WORLD
    myrank = comm.Get_rank()
    ds = yt.frontends.libyt.libytDataset()
    dimensions = [8, 8, 8]

    for gid in range(libyt.param_yt["num_grids"]):
        # Get data from control group, this is the original data with no ghost-cell.
        filepath = "../.github/data/step" + str(step) + "/Dens_grid" + str(gid) + ".txt"
        df = pd.read_csv(filepath, header=None)
        control_grid = df.to_numpy()
        control_grid = np.reshape(control_grid, (dimensions[0], dimensions[1], dimensions[2]))
        control_grid = np.swapaxes(control_grid, 0, 2)

        # Get data from libyt DataIO, the grid does not contain ghost-cell.
        grid = ds.index.grids[gid]["gamer", "Dens"]
        grid = np.asarray(grid)

        # Compare them and print the result in file
        with open("MPI" + str(myrank) + "_result.txt", "a") as f:
            errors = np.sum(control_grid - grid) / (dimensions[0] * dimensions[1] * dimensions[2])
            f.write(str(errors) + "\n")

    step += 1

