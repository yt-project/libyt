import yt
from mpi4py import MPI
import pandas as pd
import numpy as np

yt.enable_parallelism()
step = 0
comm = MPI.COMM_WORLD
myrank = comm.Get_rank()

class DataIOTestFailed(Exception):
    """
    Raised when the grid data read by libyt DataIO and
    grid data in C++ array in a cell differ more than
    1e-6.
    """
    criteria = 1e-9

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
    slc = yt.SlicePlot(ds, "z", ("gamer", "InvDens"))

    if yt.is_root():
        slc.save()

def test_function():
    global step

    ds = yt.frontends.libyt.libytDataset()

    for gid in range(ds.index.num_grids):
        # Read simulation data, this is the original data in [x][y][z] orientation.
        filepath = "./data/Dens_grid{}_step{}.txt".format("%d" % gid, "%d" % step)
        df = pd.read_csv(filepath, header=None)
        sim_data = df.to_numpy()
        dimensions = ds.index.grid_dimensions[gid]
        sim_data = sim_data.reshape(dimensions).swapaxes(0, 2)

        # Get data from libyt DataIO.
        data_io = ds.index.grids[gid][("gamer", "Dens")]
        data_io = np.asarray(data_io)

        # Compare them, if bigger than criteria, raise an error
        diff = np.sum(np.absolute(sim_data - data_io)) / (dimensions[0] * dimensions[1] * dimensions[2])
        if diff > DataIOTestFailed.criteria:
            err_msg = "On MPI rank {}, step {}, density grid (id={}) is different from simulation data {}.\n" \
                      "DataIOTest FAILED.\n".format("%d" % myrank, "%d" % step, "%d" % gid, "%.10e" % diff)
            raise DataIOTestFailed(err_msg)

    step += 1

