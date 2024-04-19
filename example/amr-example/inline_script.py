import yt_libyt
import yt

# Must include this line, if you are running in parallel.
yt.enable_parallelism()

def yt_inline_ProjectionPlot( fields ):
    # Load the data, just like using yt.load()
    ds = yt_libyt.libytDataset()
    
    # Do yt operation
    prjz = yt.ProjectionPlot(ds, 'z', fields)
    
    # Save figure only on root process
    if yt.is_root():
        prjz.save()

def yt_inline_ProfilePlot():
    ds = yt_libyt.libytDataset()
    profile = yt.ProfilePlot(ds, "x", ["density"])
    
    if yt.is_root():
        profile.save()
    
def yt_inline_ParticlePlot():
    ds = yt_libyt.libytDataset()
    par = yt.ParticlePlot(ds, "particle_position_x", "particle_position_y", "Level", center = 'c')

    if yt.is_root():
        par.save()

def yt_derived_field_demo():
    ds = yt_libyt.libytDataset()
    slc = yt.SlicePlot(ds, "z", ("gamer", "InvDens"))
    
    if yt.is_root():
        slc.save()

def test_function():
    pass
