import yt

# Must include this line, if you are running in parallel.
yt.enable_parallelism()

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
    pass
