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
    
def yt_inline_ParticlePlot():
    # YT Particle Plot does not support parallelism for now
    # So we run mpirun -np 1
    ds = yt.frontends.libyt.libytDataset()
    
    par = yt.ParticleProjectionPlot(ds, "z")

    par.save()
        profile.save()

def test_user_parameter():
    import libyt
    print("user_int = ", libyt.param_user['user_int'])
