import yt

yt.enable_parallelism()

def yt_inline_ProjectionPlot():
    ds = yt.frontends.libyt.libytDataset()
    prjz = yt.ProjectionPlot(ds, 'z', 'density')

    if yt.is_root():
        prjz.save()
    
def yt_inline_ProfilePlot():
    ds = yt.frontends.libyt.libytDataset()
    profile = yt.ProfilePlot(ds, "x", ["density"])

    if yt.is_root():
        profile.save()

