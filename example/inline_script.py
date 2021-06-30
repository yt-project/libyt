import yt

yt.enable_parallelism()

def yt_inline_ProjectionPlot( fields ):
    ds = yt.frontends.libyt.libytDataset()
    prjz = yt.ProjectionPlot(ds, 'z', fields)

    if yt.is_root():
        prjz.save()
    
def yt_inline_ProfilePlot():
    ds = yt.frontends.libyt.libytDataset()
    profile = yt.ProfilePlot(ds, "x", ["density"])

    if yt.is_root():
        profile.save()

def test_user_parameter():
    import libyt
    print("user_int = ", libyt.param_user['user_int'])
