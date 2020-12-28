import yt

yt.enable_parallelism()

def yt_inline():
    ds = yt.frontends.libyt.libytDataset()
    # ProjectionPlot, Serial
    ######################################
    prjz = yt.ProjectionPlot(ds, 'z', 'density')

    if yt.is_root():
        prjz.save()
    # SlicePlot, Serial
    ######################################
    #sz = yt.SlicePlot( ds, 'z', 'Dens', center='c' )
    #sz.set_unit( 'Dens', 'msun/kpc**3' )
    #sz.set_zlim( 'Dens', 1.0e0, 1.0e6 )
    #sz.annotate_grids( periodic=False )
    #sz.save()

