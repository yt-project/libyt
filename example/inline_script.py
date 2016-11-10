import yt

def yt_inline():
    ds = yt.frontends.libyt.libytDataset()
    sz = yt.SlicePlot( ds, 'z', 'density', center='c' )

    sz.set_zlim( 'density', 1.0e0, 1.0e6 )
    sz.annotate_grids( periodic=False )
    sz.save()

