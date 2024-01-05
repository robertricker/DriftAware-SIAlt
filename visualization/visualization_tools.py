import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib
import os
import subprocess


def visu_xarray(x, y, z, figsize, vmin, vmax, n_level, cmap, time_string, label, outfile, iceconc=None):
    crs = ccrs.LambertAzimuthalEqualArea(central_longitude=0.0, central_latitude=90.0, false_easting=0.0,
                                         false_northing=0.0)
    fig = plt.figure(figsize=figsize)

    xc, yc = np.meshgrid(x, y)

    bounds = list(np.linspace(vmin, vmax, num=n_level))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
    ax = plt.subplot(projection=crs)

    ax.set_extent([-3000000, 3000000, -3000000, 3000000], crs=crs)
    if iceconc:
        if len(iceconc) > 0:
            iceconc["ice_conc"][iceconc["ice_conc"] < 15.0] = np.nan
            iceconc["ice_conc"][iceconc["ice_conc"] >= 15.0] = 1.0
            ax.pcolormesh(xc, -yc, iceconc["ice_conc"], cmap='Greys', vmin=1, vmax=2, alpha=0.4)

    im = ax.pcolormesh(xc, yc, z, cmap=cmap, norm=norm)
    ax.add_feature(cartopy.feature.OCEAN, facecolor=[(0.16, 0.22, 0.33)])
    ax.add_feature(cartopy.feature.LAND, facecolor=[(0.05, 0.07, 0.14)], zorder=2)
    ax.coastlines(linewidth=0.15, color='white', zorder=3)
    ax.axis("off")
    cax = ax.inset_axes([0.51, 0.93, 0.45, 0.02], transform=ax.transAxes)
    cb = plt.colorbar(im, ax=ax, shrink=0.7, orientation='horizontal', cax=cax)
    plt.annotate(time_string[6:8] + '.' + time_string[4:6] + '.' + time_string[0:4],
                 xy=(0.51, 0.965), fontsize=12, xycoords='axes fraction', color='white')

    cb.set_label(label=label, size=12, color='white')
    cb.ax.tick_params(labelsize=12, color='white', labelcolor='white')
    cb.outline.set_linewidth(0)
    plt.close(fig)
    fig.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=120)


def make_gif(out_dir, var):
    cmd_str = "convert -delay 10 -loop 0 *" + var + ".png " + var + ".gif"
    wd = os.getcwd()
    os.chdir(out_dir)
    subprocess.run(cmd_str, shell=True)
    os.chdir(wd)
