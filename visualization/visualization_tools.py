import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib
import os
import subprocess
from matplotlib.colors import ListedColormap


def visu_xarray(x, y, z, figsize, vmin, vmax, n_level, cmap, time_string, label, outfile, iceconc=None):
    crs = ccrs.LambertAzimuthalEqualArea(central_longitude=0.0, central_latitude=90.0, false_easting=0.0,
                                         false_northing=0.0)
    fig = plt.figure(figsize=figsize)

    xc, yc = np.meshgrid(x, y)

    bounds = list(np.linspace(vmin, vmax, num=n_level))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
    ax = plt.subplot(projection=crs)
    ax.set_extent([-3800000, 3000000, -3000000, 3800000], crs=crs)
    if iceconc:
        if len(iceconc) > 0:
            iceconc["ice_conc"][iceconc["ice_conc"] < 15.0] = np.nan
            iceconc["ice_conc"][iceconc["ice_conc"] >= 15.0] = 1.0
            ax.pcolormesh(xc, -yc, iceconc["ice_conc"], cmap=ListedColormap([(1,1,1)]))

    im = ax.pcolormesh(xc, yc, z, cmap=cmap, norm=norm)
    # ax.add_feature(cartopy.feature.OCEAN, facecolor=[(0.16, 0.22, 0.33)])
    # ax.add_feature(cartopy.feature.LAND, facecolor=[(0.05, 0.07, 0.14)], zorder=2)
    ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.86, 0.87, 0.9))
    ax.add_feature(cartopy.feature.LAND, facecolor=(0.73, 0.74, 0.75), zorder=2)
    ax.coastlines(linewidth=0.15, color='black', zorder=3)
    ax.axis("off")
    cax = ax.inset_axes([0.51, 0.93, 0.45, 0.02], transform=ax.transAxes)
    cb = plt.colorbar(im, ax=ax, shrink=0.7, orientation='horizontal', cax=cax)
    # plt.annotate(time_string[6:8] + '.' + time_string[4:6] + '.' + time_string[0:4],
    #              xy=(0.51, 0.965), fontsize=12, xycoords='axes fraction', color='white')
    plt.annotate(time_string[6:8] + '.' + time_string[4:6] + '.' + time_string[0:4],
                 xy=(0.51, 0.965), fontsize=12, xycoords='axes fraction', color='black')

    # lon = np.linspace(0, 2 * np.pi, 100) * 180 / np.pi
    # ax.plot(lon, 88.0 * np.ones_like(lon), transform=ccrs.Geodetic(), linestyle='--', color='black', linewidth=1.0)

    # cb.set_label(label=label, size=12, color='white')
    # cb.ax.tick_params(labelsize=12, color='white', labelcolor='white')
    cb.set_label(label=label, size=12, color='black')
    cb.ax.tick_params(labelsize=12, color='black', labelcolor='black')
    cb.outline.set_linewidth(0)
    max_ticks = 2
    cb.ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
    plt.close(fig)
    fig.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=180)


def make_gif(out_dir, var):
    cmd_str = "convert -delay 10 -loop 0 *" + var + ".png " + var + ".gif"
    wd = os.getcwd()
    os.chdir(out_dir)
    subprocess.run(cmd_str, shell=True)
    os.chdir(wd)
