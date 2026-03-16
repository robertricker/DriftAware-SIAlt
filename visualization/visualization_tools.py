import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import os
import subprocess
from matplotlib.colors import ListedColormap
import cartopy.feature as cfeature


def visu_xarray(x, y, z, figsize, vmin, vmax, n_level, cmap, time_string, label, outfile, hem, iceconc=None):
    if hem == 'nh':
        crs = ccrs.LambertAzimuthalEqualArea(central_longitude=0.0, central_latitude=90.0, false_easting=0.0,
                                             false_northing=0.0)
    else:
        crs = ccrs.LambertAzimuthalEqualArea(central_longitude=0.0, central_latitude=-90.0, false_easting=0.0,
                                             false_northing=0.0)
    fig = plt.figure(figsize=figsize)

    xc, yc = np.meshgrid(x, y)
    fontsize = 18

    bounds = list(np.linspace(vmin, vmax, num=n_level))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
    ax = plt.subplot(projection=crs)
    if hem=='nh':
        ax.set_extent([-3850000, 3000000, -3000000, 3850000], crs=crs)
    else:
        ax.set_extent([-3700000, 3700000, -3150000, 4100000], crs=crs)
    if iceconc:
        if len(iceconc) > 0:
            iceconc["ice_conc"][iceconc["ice_conc"] < 15.0] = np.nan
            iceconc["ice_conc"][iceconc["ice_conc"] >= 15.0] = 1.0
            ax.pcolormesh(xc, yc, iceconc["ice_conc"], cmap=ListedColormap([(1, 1, 1)]))

    im = ax.pcolormesh(xc, yc, z, cmap=cmap, norm=norm)
    #ax.add_feature(cartopy.feature.OCEAN, facecolor=(0.86, 0.87, 0.9))
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', '50m',
                                             facecolor=(0.73, 0.74, 0.75), edgecolor='black')
    shelves = cfeature.NaturalEarthFeature('physical', 'antarctic_ice_shelves_polys', '50m',
                                             facecolor='lightgray', edgecolor='lightgray')
    ax.add_feature(shelves, linewidth=0.2)
    ax.add_feature(coastline, linewidth=0.2)
    ax.axis("off")
    plt.annotate(
        f"{time_string[0:4]}-{time_string[4:6]}-{time_string[6:8]}",
        xy=(0.01, 0.95),
        fontsize=fontsize,
        xycoords='axes fraction',
        color='black')

    lon = np.linspace(0, 2 * np.pi, 100) * 180 / np.pi
    if hem=='nh':
        ax.plot(lon, 88.0 * np.ones_like(lon), transform=ccrs.Geodetic(), linestyle='--', color='black', linewidth=1.0)
    else:
        ax.plot(lon, -88.0 * np.ones_like(lon), transform=ccrs.Geodetic(), linestyle='--', color='black', linewidth=1.0)
    cax = ax.inset_axes([0, -0.05, 1, 0.025], transform=ax.transAxes)
    cb = plt.colorbar(im, ax=ax, orientation='horizontal', cax=cax)
    cb.set_label(label=label, size=fontsize, color='black')
    cb.ax.tick_params(labelsize=fontsize, color='black', labelcolor='black')
    cb.ax.tick_params(which='both', length=0)
    cb.outline.set_linewidth(0)
    cb.ax.xaxis.set_major_locator(plt.MaxNLocator(2))
    fig.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=180)
    
    plt.close(fig)


def make_gif(out_dir, var):
    cmd_str = "convert -delay 10 -loop 0 *" + var + ".png " + var + ".gif"
    wd = os.getcwd()
    os.chdir(out_dir)
    subprocess.run(cmd_str, shell=True)
    os.chdir(wd)
