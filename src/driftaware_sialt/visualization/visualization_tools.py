import os
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def _projection_and_extent(hem):
    if hem == 'nh':
        central_latitude = 90.0
        extent = [-4150000, 3000000, -3150000, 4000000]
    else:
        central_latitude = -90.0
        extent = [-3450000, 3850000, -3200000, 4100000]

    projection = ccrs.LambertAzimuthalEqualArea(
        central_longitude=0.0,
        central_latitude=central_latitude,
        false_easting=0.0,
        false_northing=0.0,
    )
    return projection, extent


def _add_map_features(ax):
    ax.set_facecolor("white")
    coastline = cfeature.NaturalEarthFeature(
        'physical',
        'coastline',
        '50m',
        facecolor=(0.95, 0.95, 0.95),
        edgecolor='dimgrey',
    )
    shelves = cfeature.NaturalEarthFeature(
        'physical',
        'antarctic_ice_shelves_polys',
        '50m',
        facecolor='lightgray',
        edgecolor='lightgray',
    )
    ax.add_feature(shelves, linewidth=0.4)
    ax.add_feature(coastline, linewidth=0.4)


def _add_date_annotation(ax, time_string, fontsize):
    plt.annotate(
        f"{time_string[0:4]}-{time_string[4:6]}-{time_string[6:8]}",
        xy=(0.0, 0.0),
        fontsize=fontsize,
        xycoords='axes fraction',
        color='black',
    )


def _add_colorbar(ax, image, label, fontsize):
    cax = ax.inset_axes([0, -0.08, 1, 0.035], transform=ax.transAxes)
    cb = plt.colorbar(image, ax=ax, orientation='horizontal', cax=cax)
    cb.set_label(label=label, size=fontsize, color='black')
    cb.ax.tick_params(labelsize=fontsize, color='black', labelcolor='black')
    cb.ax.tick_params(which='both', length=0)
    cb.outline.set_linewidth(0)
    cb.ax.xaxis.set_major_locator(plt.MaxNLocator(2))


def visu_xarray(
    x,
    y,
    z,
    figsize,
    vmin,
    vmax,
    n_level,
    cmap,
    time_string,
    label,
    outfile,
    hem,
):
    """Plot a gridded variable on a polar map and save it to ``outfile``."""
    projection, extent = _projection_and_extent(hem)
    fig = plt.figure(figsize=figsize)
    ax = plt.subplot(projection=projection)
    ax.set_extent(extent, crs=projection)

    xc, yc = np.meshgrid(x, y)
    bounds = list(np.linspace(vmin, vmax, num=n_level))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend='neither')

    image = ax.pcolormesh(xc, yc, z, cmap=cmap, norm=norm)
    _add_map_features(ax)
    ax.axis("off")

    fontsize = 16
    _add_date_annotation(ax, time_string, fontsize)
    _add_colorbar(ax, image, label, fontsize)

    fig.savefig(outfile, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close(fig)


def make_gif(out_dir, var):
    cmd_str = "convert -delay 10 -loop 0 *" + var + ".png " + var + ".gif"
    wd = os.getcwd()
    os.chdir(out_dir)
    subprocess.run(cmd_str, shell=True)
    os.chdir(wd)
