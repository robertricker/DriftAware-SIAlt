import datetime
import stacking_lib
import gridding_lib
import pandas as pd
import geopandas as gpd
import shapely as shp
import numpy as np
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.interpolate import griddata
import warnings
import cartopy.crs as ccrs
import cartopy
import glob
import logging
import sys
from datetime import date
import datetime
from multiprocessing.pool import Pool
from astropy.time import Time
from tqdm.notebook import tqdm
from pyproj import Proj, transform
import gps_time
import re
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib
import statistics as stat
import sys
import os


def visu_xarray(x,y,z,figsize,vmin,vmax,n_level,cmap,time_string,label,outfile):
    fig = plt.figure(figsize=figsize)
    xc,yc = np.meshgrid(x,y)
    bounds = list(np.linspace(vmin,vmax,num=n_level))
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N, extend='neither')
    ax = plt.subplot(projection=ccrs.NorthPolarStereo(central_longitude=-45))
    ax.set_extent([0, 180, 50, 90])
    ax.set_facecolor("aliceblue")
    im = ax.pcolormesh(xc, yc, z,cmap=cmap,vmin=vmin,vmax=vmax,norm=norm)
    ax.coastlines(linewidth=0.2)
    ax.add_feature(cartopy.feature.LAND)
    ax.outline_patch.set_edgecolor('white')
    gl=ax.gridlines(draw_labels=False,linewidth=0.25,x_inline=False, y_inline=False)
    gl.xlocator = mticker.FixedLocator(np.arange(360/30)*30-180)

    cax = ax.inset_axes([0.4, 0.93, 0.55, 0.02], transform=ax.transAxes)
    cb = plt.colorbar(im,ax=ax,shrink=0.7,orientation='horizontal',cax=cax)
    plt.annotate(time_string[6:8]+'.'+time_string[4:6]+'.'+time_string[0:4],
                 xy=(0.4, 0.965),fontsize=12, xycoords='axes fraction')

    cb.set_label(label=label,size=12)
    cb.ax.tick_params(labelsize=12)
    cb.outline.set_linewidth(0)
    plt.close(fig)
    fig.savefig(outfile, bbox_inches='tight')


def generate_plots(config):
    if config["options"]["sensor"] == 'is2':
        source_list = sorted(glob.glob(config["dir"]["is2_netcdf"] + "/" + '*.nc'))
        out_dir = config["dir"]["is2_plots"]
    else:
        source_list = sorted(glob.glob(config["dir"]["cs2_netcdf"] + "/" + '*.nc'))
        out_dir = config["dir"]["cs2_plots"]

    var = config["options"]["plot_var"]
    figsize = (8,8)

    for file in source_list:
        print(file)
        time_string = re.search('nh-(.+?)-(.*).nc', os.path.basename(file)).group(1)
        data  = xr.open_dataset(file,decode_times=False)

        if var == 'laser_freeboard':
            vmin, vmax, n_level = 0,60,13
            cmap = plt.cm.plasma
            scaling = 100.0
            label = 'Total freeboard in cm'

        elif var == "dist_acquisition":
            vmin, vmax, n_level = 0,150,16
            cmap = plt.cm.plasma
            scaling = 1.0
            label = 'Distance to data aquisition in km'

        elif var== "time_offset_acquisition":
            vmin, vmax, n_level = 0,15,16
            cmap = plt.cm.plasma
            scaling = 1.0
            label = 'time offfset to data aquisition in days'
        else:
            break

        outfile = out_dir + re.split('.nc', os.path.basename(file))[0] + '_' + var + '.png'

        visu_xarray(data.xc,data.yc,data[var]*scaling,
                    figsize,
                    vmin, vmax, n_level,
                    cmap,
                    time_string,
                    label,
                    outfile)






