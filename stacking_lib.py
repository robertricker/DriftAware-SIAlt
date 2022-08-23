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
from read_icesat2.read_ICESat2_ATL10 import read_HDF5_ATL10
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
import pickle
import statistics as stat
import sys
import os

warnings.filterwarnings('ignore')
logging.disable(sys.maxsize)
warnings.filterwarnings("ignore")


def reproject(x, y, in_epsg, out_epsg):
    inProj = Proj(init=in_epsg)
    outProj = Proj(init=out_epsg)
    xout, yout = transform(inProj, outProj, x, y)
    return xout, yout


def getXY(pt):
    return (pt.x, pt.y)


def get_ice_concentration(filename):
    data = netCDF4.Dataset(filename[0])
    x, y = reproject(np.ma.getdata(data.variables['lon'][:, :]).flatten(),
                     np.ma.getdata(data.variables['lat'][:, :]).flatten(),
                     'epsg:4326', 'epsg:3413')

    value = np.ma.getdata(data.variables['ice_conc'][0, :, :]).flatten()
    grid_x, grid_y = np.meshgrid(np.ma.getdata(data.variables['xc'][:] * 1000.0),
                                 np.ma.getdata(data.variables['yc'][:] * 1000.0))
    coords = np.transpose(np.vstack((x, y)))

    value_i = griddata(coords, value, (grid_x, grid_y), method='nearest')
    value_i[value_i < 15] = 0

    return {"xc": grid_x, "yc": grid_y, "ice_conc": value_i}


def ice_concentration_correction(xc, yc, ice_conc, x ,y):
    f_ice_conc = interpolate.interp2d(xc, yc, ice_conc, kind='linear')
    ice_conc_proj = [f_ice_conc(x[i], y[i]) for i in range(len(x)) ]
    return np.asarray(ice_conc_proj)[:,0]/100.0


def get_ice_drift(filename, ice_conc):
    data = netCDF4.Dataset(filename[0])
    dx = np.ma.getdata(data.variables['dX'][0, :, :]).flatten()
    dy = np.ma.getdata(data.variables['dY'][0, :, :]).flatten()

    x, y = reproject(np.ma.getdata(data.variables['lon'][:, :]).flatten()[dx != -1e10],
                     np.ma.getdata(data.variables['lat'][:, :]).flatten()[dx != -1e10],
                     'epsg:4326', 'epsg:3413')

    dx = dx[dx != -1e10]
    dy = dy[dy != -1e10]

    coords = np.transpose(np.vstack((x, y)))

    dx_i = griddata(coords, dx, (ice_conc["xc"], ice_conc["yc"]), method='nearest')
    dy_i = griddata(coords, dy, (ice_conc["xc"], ice_conc["yc"]), method='nearest')

    dx_i[ice_conc["ice_conc"] == 0] = 0
    dy_i[ice_conc["ice_conc"] == 0] = 0

    return {"xc": ice_conc["xc"], "yc": ice_conc["yc"], "dx": dx_i, "dy": dy_i}


def drift_correction(xc, yc, dx, dy, x, y):
    # all in meters
    np.ma.getdata(dx)[np.ma.getdata(dx) == -1e+10] = 0
    np.ma.getdata(dy)[np.ma.getdata(dy) == -1e+10] = 0
    f_dx = interpolate.interp2d(xc, yc, dx, kind='linear')
    f_dy = interpolate.interp2d(xc, yc, dy, kind='linear')
    dx_proj = [f_dx(x[i], y[i]) for i in range(len(x))]
    dy_proj = [f_dy(x[i], y[i]) for i in range(len(y))]
    # output in m/h
    return np.asarray(dx_proj)[:, 0] / 48.0, np.asarray(dy_proj)[:, 0] / 48.0

def atl10_to_gdf(file_list):

    gdf_list = list()
    for file in file_list:
        IS2_atl10_mds, IS2_atl10_attrs, IS2_atl10_beams = read_HDF5_ATL10(file)
        beam_list = list()
        for beam in IS2_atl10_beams:
            tmp = pd.DataFrame.from_dict(IS2_atl10_mds[beam]['freeboard_beam_segment']['beam_freeboard'])
            tmp['beam'] = beam
            beam_list.append(tmp)

        df = pd.concat([df for df in beam_list]).pipe(gpd.GeoDataFrame)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
        gdf = gdf.to_crs(3413)
        gdf = gdf[(gdf['beam_fb_height'] < 10.0) &
                  (gdf['latitude'] > 50.0)]
        gdf_list.append(gdf)

    gdf_final = pd.concat([gdf for gdf in gdf_list]).pipe(gpd.GeoDataFrame)
    gdf_final.crs = gdf_list[0].crs

    return gdf_final.reset_index(drop=True)


def cs2l2p_to_gdf(filename):
    data = netCDF4.Dataset(filename[0])
    d = {
        'latitude': np.array(data["latitude"]),
        'longitude': np.array(data["longitude"]),
        'sea_ice_freeboard': np.array(data["sea_ice_freeboard"]),
        'sea_ice_thickness': np.array(data["sea_ice_thickness"]),
        'snow_depth': np.array(data["snow_depth"]),
        'time': np.array(data["time"])
        }
    df = pd.DataFrame(data=d)
    gdf = gpd.GeoDataFrame(df,geometry=gpd.points_from_xy(df.longitude, df.latitude),crs=4326)
    return gdf.to_crs(3413)


def concat_gdfs(gdf_array_index, master, row_lim, sensor):
    gdf_list = list()
    if sensor == 'is2':
        beams = np.array(['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'])
        for i in range(0, row_lim + 1):
            for beam in beams:
                if len(master[beam][str(i)][str(gdf_array_index)]) != 0:
                    gdf_list.append(master[beam][str(i)][str(gdf_array_index)])
                    del master[beam][str(i)][str(gdf_array_index)]
    if sensor == 'cs2':
        for i in range(0, row_lim + 1):
            if len(master[str(i)][str(gdf_array_index)]) != 0:
                gdf_list.append(master[str(i)][str(gdf_array_index)])
                del master[str(i)][str(gdf_array_index)]

    return pd.concat([gdf for gdf in gdf_list]).pipe(gpd.GeoDataFrame).reset_index(drop=True)


def display_scheme(scheme, beam, sensor):
    fig, (ax0) = plt.subplots(1, 1)
    if sensor == 'is2':
        ax0.pcolor(scheme[beam, :, :], edgecolors='white', linewidths=1)
    else:
        ax0.pcolor(scheme[:, :], edgecolors='white', linewidths=1)
    ax0.set_title('Temporal distribution of dataframes')
    plt.show()


def merge_forward_reverse_stacks(start_date, t_series_length, list_f, list_r, outdir):

    for j in range(0, t_series_length):
        cday = start_date + datetime.timedelta(days=j)
        subs = cday.strftime("%Y%m%d")
        print(subs)
        file_f = [i for i in list_f if subs in re.search('-(.+?)-*.geojson', os.path.basename(i)).group(1)]
        file_r = [i for i in list_r if subs in re.search('-(.+?)-*.geojson', os.path.basename(i)).group(1)]
        stack_f = gpd.read_file(file_f[0])
        stack_r = gpd.read_file(file_r[0])
        stack_r = stack_r[stack_r.t0 != stack_r.t1]
        frames = [stack_f, stack_r]
        result = pd.concat(frames)
        result = result.reset_index(drop=True)
        result.to_file(outdir + os.path.basename(file_f[0]).replace("-f-", "-"), driver="GeoJSON")
        os.remove(file_f[0])
        os.remove(file_r[0])


def stack_structure(sensor, t_window_length, t_series_length):
    days1 = np.arange(t_series_length).astype(str)
    days2 = days1 if t_series_length <= t_window_length else np.arange(t_window_length).astype(str)
    master = {}
    if sensor == 'is2':
        beams = np.array(['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'])
        for beam in beams:
            master[beam]={}
            for day2 in days2:
                master[beam][day2]={}
                for day1 in days1:
                    master[beam][day2][day1]={}
        scheme = np.zeros([len(beams),len(days2),len(days1)])

    else:
        for day2 in days2:
            master[day2]={}
            for day1 in days1:
                master[day2][day1]={}
        scheme = np.zeros([len(days2),len(days1)])

    return master, scheme


def baseline_proc(i, master, grid, scheme, prod, sensor, ice_conc):
    if sensor == 'is2':
        beams = np.array(['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'])
        for beam in prod.beam.unique():
            tmp = prod[['beam_fb_height', 'geometry', 'beam']].copy().loc[prod['beam'] == beam].drop(columns=['beam'])
            tmp = tmp.reset_index(drop=True)
            tmp_grid = gridding_lib.grid_data(tmp, grid, ['beam_fb_height'],['freeboard'])
            tmp_grid['xu'], tmp_grid['yu'] = tmp_grid.index.get_level_values('x'), tmp_grid.index.get_level_values('y')
            tmp_grid['t0'], tmp_grid['t1'] = i, i
            tmp_grid['beam'] = beam
            tmp = tmp_grid.copy().reset_index(drop=True)
            tmp = gpd.GeoDataFrame(tmp, geometry=gpd.points_from_xy(tmp_grid['xu'].values, tmp_grid['yu'].values),
                                   crs=3413)

            tmp["start_location_x"], tmp["start_location_y"] = tmp.geometry.x, tmp.geometry.y

            tmp["ice_conc"] = ice_concentration_correction(ice_conc["xc"][0, :], ice_conc["yc"][:, 0],
                                                           ice_conc['ice_conc'],
                                                           tmp['xu'].values, tmp['yu'].values)
            master[beam][str(0)][str(i)] = tmp
            scheme[(beams == beam).argmax(), 0, i] = 1
    else:
        tmp_grid = gridding_lib.grid_data(prod, grid, ['sea_ice_freeboard'],['freeboard'])
        tmp_grid['xu'], tmp_grid['yu'] = tmp_grid.index.get_level_values('x'), tmp_grid.index.get_level_values('y')
        tmp_grid['t0'], tmp_grid['t1'] = i, i
        tmp = tmp_grid.copy().reset_index(drop=True)
        tmp = gpd.GeoDataFrame(tmp, geometry=gpd.points_from_xy(tmp_grid['xu'].values, tmp_grid['yu'].values), crs=3413)

        tmp["ice_conc"] = ice_concentration_correction(ice_conc["xc"][0, :], ice_conc["yc"][:, 0],
                                                       ice_conc['ice_conc'],
                                                       tmp['xu'].values, tmp['yu'].values)
        master[str(0)][str(i)] = tmp
        scheme[0, i] = 1

    return master, scheme


def drift_proc(i, master, scheme, t_window_length, sensor, ice_conc, ice_drift, direct, range0):
    if sensor == 'is2':
        beams = np.array(['gt1l', 'gt1r', 'gt2l', 'gt2r', 'gt3l', 'gt3r'])
        m = 0
        for beam in beams:
            m = 0
            end = i + 2 if direct == 1 else range0 - i + 2
            for j in range(1, end):
                if j >= t_window_length: continue
                m = m + 1
                if len(master[beam][str(j - 1)][str(i)]) == 0: continue
                tmp_grid = master[beam][str(j - 1)][str(i)].copy().reset_index(drop=True)
                tmp_grid = apply_drift_correction(i, j, tmp_grid, ice_drift, ice_conc, direct)
                master[beam][str(j)][str(i + direct)] = tmp_grid
                scheme[(beams == beam).argmax(), j, i + direct] = 1
    else:
        m = 0
        end = i + 2 if direct == 1 else range0 - i + 2
        for j in range(1, end):
            if j >= t_window_length: continue
            m = m + 1
            if len(master[str(j - 1)][str(i)]) == 0: continue
            tmp_grid = master[str(j - 1)][str(i)].copy().reset_index(drop=True)
            tmp_grid = apply_drift_correction(i, j, tmp_grid, ice_drift, ice_conc, direct)
            master[str(j)][str(i + direct)] = tmp_grid
            scheme[j, i + direct] = 1

    return master, scheme, m


def apply_drift_correction(i, j, tmp_grid, ice_drift, ice_conc, direct):
    dt = 24  # time step in hours
    x_corr, y_corr = drift_correction(ice_drift["xc"][0, :], ice_drift["yc"][:, 0],
                                      ice_drift['dx'] * 1000.0, ice_drift['dy'] * 1000.0,
                                      tmp_grid['xu'].values, tmp_grid['yu'].values)
    if direct == 1:
        xu = tmp_grid['xu'].values + (x_corr * dt)
        yu = tmp_grid['yu'].values + (y_corr * dt)
        tt = i + direct - j
    else:
        xu = tmp_grid['xu'].values - (x_corr * dt)
        yu = tmp_grid['yu'].values - (y_corr * dt)
        tt = i + direct + j

    tmp_grid = gpd.GeoDataFrame(tmp_grid, geometry=gpd.points_from_xy(xu, yu), crs=3413)
    tmp_grid['xu'], tmp_grid['yu'] = xu, yu
    tmp_grid['t0'], tmp_grid['t1'] = tt, i + direct

    tmp_grid["ice_conc"] = ice_concentration_correction(ice_conc["xc"][0, :], ice_conc["yc"][:, 0],
                                                        ice_conc['ice_conc'],
                                                        tmp_grid['xu'].values, tmp_grid['yu'].values)
    return tmp_grid[tmp_grid["ice_conc"] > 0.15]


def stack_proc(t_start, t_window_length, t_series_length, hem, sensor,
               product_list, osi405_list, osi430b_list, direct, grid, outdir):
    if direct == 1:
        range0 = 0
        range1 = t_series_length
        direct_str = 'f'
    else:
        range0 = t_series_length - 1
        range1 = -1
        direct_str = 'r'

    if sensor == 'is2':
        date_str = r"_((\d+)_(\d+))"
        date_pattern = '%Y%m%d%H%M%S'
        add_str = ''
        match_group = 2
        name_str = [os.path.basename(product_list[0]).split("_")[0][0:5],
                    'v'+os.path.basename(product_list[0]).split("_")[3]]
    else:
        date_str = r"-(\d+)-"
        date_pattern = '%Y%m%d%H%M'
        add_str = '1200'
        match_group = 1
        name_str = ['cs2-' + os.path.basename(product_list[0]).split("-")[2],
                    os.path.basename(product_list[0]).split("-")[8][1:5]]

    dt1d = datetime.timedelta(days=1)

    master, scheme = stack_structure(sensor, t_window_length, t_series_length)

    for i in range(range0, range1, direct):

        date_list = list()
        t0 = t_start + datetime.timedelta(days=i)
        t1 = t_start + datetime.timedelta(days=i + 1)
        print('Day: ' + t0.strftime("%d-%m-%Y"))

        for file in product_list:
            match = re.search(date_str, file)
            date_list.append(datetime.datetime.strptime(match.group(match_group) + add_str, date_pattern))
        file_sel = [product_list[date_list.index(d)] for d in date_list if d > t0 and d < t1]

        osi430b_date = list()
        for file in osi430b_list:
            match = re.search(r"_(\d+)", file)
            osi430b_date.append(datetime.datetime.strptime(match.group(1), '%Y%m%d%H%M'))
        osi430b_sel = [osi430b_list[osi430b_date.index(d)] for d in osi430b_date if d > t0 and d < t1]
        if len(osi430b_sel) == 0:
            while len(osi430b_sel) == 0:
                osi430b_sel = [osi430b_list[osi430b_date.index(d)] for d in osi430b_date if d > t0 and d < t1]
                t0 = t0 - dt1d
                t1 = t1 - dt1d
        print(osi430b_sel)
        t0 = t_start + datetime.timedelta(days=i)
        t1 = t_start + datetime.timedelta(days=i + 1)

        osi430b_sel_f = [osi430b_list[osi430b_date.index(d)] for d in osi430b_date if d > t0 + direct*dt1d and d < t1 + direct*dt1d]
        if len(osi430b_sel_f) == 0:
            while len(osi430b_sel_f) == 0:
                osi430b_sel_f = [osi430b_list[osi430b_date.index(d)] for d in osi430b_date if d > t0 + direct*dt1d and d < t1 + direct*dt1d]
                t0 = t0 - dt1d
                t1 = t1 - dt1d
        print(osi430b_sel_f)
        t0 = t_start + datetime.timedelta(days=i)
        t1 = t_start + datetime.timedelta(days=i + 1)

        osi405_date = list()
        for file in osi405_list:
            match = re.search(r"_((\d+)-(\d+))", file)
            osi405_date.append(datetime.datetime.strptime(match.group(2), '%Y%m%d%H%M') + dt1d)
        osi405_sel = [osi405_list[osi405_date.index(d)] for d in osi405_date if d > t0 and d < t1]
        print(osi405_sel)

        if file_sel and osi430b_sel and osi405_sel:
            ice_conc = get_ice_concentration(osi430b_sel)
            if sensor == 'is2':
                prod = atl10_to_gdf(file_sel)
            else:
                prod = cs2l2p_to_gdf(file_sel)
            master, scheme = baseline_proc(i, master, grid, scheme, prod, sensor, ice_conc)

        if osi430b_sel and osi430b_sel_f and osi405_sel:
            ice_conc = get_ice_concentration(osi430b_sel)
            ice_conc_f = get_ice_concentration(osi430b_sel_f)
            ice_drift = get_ice_drift(osi405_sel, ice_conc)
            if (direct == -1 and i > 0) or (direct == 1 and i < t_series_length - 1):
                master, scheme, m = drift_proc(i, master, scheme, t_window_length, sensor,
                                               ice_conc_f, ice_drift, direct, range0)

        gdf_final = concat_gdfs(i, master, m, sensor)
        gdf_final = gdf_final.drop(columns=['xu', 'yu'])

        outfile = (name_str[0] + "-" +
                   hem + "-" +
                   t0.strftime("%Y%m%d") + "-" +
                   name_str[1] + "-" +
                   direct_str + "-" +
                   "{:.0f}".format(grid.geometry[0].length / 4 / 100.0) + '.geojson')

        print("outfile: " + outfile)
        gdf_final.to_file(outdir + outfile, driver="GeoJSON")

    return scheme
