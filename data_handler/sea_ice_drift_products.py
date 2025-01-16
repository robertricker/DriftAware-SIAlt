import netCDF4
import numpy as np
from scipy.interpolate import griddata
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import sobel
from data_handler.sea_ice_concentration_products import SeaIceConcentrationProducts
from io_tools import transform_coords
import datetime
import sys
import os
import re


class SeaIceDriftProducts(SeaIceConcentrationProducts):
    def __init__(self, **kwargs):

        super().__init__()

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.ice_drift = None

        self.function_map = {
            'osi405': self.get_ice_drift,
            'osi455': self.get_ice_drift
        }

        self.config = {
            'osi405': {
                'hem_nh': '_nh_',
                'hem_sh': '_sh_',
                'date_str': '{12}',
                'date_pt': '%Y%m%d%H%M',
                'ref_time': datetime.datetime(1978, 1, 1, 0, 0, 0),
                'ref_daytime_corr': datetime.timedelta(days=1).total_seconds(),
                'time_span': 48,
                'date_offset': datetime.timedelta(days=1)
            },
            'osi455': {
                'hem_nh': '_nh_',
                'hem_sh': '_sh_',
                'date_str': '{12}',
                'date_pt': '%Y%m%d%H%M',
                'ref_time': datetime.datetime(1978, 1, 1, 0, 0, 0),
                'ref_daytime_corr': datetime.timedelta(days=0).total_seconds(),
                'time_span': 24,
                'date_offset': datetime.timedelta(days=0)
            }
        }

    def get_ice_drift(self, target_files, ice_conc):
        epoch = datetime.datetime(1970, 1, 1, 0, 0, 0)
        ref_time = self.config[self.product_id]['ref_time']
        data = netCDF4.Dataset(target_files)
        dx_dy_unc = np.ma.getdata(data.variables['uncert_dX_and_dY'][0, :, :]).flatten()

        x0, y0 = transform_coords(np.ma.getdata(data.variables['lon'][:, :]).flatten()[dx_dy_unc != -1e10],
                                  np.ma.getdata(data.variables['lat'][:, :]).flatten()[dx_dy_unc != -1e10],
                                  'epsg:4326', self.out_epsg)

        x1, y1 = transform_coords(np.ma.getdata(data.variables['lon1'][:, :]).flatten()[dx_dy_unc != -1e10],
                                  np.ma.getdata(data.variables['lat1'][:, :]).flatten()[dx_dy_unc != -1e10],
                                  'epsg:4326', self.out_epsg)

        time_bnds = np.ma.getdata(data.variables['time_bnds']).data.flatten() + (ref_time - epoch).total_seconds()
        time_bnds[0] = time_bnds[0] + self.config[self.product_id]['ref_daytime_corr']

        dx = x1 - x0
        dy = y1 - y0

        coords = np.transpose(np.vstack((x0, y0)))

        xc, yc = ice_conc["xc"], ice_conc["yc"]
        ice_conc = ice_conc["ice_conc"]

        dx_i = griddata(coords, dx, (xc, yc), method='linear')
        dy_i = griddata(coords, dy, (xc, yc), method='linear')
        dx_dy_unc_i = griddata(coords, dx_dy_unc[dx_dy_unc != -1e10], (xc, yc), method='linear')

        dx_fill = griddata(coords, dx, (xc, yc), method='nearest')
        dy_fill = griddata(coords, dy, (xc, yc), method='nearest')
        dx_dy_unc_fill = griddata(coords, dx_dy_unc[dx_dy_unc != -1e10], (xc, yc), method='nearest')

        invalid = np.isnan(dx_i) | np.isnan(dy_i)
        dx_i[invalid] = dx_fill[invalid]
        dy_i[invalid] = dy_fill[invalid]
        dx_dy_unc_i[invalid] = dx_dy_unc_fill[invalid]

        dx_i[ice_conc == 0] = 0
        dy_i[ice_conc == 0] = 0
        dx_dy_unc_i[ice_conc == 0] = 0

        self.ice_drift = {"xc": xc, "yc": yc, "dx": dx_i, "dy": dy_i, "dx_dy_unc": dx_dy_unc_i*1000.0,
                          "time_bnds": time_bnds}

    def drift_correction(self, x, y):
        time_span = self.config[self.product_id]['time_span']
        xc, yc = self.ice_drift["xc"][0, :], self.ice_drift["yc"][:, 0]
        dx, dy = self.ice_drift['dx'], self.ice_drift['dy']
        dx_dy_unc = self.ice_drift['dx_dy_unc']
        np.ma.getdata(dx)[np.ma.getdata(dx) == -1e+10] = 0
        np.ma.getdata(dy)[np.ma.getdata(dy) == -1e+10] = 0
        # infinite values can occure in drift files
        np.ma.getdata(dx)[np.ma.getdata(dx) == np.inf] = 0
        np.ma.getdata(dy)[np.ma.getdata(dy) == np.inf] = 0
        # Check if xc and yc are in descending order
        if xc[0] > xc[-1]:
            xc = xc[::-1]
            dx = dx[:, ::-1]
            dy = dy[:, ::-1]
            dx_dy_unc = dx_dy_unc[:, ::-1]
        if yc[0] > yc[-1]:
            yc = yc[::-1]
            dx = dx[::-1, :]
            dy = dy[::-1, :]
            dx_dy_unc = dx_dy_unc[::-1, :]

        f_dx = RegularGridInterpolator((xc, yc), dx.T, method='linear')
        f_dy = RegularGridInterpolator((xc, yc), dy.T, method='linear')
        f_dx_dy_unc = RegularGridInterpolator((xc, yc), dx_dy_unc.T, method='linear')

        dx_proj = f_dx((x, y))
        dy_proj = f_dy((x, y))
        dx_dy_unc_proj = f_dx_dy_unc((x, y))

        # output in m/h
        return dx_proj.flatten() / time_span, dy_proj.flatten() / time_span, dx_dy_unc_proj / time_span

    def deformation(self, x, y):
        time_span = self.config[self.product_id]['time_span']
        xc, yc = self.ice_drift["xc"][0, :], self.ice_drift["yc"][:, 0]
        dx, dy = self.ice_drift['dx'], self.ice_drift['dy']
        np.ma.getdata(dx)[np.ma.getdata(dx) == -1e+10] = 0
        np.ma.getdata(dy)[np.ma.getdata(dy) == -1e+10] = 0
        u, v = dx / (time_span * 3600), dy / (time_span * 3600)  # in m/s

        grid_spacing = abs(xc[1] - xc[0])

        dudx = sobel(u, axis=1, mode='constant') / (grid_spacing * 8)
        dudy = sobel(u, axis=0, mode='constant') / (grid_spacing * 8)
        dvdx = sobel(v, axis=1, mode='constant') / (grid_spacing * 8)
        dvdy = sobel(v, axis=0, mode='constant') / (grid_spacing * 8)

        div = (dudx + dvdy) * 86400.0  # 1/day
        she = np.sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2) * 86400.0  # 1/day

        # Check if xc and yc are in descending order
        if xc[0] > xc[-1]:
            xc = xc[::-1]
            div = div[:, ::-1]
            she = she[:, ::-1]
        if yc[0] > yc[-1]:
            yc = yc[::-1]
            div = div[::-1, :]
            she = she[::-1, :]

        f_div = RegularGridInterpolator((xc, yc), div.T, method='linear')
        f_she = RegularGridInterpolator((xc, yc), she.T, method='linear')

        div_proj = f_div((x, y))
        she_proj = f_she((x, y))

        return np.around(div_proj.flatten(), 4), np.around(she_proj.flatten(), 4)
