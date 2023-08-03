import netCDF4
import numpy as np
from scipy.interpolate import griddata
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
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
                'time_span': 48,
                'date_offset': datetime.timedelta(days=1)
            },
            'osi455': {
                'hem_nh': '_nh_',
                'hem_sh': '_sh_',
                'date_str': '{12}',
                'date_pt': '%Y%m%d%H%M',
                'time_span': 24,
                'date_offset': datetime.timedelta(days=0)
            }
        }

    def get_ice_drift(self, target_files, ice_conc):
        data = netCDF4.Dataset(target_files)
        dx_dy_unc = np.ma.getdata(data.variables['uncert_dX_and_dY'][0, :, :]).flatten()

        x0, y0 = transform_coords(np.ma.getdata(data.variables['lon'][:, :]).flatten()[dx_dy_unc != -1e10],
                                  np.ma.getdata(data.variables['lat'][:, :]).flatten()[dx_dy_unc != -1e10],
                                  'epsg:4326', self.out_epsg)

        x1, y1 = transform_coords(np.ma.getdata(data.variables['lon1'][:, :]).flatten()[dx_dy_unc != -1e10],
                                  np.ma.getdata(data.variables['lat1'][:, :]).flatten()[dx_dy_unc != -1e10],
                                  'epsg:4326', self.out_epsg)

        dx = x1 - x0
        dy = y1 - y0

        coords = np.transpose(np.vstack((x0, y0)))

        xc, yc = ice_conc["xc"], ice_conc["yc"]
        ice_conc = ice_conc["ice_conc"]

        dx_i = griddata(coords, dx, (xc, yc), method='nearest')
        dy_i = griddata(coords, dy, (xc, yc), method='nearest')
        dx_dy_unc_i = griddata(coords, dx_dy_unc[dx_dy_unc != -1e10], (xc, yc), method='nearest')

        dx_i[ice_conc == 0] = 0
        dy_i[ice_conc == 0] = 0
        dx_dy_unc_i[ice_conc == 0] = 0

        self.ice_drift = {"xc": xc, "yc": yc, "dx": dx_i, "dy": dy_i, "dx_dy_unc": dx_dy_unc_i*1000.0}

    def drift_correction(self, x, y):
        time_span = self.config[self.product_id]['time_span']
        xc, yc = self.ice_drift["xc"][0, :], self.ice_drift["yc"][:, 0]
        dx, dy = self.ice_drift['dx'], self.ice_drift['dy']
        dx_dy_unc = self.ice_drift['dx_dy_unc']
        np.ma.getdata(dx)[np.ma.getdata(dx) == -1e+10] = 0
        np.ma.getdata(dy)[np.ma.getdata(dy) == -1e+10] = 0
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
