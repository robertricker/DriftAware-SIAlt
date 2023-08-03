import netCDF4
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from io_tools import transform_coords
import datetime
import glob
import sys
import os
import re


class SeaIceConcentrationProducts:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.file_list = None
        self.file_dates = None

        self.function_map = {
            'osi430b': self.get_ice_concentration,
            'osi450': self.get_ice_concentration
        }

        self.config = {
            'osi430b': {
                'hem_nh': '_nh_',
                'hem_sh': '_sh_',
                'date_str': '{12}',
                'date_pt': '%Y%m%d%H%M',
                'date_offset': datetime.timedelta(days=0)
            },
            'osi450': {
                'hem_nh': '_nh_',
                'hem_sh': '_sh_',
                'date_str': '{12}',
                'date_pt': '%Y%m%d%H%M',
                'date_offset': datetime.timedelta(days=0)
            }
        }

    def get_ice_concentration(self, target_files):
        data = netCDF4.Dataset(target_files)
        x, y = transform_coords(np.ma.getdata(data.variables['lon'][:, :]).flatten(),
                                np.ma.getdata(data.variables['lat'][:, :]).flatten(),
                                'epsg:4326', self.out_epsg)

        value = np.ma.getdata(data.variables['ice_conc'][0, :, :]).flatten()
        xc, yc = np.meshgrid(np.ma.getdata(data.variables['xc'][:] * 1000.0),
                             np.ma.getdata(data.variables['yc'][:] * 1000.0))
        coords = np.transpose(np.vstack((x, y)))

        ice_conc = griddata(coords, value, (xc, yc), method='nearest')
        ice_conc[ice_conc < 15] = 0
        return {"xc": xc, "yc": yc, "ice_conc": ice_conc}

    @staticmethod
    def interp_ice_concentration(ice_conc, x, y):
        xc, yc = ice_conc["xc"][0, :], ice_conc["yc"][:, 0]
        arr = ice_conc["ice_conc"]
        # Check if xc and yc are in descending order
        if xc[0] > xc[-1]:
            xc = xc[::-1]
            arr = arr[:, ::-1]
        if yc[0] > yc[-1]:
            yc = yc[::-1]
            arr = arr[::-1, :]

        interp_func = RegularGridInterpolator((xc, yc), arr.T, method='linear')
        ice_conc_interp = interp_func((x, y))
        return ice_conc_interp.flatten() / 100.0

    def get_file_list(self, directory):
        config = self.config[self.product_id]
        hem = config['hem_' + self.hem]
        pattern = os.path.join(directory, "**", "*" + hem + "*")
        file_list = sorted(glob.glob(pattern, recursive=True))
        self.file_list = file_list

    def get_file_dates(self):
        config = self.config[self.product_id]
        date_str = config['date_str']
        date_pt = config['date_pt']
        dates = [
            datetime.datetime.strptime(re.search(r'\d' + date_str, file).group(), date_pt)
            for file in self.file_list
        ]
        self.file_dates = [date + config['date_offset'] for date in dates]

    def get_target_files(self, t0, t1):
        dt1d = datetime.timedelta(days=1)
        dates = self.file_dates
        file_list = self.file_list
        file = [file_list[dates.index(d)] for d in dates if t0 <= d < t1]
        if len(file) == 0:
            t0i, t1i = t0, t1
            while file and (abs(t0i - t0) < datetime.timedelta(days=10)):
                file = (file_list[dates.index(d)] for d in dates if t0i <= d < t1i)
                t0i, t1i = t0i - dt1d, t1i - dt1d
        return file[0]
