import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import datetime
import glob
import os
import re


class OceanHeatFluxProducts:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.file_list = None
        self.file_dates = None

        self.function_map = {
            'sose': self.get_ocean_heat_flux}

        self.config = {
            'sose': {
                'hem_nh': 'NH/',
                'hem_sh': 'SH',
                'date_str': '{8}',
                'date_pt': '%Y%m%d',
                'date_offset': datetime.timedelta(days=0)
            }
        }

    def get_ocean_heat_flux(self, target_files):
        with netCDF4.Dataset(target_files) as data:
            xc = np.asanyarray(data['xc'][:])
            yc = np.asanyarray(data['yc'][:])
            ohf = np.ma.filled(data['OHF'][:], np.nan)
        xc, yc = np.meshgrid(xc,
                             yc)

        self.ohf = {"xc": xc, "yc": yc, "ohf": ohf}

    def interp_ocean_heat_flux(self, x, y):
        xc, yc = self.ohf["xc"][0, :], self.ohf["yc"][:, 0]
        arr = self.ohf["ohf"]
        if arr.ndim == 3:
            arr = arr[0]
        # Check if xc and yc are in descending order
        if xc[0] > xc[-1]:
            xc = xc[::-1]
            arr = arr[:, ::-1]

        if yc[0] > yc[-1]:
            yc = yc[::-1]
            arr = arr[::-1, :]

        interp_func = RegularGridInterpolator((xc, yc), arr.T, method='linear')

        ohf_interp = interp_func((x, y))

        return ohf_interp.flatten()

    def get_file_list(self, directory):
        config = self.config[self.product_id]
        hem = config['hem_' + self.hem]
        pattern = os.path.join(directory, hem + "*/*/*")
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
            while not file and (abs(t0i - t0) < datetime.timedelta(days=5)):
                file = [file_list[dates.index(d)] for d in dates if t0i <= d < t1i]
                t0i, t1i = t0i - dt1d, t1i - dt1d
        return file[0] if file else None

    
