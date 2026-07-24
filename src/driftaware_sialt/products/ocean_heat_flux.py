import netCDF4
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from driftaware_sialt.io_tools import transform_coords
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
            'sose_155': self.get_ocean_heat_flux}

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
        data = netCDF4.Dataset(target_files)
        xc = data['xc']
        yc = data['yc']
        ohf = data['OHF']
        ohf2 = data['OHF2']
        ohf3 = data['OHF3']
        ohf4 = data['OHF4']
        ohf5 = data['OHF5']
        ohf6 = data['OHF6']
        xc, yc = np.meshgrid(xc,
                             yc)
        lon, lat = transform_coords(xc, yc, self.out_epsg, 'epsg:4326')
        
        self.ohf = {"xc": xc, "yc": yc, "ohf": ohf, "ohf2": ohf2, 
                    "ohf3": ohf3, "ohf4": ohf4, 
                    "ohf5": ohf5, "ohf6": ohf6}

    def interp_ocean_heat_flux(self, x, y):
        xc, yc = self.ohf["xc"][0, :], self.ohf["yc"][:, 0]
        arr = self.ohf["ohf"][0]
        arr2 = self.ohf["ohf2"][0]
        arr3 = self.ohf["ohf3"][0]
        arr4 = self.ohf["ohf4"][0]
        arr5 = self.ohf["ohf5"][0]
        arr6 = self.ohf["ohf6"][0]
        # Check if xc and yc are in descending order
        if xc[0] > xc[-1]:
            xc = xc[::-1]
            arr = arr[:, ::-1]
            arr2 = arr2[:, ::-1]
            arr3 = arr3[:, ::-1]
            arr4 = arr4[:, ::-1]
            arr5 = arr5[:, ::-1]
            arr6 = arr6[:, ::-1]
        if yc[0] > yc[-1]:
            yc = yc[::-1]
            arr = arr[::-1, :]
            arr2 = arr2[::-1, :]
            arr3 = arr3[::-1, :]
            arr4 = arr4[::-1, :]
            arr5 = arr5[::-1, :]
            arr6 = arr6[::-1, :]

        interp_func = RegularGridInterpolator((xc, yc), arr.T, method='linear')
        interp_func2 = RegularGridInterpolator((xc, yc), arr2.T, method='linear')
        interp_func3 = RegularGridInterpolator((xc, yc), arr3.T, method='linear')
        interp_func4 = RegularGridInterpolator((xc, yc), arr4.T, method='linear')
        interp_func5 = RegularGridInterpolator((xc, yc), arr5.T, method='linear')
        interp_func6 = RegularGridInterpolator((xc, yc), arr6.T, method='linear')

        ohf_interp = interp_func((x, y))
        ohf_interp2 = interp_func2((x, y))
        ohf_interp3 = interp_func3((x, y))
        ohf_interp4 = interp_func4((x, y))
        ohf_interp5 = interp_func5((x, y))
        ohf_interp6 = interp_func6((x, y))

        return ohf_interp.flatten(), ohf_interp2.flatten(), ohf_interp3.flatten(), ohf_interp4.flatten(), ohf_interp5.flatten(), ohf_interp6.flatten()

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
        return file[0]

    
