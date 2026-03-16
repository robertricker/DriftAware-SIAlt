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


class SeaIceThicknessClimProducts:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.file_list = None
        self.file_dates = None

        self.function_map = {
            'mms_clim': self.get_tFB_clim,
        }

        self.config = {
            'mms_clim': {
                'date_str': '{4}',
                'date_pt': 'icdc_%m%d',
                'date_offset': datetime.timedelta(days=0)
            },
        }

    def get_tFB_clim(self, target_files):
        data = netCDF4.Dataset(target_files)
        x, y = transform_coords(np.ma.getdata(data.variables['lon'][:, :]).flatten(),
                                np.ma.getdata(data.variables['lat'][:, :]).flatten(),
                                'epsg:4326', self.out_epsg)

        tFB_interp = np.ma.getdata(data.variables['tFB_interp'][:, :]).flatten()
        sigma_tFB_interp = np.ma.getdata(data.variables['sigma_tFB_interp'][:, :]).flatten()
        longitude_value = np.ma.getdata(data.variables['lon'][:, :]).flatten()
        latitude_value = np.ma.getdata(data.variables['lat'][:, :]).flatten()

        xc, yc = np.meshgrid(np.ma.getdata(data.variables['xc'][:]),
                             np.ma.getdata(data.variables['yc'][:]))
        coords = np.transpose(np.vstack((x, y)))

        tFB_interp = griddata(coords, tFB_interp, (xc, yc), method='nearest')
        sigma_tFB_interp = griddata(coords, sigma_tFB_interp, (xc, yc), method='nearest')
        return {"xc": xc, "yc": yc, 
                "tFB_interp": tFB_interp,
                "sigma_tFB_interp": sigma_tFB_interp,
                "longitude": longitude_value, 
                "latitude": latitude_value}

    
    def get_file_list(self, directory):
        config = self.config[self.product_id]
        pattern = os.path.join(directory, "**", "SOSIMBA_s3c2_total_freeboard_climatology_icdc_" + "*.nc")
        file_list = sorted(glob.glob(pattern, recursive=True))
        self.file_list = file_list

    def get_file_dates(self):
        config = self.config[self.product_id]
        date_str = config['date_str']
        date_pt = config['date_pt']

        dates = []
        for file in self.file_list:
            try:
                date = datetime.datetime.strptime(
                    re.search(r'icdc_\d' + date_str, file).group(), date_pt
                )
                dates.append(date)
            except ValueError as e:
                if "day is out of range" in str(e):  
                    continue  
                else:
                    raise

        self.file_dates = [date + config['date_offset'] for date in dates]
    
    def get_target_files(self, t0, t1):
        dt1d = datetime.timedelta(days=1)
        dates = self.file_dates
        file_list = self.file_list

        def select_files(t0, t1, dates, file_list):
            # Cas normal : pas de passage au 31/12
            if (t0.month, t0.day) < (t1.month, t1.day):
                candidates = [f for d, f in zip(dates, file_list)
                            if (t0.month, t0.day) <= (d.month, d.day) < (t1.month, t1.day)]
            else:
                # Cas wrap-around : ex t0=12/31, t1=01/02
                candidates = [f for d, f in zip(dates, file_list)
                            if (d.month, d.day) >= (t0.month, t0.day) or
                                (d.month, d.day) < (t1.month, t1.day)]
            return candidates

        file = select_files(t0, t1, dates, file_list)

        if len(file) == 0:
            t0i, t1i = t0, t1
            while not file and (abs(t0i - t0) < datetime.timedelta(days=8)):
                file = select_files(t0i, t1i, dates, file_list)
                t0i, t1i = t0i - dt1d, t1i - dt1d

        return file[0] if file else None