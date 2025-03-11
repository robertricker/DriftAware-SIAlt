import netCDF4
import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from loguru import logger
from io_tools import transform_coords
import datetime
import glob
import sys
import os
import re


class AirTemperatureProducts:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.file_list = None
        self.file_dates = None

        self.function_map = {
            'era5': self.get_air_temperature}

        self.config = {
            'era5': {
                'hem_nh': 'NH/',
                'hem_sh': 'SH/',
                'date_str': '{8}',
                'date_pt': '%Y%m%d',
                'date_offset': datetime.timedelta(days=0)
            }
        }

    def get_air_temperature(self, target_files):
        data = netCDF4.Dataset(target_files)
        lon, lat = np.meshgrid(np.ma.getdata(data.variables['longitude'][:]), 
                               np.ma.getdata(data.variables['latitude'][:]))
        x, y = transform_coords(lon.ravel(), lat.ravel(), 'epsg:4326', self.out_epsg)

        value = np.ma.getdata(data.variables['t2m'][0, :, :]).flatten() - 273.15 #deg C
        xy_ravel = np.linspace(-5387500, 5387500, 432)

        xc, yc = np.meshgrid(xy_ravel,
                             xy_ravel)
        
        coords = np.transpose(np.vstack((x, y)))

        air_temp = griddata(coords, value, (xc, yc), method='nearest')
        #lon, lat = transform_coords(xc[::-1,:], yc[::-1,:], self.out_epsg, 'epsg:4326')
        self.air_temp = {"xc": xc[::-1,:], "yc": yc[::-1,:], "t2m": air_temp[::-1,:]}

    def interp_air_temperature(self, x, y):
        xc, yc = self.air_temp["xc"][0, :], self.air_temp["yc"][:, 0]
        arr = self.air_temp["t2m"]
        # Check if xc and yc are in descending order
        if xc[0] > xc[-1]:
            xc = xc[::-1]
            arr = arr[:, ::-1]
        if yc[0] > yc[-1]:
            yc = yc[::-1]
            arr = arr[::-1, :]

        interp_func = RegularGridInterpolator((xc, yc), arr.T, method='linear')
        air_temp_interp = interp_func((x, y))
        return air_temp_interp.flatten()

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

    def thermodyn_growth(self, hice, hsnow, x, y, direct):
        t2m = self.interp_air_temperature(x, y)
        
        L = 3*1e8 # Latent heat of fusion
        T_0 = -1.9 # temperature at the ice-water interface
        k_ice = 2 # thermal conductivity of the ice
        k_snow = 0.33 # thermal conductivity of the snow
        F = 2 # the ocean heat flux, is assumed to be constant #TODO, take it not constant ?
        dt = 86400 # daily

        if direct == 1:
            deltaH = direct * dt * (-1/L) * (F + (t2m - T_0)*((k_ice * k_snow)/(k_ice * hsnow + k_snow * hice)))
            Hf = deltaH + hice
        elif direct == -1:
            # in this case hice = Hf
            Hf = np.copy(hice)
            # need to find hice, knowing Hf in the previous equation : DeltaH = A*(F + B/(C+k_snow*hice)) with :
            A = -1/L
            B = (t2m-T_0)*k_ice*k_snow
            C = k_ice*hsnow
            # polynom's coefficients a*hice**2 + b*hice + c = 0 
            a = -k_snow
            b = -(C - Hf*k_snow + dt*A*F*k_snow)
            c = -dt*A*C*F - dt*A*B + C*Hf
            # Discriminant :
            D = b**2 - 4*a*c
            if D >= 0 :
                s1 = (-b-np.sqrt(D))/(2*a)
                #s2 = (-b+np.sqrt(D))/(2*a)
                deltaH = (Hf - s1) #already in the time direction
                Hf = s1.copy() #that is in fact Hice ... 
            else:
                logger.error('No real solution to this polynom, the discriminant is equal to: %s', D)
            
            # in this case, if deltaH>0 <=> Hf>Hi means that going back in time there is melting and going with t>0 there is freezing.
            # So deltaH should be removed to the Hi the most advanced in time to get the hice.
            # if we consider the time t to correct the hice (which is Hf) from the thermodynamic we should consider the t2m at t-1day.

        return deltaH.values, Hf.values
    
    #def thermodyn_growth(self, hice, hsnow, x, y):
        t2m = self.interp_air_temperature(x, y)
        
        L = 3*1e8 # Latent heat of fusion
        T_0 = -1.9 # temperature at the ice-water interface
        k_ice = 2 # thermal conductivity of the ice
        k_snow = 0.33 # thermal conductivity of the snow
        F = 2 # the ocean heat flux, is assumed to be constant #TODO, take it not constant ?
        dt = 86400 # daily

        deltaH = dt * (-1/L) * (F + (t2m - T_0)*((k_ice * k_snow)/(k_ice * hsnow + k_snow * hice)))
        return deltaH