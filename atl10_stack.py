import pandas as pd
import geopandas as gpd
import shapely as shp
import numpy as np
import xarray as xr
import netCDF4
import matplotlib.pyplot as plt
from scipy import interpolate
from read_icesat2.read_ICESat2_ATL10 import read_HDF5_ATL10
import warnings
import cartopy.crs as ccrs
import cartopy
import glob
import logging, sys
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

