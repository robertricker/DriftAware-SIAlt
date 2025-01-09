import numpy as np
import pandas as pd
import geopandas as gpd
import glob
import re
import datetime
import h5py
import netCDF4
import os
from astropy.time import Time


class SeaIceThicknessProducts:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.file_list = None
        self.file_dates = None
        self.target_files = None
        self.product = None

        self.function_map = {
            ('icesat2', 'total_freeboard'): self.atl10_to_gdf,
            ('icesat2', 'sea_ice_thickness'): self.is2sitdat4_to_gdf,
            ('cryosat2', 'sea_ice_freeboard'): self.cci_l2p_to_gdf,
            ('cryosat2', 'sea_ice_thickness'): self.cci_l2p_to_gdf,
            ('sentinel3a', 'sea_ice_freeboard'): self.cci_l2p_to_gdf,
            ('sentinel3a', 'sea_ice_thickness'): self.cci_l2p_to_gdf,
            ('sentinel3b', 'sea_ice_freeboard'): self.cci_l2p_to_gdf,
            ('sentinel3b', 'sea_ice_thickness'): self.cci_l2p_to_gdf,
            ('envisat', 'sea_ice_freeboard'): self.cci_l2p_to_gdf,
            ('envisat', 'sea_ice_thickness'): self.cci_l2p_to_gdf
        }

        self.config = {
            'icesat2': {
                'total_freeboard': {
                    'hem_nh': '-01',
                    'hem_sh': '-02',
                    'date_str': '{14}',
                    'date_pt': '%Y%m%d%H%M%S'
                },
                'sea_ice_thickness': {
                    'hem_nh': 'IS2SITDAT4_01',
                    'hem_sh': 'IS2SITDAT4_02',
                    'date_str': '{14}',
                    'date_pt': '%Y%m%d%H%M%S'
                }
            },
            'cryosat2': {
                'sea_ice_thickness': {
                    'hem_nh': '-nh-',
                    'hem_sh': '-sh-',
                    'date_str': '{8}',
                    'date_pt': '%Y%m%d'
                },
                'sea_ice_freeboard': {
                    'hem_nh': '-nh-',
                    'hem_sh': '-sh-',
                    'date_str': '{8}',
                    'date_pt': '%Y%m%d'
                }
            },
            'sentinel3a': {
                'sea_ice_thickness': {
                    'hem_nh': '-nh-',
                    'hem_sh': '-sh-',
                    'date_str': '{8}',
                    'date_pt': '%Y%m%d'
                },
                'sea_ice_freeboard': {
                    'hem_nh': '-nh-',
                    'hem_sh': '-sh-',
                    'date_str': '{8}',
                    'date_pt': '%Y%m%d'
                }
            },
            'sentinel3b': {
                'sea_ice_thickness': {
                    'hem_nh': '-nh-',
                    'hem_sh': '-sh-',
                    'date_str': '{8}',
                    'date_pt': '%Y%m%d'
                },
                'sea_ice_freeboard': {
                    'hem_nh': '-nh-',
                    'hem_sh': '-sh-',
                    'date_str': '{8}',
                    'date_pt': '%Y%m%d'
                }
            },
            'envisat': {
                'sea_ice_thickness': {
                    'hem_nh': '-NH-',
                    'hem_sh': '-SH-',
                    'date_str': '{8}',
                    'date_pt': '%Y%m%d'
                },
                'sea_ice_freeboard': {
                    'hem_nh': '-NH-',
                    'hem_sh': '-SH-',
                    'date_str': '{8}',
                    'date_pt': '%Y%m%d'
                }
            }
        }

    def get_product(self):
        key = (self.sensor, self.target_var)
        product = self.function_map[key]()
        self.product = product

    @staticmethod
    def read_atl10(filename, attributes=False):
        with h5py.File(filename, 'r') as fileid:
            atl10_data = {}
            atl10_attrs = {}
            atl10_beams = []
            list_gtx = [k for k in fileid.keys() if bool(re.match(r'gt\d[lr]', k))]
            for gtx in list_gtx:
                try:
                    fileid[gtx]['freeboard_segment']['height_segment_id']
                except KeyError:
                    pass
                else:
                    atl10_beams.append(gtx)

            for gtx in atl10_beams:
                atl10_data[gtx] = {}
                atl10_data[gtx]['freeboard_segment'] = {}
                #atl10_data[gtx]['freeboard_segment']['beam_freeboard'] = {}
                atl10_data[gtx]['freeboard_segment']['geophysical'] = {}
                atl10_data[gtx]['freeboard_segment']['heights'] = {}
                atl10_data[gtx]['leads'] = {}

                for key, val in fileid[gtx]['freeboard_segment'].items():
                    if isinstance(val, h5py.Dataset):
                        atl10_data[gtx]['freeboard_segment'][key] = val[:]
                    elif isinstance(val, h5py.Group):
                        for k, v in val.items():
                            atl10_data[gtx]['freeboard_segment'][key][k] = v[:]

                if attributes:
                    # getting attributes of icesat-2 atl10 beam variables
                    atl10_attrs[gtx] = {}
                    atl10_attrs[gtx]['freeboard_segment'] = {}
                    #atl10_attrs[gtx]['freeboard_segment']['beam_freeboard'] = {}
                    atl10_attrs[gtx]['freeboard_segment']['geophysical'] = {}
                    atl10_attrs[gtx]['freeboard_segment']['heights'] = {}
                    atl10_attrs[gtx]['leads'] = {}
                    # global group attributes for atl10 beam
                    for att_name, att_val in fileid[gtx].attrs.items():
                        atl10_attrs[gtx][att_name] = att_val
                    for key, val in fileid[gtx]['freeboard_segment'].items():
                        atl10_attrs[gtx]['freeboard_segment'][key] = {}
                        for att_name, att_val in val.attrs.items():
                            atl10_attrs[gtx]['freeboard_segment'][key][att_name] = att_val
                        if isinstance(val, h5py.Group):
                            for k, v in val.items():
                                atl10_attrs[gtx]['freeboard_segment'][key][k] = {}
                                for att_name, att_val in v.attrs.items():
                                    atl10_attrs[gtx]['freeboard_segment'][key][k][att_name] = att_val

            # icesat-2 orbit_info group
            atl10_data['orbit_info'] = {}
            for key, val in fileid['orbit_info'].items():
                atl10_data['orbit_info'][key] = val[:]

            atl10_data['ancillary_data'] = {}
            atl10_attrs['ancillary_data'] = {}
            for key in ['atlas_sdp_gps_epoch']:
                # get each hdf5 variable
                atl10_data['ancillary_data'][key] = fileid['ancillary_data'][key][:]
                # getting attributes of group and included variables
                if attributes:
                    # -- variable attributes
                    atl10_attrs['ancillary_data'][key] = {}
                    for att_name, att_val in fileid['ancillary_data'][key].attrs.items():
                        atl10_attrs['ancillary_data'][key][att_name] = att_val

        return atl10_data, atl10_attrs, atl10_beams

    def atl10_to_gdf(self):
        atlas_sdp_gps_epoch = 1198800018.0
        gdf_list = list()
        for file in self.target_files:
            atl10_data, atl10_attrs, atl10_beams = self.read_atl10(file, attributes=True)
            beam_list = list()
            for beam in atl10_beams:
                beam_freeboard_keys = {key: value for key, value in atl10_data[beam]['freeboard_segment'].items() if key not in ['geophysical', 'heights']} #group "bean_freeboard" in v5 doesn't exist anymore in v6
                tmp = pd.DataFrame.from_dict(beam_freeboard_keys)
                tmp['beam'] = beam
                tmp['beam_type'] = atl10_attrs[beam]['atlas_beam_type'].decode('utf8')
                beam_list.append(tmp)

            df = pd.concat([df for df in beam_list]).pipe(gpd.GeoDataFrame)
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
            gdf = gdf.to_crs(self.out_epsg)

            #gdf = gdf[(gdf['beam_fb_height'] < 10.0) &
            #          (gdf['latitude'] > 50.0)] # not compatible with southern hemisphere
            gdf = gdf[(gdf['beam_fb_height'] < 10.0)]
            gdf_list.append(gdf)

        gdf_final = pd.concat(gdf_list).pipe(gpd.GeoDataFrame)
        gdf_final.crs = gdf_list[0].crs
        gdf_final['time'] = Time(gdf_final['delta_time'] + atlas_sdp_gps_epoch, format='gps').to_datetime()
        gdf_final['time'] = (gdf_final['time'] - datetime.datetime(1970, 1, 1)).dt.total_seconds()
        gdf_final.rename(columns={"beam_fb_confidence": "total_freeboard_confidence",
                                  "beam_fb_height": "total_freeboard",
                                  "beam_fb_quality_flag": "total_freeboard_quality_flag",
                                  "beam_fb_unc": "total_freeboard_l2_unc"}, inplace=True) #sigma -> unc in v6

        return gdf_final.reset_index(drop=True)

    def is2sitdat4_to_gdf(self):
        gdf_list = list()
        for file in self.target_files:
            data = netCDF4.Dataset(file)

            start_idx = os.path.basename(file).find("bnum") + 5
            end_idx = start_idx + 4
            beam = os.path.basename(file)[start_idx:end_idx]

            d = {
                'latitude': np.array(data["latitude"]),
                'longitude': np.array(data["longitude"]),
                'sea_ice_freeboard': np.array(data["freeboard"]),
                'sea_ice_thickness': np.array(data["ice_thickness"]),
                'sea_ice_thickness_l2_unc': np.array(data["ice_thickness_unc"]),
                'snow_depth': np.array(data["snow_depth"]),
                'ssh_flag': np.array(data["ssh_flag"]),
                'time': Time(np.array(data["gps_seconds"]), format='gps').to_datetime(),
                'beam': beam,
                'beam_type': "strong"
            }
            df = pd.DataFrame(data=d)
            df = df.dropna(how='any')
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
            gdf = gdf.to_crs(self.out_epsg)
            gdf = gdf[gdf['latitude'] > 50.0]
            gdf['time'] = (gdf['time'] - datetime.datetime(1970, 1, 1)).dt.total_seconds()
            gdf_list.append(gdf)

        gdf_final = pd.concat(gdf_list).pipe(gpd.GeoDataFrame)
        gdf_final.crs = gdf_list[0].crs

        return gdf_final.reset_index(drop=True)

    def cci_l2p_to_gdf(self):
        data = netCDF4.Dataset(self.target_files[0])
        latitude_field = "lat" if "lat" in data.variables else "latitude"
        longitude_field = "lon" if "lon" in data.variables else "longitude"
        d = {
            'latitude': np.array(data[latitude_field]),
            'longitude': np.array(data[longitude_field]),
            'radar_freeboard': np.array(data["radar_freeboard"]),
            'sea_ice_freeboard': np.array(data["sea_ice_freeboard"]),
            'sea_ice_thickness': np.array(data["sea_ice_thickness"]),
            'sea_ice_thickness_l2_unc': np.array(data["sea_ice_thickness_uncertainty"]),
            'sea_ice_freeboard_l2_unc': np.array(data["sea_ice_freeboard_uncertainty"]),
            'snow_depth': np.array(data["snow_depth"]),
            'time': np.array(data["time"])
        }
        df = pd.DataFrame(data=d)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
        return gdf.to_crs(self.out_epsg)

    def get_file_list(self, directory):
        hem = self.config[self.sensor][self.target_var]['hem_' + self.hem]
        file_list = [file_path for file_path in glob.iglob(os.path.join(directory, "**", "*"), recursive=True)
                     if hem.lower() in os.path.basename(file_path.lower())]
        self.file_list = file_list

    def get_file_dates(self):
        config = self.config[self.sensor][self.target_var]
        date_str = config['date_str']
        date_pt = config['date_pt']

        dates = [
            datetime.datetime.strptime(re.search(r'\d' + date_str, file).group(), date_pt)
            for file in self.file_list
        ]
        self.file_dates = dates

    def get_target_files(self, t0, t1):
        dates = self.file_dates
        file_list = self.file_list
        self.target_files = [file for date, file in zip(dates, file_list) if t0 <= date < t1]
