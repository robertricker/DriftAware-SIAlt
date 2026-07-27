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


class SeaIceThicknessMultiProducts:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.function_map = {
            ('icesat2', 'total_freeboard'): self.atl10_to_gdf,
            ('icesat2', 'sea_ice_thickness'): self.is2sitdat4_to_gdf,
            ('cryosat2', 'sea_ice_freeboard'): self.cci_l2p_to_gdf,
            ('cryosat2', 'radar_freeboard'): self.cci_l2p_to_gdf,
            ('cryosat2', 'sea_ice_thickness'): self.cci_l2p_to_gdf,
            ('sentinel3a', 'sea_ice_freeboard'): self.cci_l2p_to_gdf,
            ('sentinel3a', 'sea_ice_thickness'): self.cci_l2p_to_gdf,
            ('sentinel3a', 'radar_freeboard'): self.cci_l2p_to_gdf,
            ('sentinel3b', 'sea_ice_freeboard'): self.cci_l2p_to_gdf,
            ('sentinel3b', 'sea_ice_thickness'): self.cci_l2p_to_gdf,
            ('sentinel3b', 'radar_freeboard'): self.cci_l2p_to_gdf,
            ('envisat', 'sea_ice_freeboard'): self.cci_l2p_to_gdf,
            ('envisat', 'sea_ice_thickness'): self.cci_l2p_to_gdf,
            ('envisat', 'radar_freeboard'): self.cci_l2p_to_gdf
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
                },
                'radar_freeboard': {
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
                },
                'radar_freeboard': {
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
                },
                'radar_freeboard': {
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
                },
                'radar_freeboard': {
                    'hem_nh': '-NH-',
                    'hem_sh': '-SH-',
                    'date_str': '{8}',
                    'date_pt': '%Y%m%d'
                }
            }
        }

        self.file_list = {key: None for key in self.config}
        self.file_dates = {key: None for key in self.config}
        self.target_files = {key: None for key in self.config}
        self.product_dict = {key: None for key in self.config}
        self.product = None

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
                if 'bounding_polygons' in key:
                    continue
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

    """def atl10_to_gdf(self, sens):
        atlas_sdp_gps_epoch = 1198800018.0
        gdf_list = list()
        for file in self.target_files[sens]:
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
    """
    def atl10_to_gdf(self, sens):
        atlas_sdp_gps_epoch = 1198800018.0
        gdf_list = list()
        for file in self.target_files[sens]:
            atl10_data, atl10_attrs, atl10_beams = self.read_atl10(file, attributes=True)
            beam_list = list()
            for beam in atl10_beams:
                beam_freeboard_keys = {key: value for key, value in atl10_data[beam]['freeboard_segment'].items() if key not in ['geophysical', 'heights']} #group "bean_freeboard" in v5 doesn't exist anymore in v6
                tmp = pd.DataFrame.from_dict(beam_freeboard_keys)
                tmp['beam'] = beam
                tmp['beam_type'] = atl10_attrs[beam]['atlas_beam_type'].decode('utf8')
                
                heights = atl10_data[beam]['freeboard_segment'].get('heights', {})
                
                if 'height_segment_sigma' in heights:
                    tmp['height_segment_sigma'] = heights['height_segment_sigma']
                if 'height_segment_height' in heights:
                    tmp['height_segment_height'] = heights['height_segment_height']
                if 'ssh_n' in heights:
                    tmp['ssh_n'] = heights['ssh_n']
                if 'height_segment_length_seg' in heights:
                    tmp['height_segment_length_seg'] = heights['height_segment_length_seg']
                
                if 'height_segment_confidence' in heights:
                    tmp['height_segment_confidence'] = heights['height_segment_confidence']
                
                if 'height_segment_rms' in heights:
                    tmp['height_segment_rms'] = heights['height_segment_rms']
                
                if 'height_segment_ssh_flag' in heights:
                    tmp['height_segment_ssh_flag'] = heights['height_segment_ssh_flag']

                if 'height_segment_type' in heights:
                    tmp['height_segment_type'] = heights['height_segment_type']

                if 'height_segment_w_gaussian' in heights:
                    tmp['height_segment_w_gaussian'] = heights['height_segment_w_gaussian']

                if 'cloud_flag_asr' in heights:
                    tmp['cloud_flag_asr'] = heights['cloud_flag_asr']

                if 'cloud_flag_atm' in heights:
                    tmp['cloud_flag_atm'] = heights['cloud_flag_atm']

                if 'photon_rate' in heights:
                    tmp['photon_rate'] = heights['photon_rate']
                
                if 'backgr_r_25' in heights:
                    tmp['backgr_r_25'] = heights['backgr_r_25']
                
                if 'msw_flag' in heights:
                    tmp['msw_flag'] = heights['msw_flag']
                if 'layer_flag' in heights:
                    tmp['layer_flag'] = heights['layer_flag']
                
                
                beam_list.append(tmp)

            # Concatenate all beam dataframes into a single GeoDataFrame    
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
    
    def is2sitdat4_to_gdf(self, sens):
        gdf_list = list()
        for file in self.target_files[sens]:
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
            df = df.dropna(subset=[self.target_var])
            gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
            gdf = gdf.to_crs(self.out_epsg)
            gdf = gdf[gdf['latitude'] > 50.0]
            gdf['time'] = (gdf['time'] - datetime.datetime(1970, 1, 1)).dt.total_seconds()
            gdf_list.append(gdf)

        gdf_final = pd.concat(gdf_list).pipe(gpd.GeoDataFrame)
        gdf_final.crs = gdf_list[0].crs

        return gdf_final.reset_index(drop=True)

    def cci_l2p_to_gdf(self, sens):
        with netCDF4.Dataset(self.target_files[sens][0]) as data:
            latitude_field = "lat" if "lat" in data.variables else "latitude"
            longitude_field = "lon" if "lon" in data.variables else "longitude"
            keep = np.ma.filled(data["flag_miz"][:] != 2, True)

            def selected_values(variable):
                return np.asanyarray(data[variable][:])[keep]

            d = {
                'latitude': selected_values(latitude_field),
                'longitude': selected_values(longitude_field),
                'radar_freeboard': selected_values("radar_freeboard"),
                'sea_ice_freeboard': selected_values("sea_ice_freeboard"),
                'sea_ice_thickness': selected_values("sea_ice_thickness"),
                'sea_ice_thickness_l2_unc': selected_values(
                    "sea_ice_thickness_uncertainty"),
                'sea_ice_freeboard_l2_unc': selected_values(
                    "sea_ice_freeboard_uncertainty"),
                'radar_freeboard_l2_unc': selected_values(
                    "radar_freeboard_uncertainty"),
                'snow_depth': selected_values("snow_depth"),
                'time': selected_values("time")
            }
        df = pd.DataFrame(data=d)
        df = df.dropna(subset=[self.target_var])
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326)
        return gdf.to_crs(self.out_epsg)

    def get_file_list(self, directory):
        for sens in self.sensor:
            hem = self.config[sens][self.target_var]['hem_' + self.hem]
            if sens != 'icesat2':
                file_list = [file_path for file_path in glob.iglob(os.path.join(directory[sens], "**", "*"), recursive=True)
                            if hem.lower() in os.path.basename(file_path.lower()) ]
            else:
                file_list = [file_path for file_path in glob.iglob(os.path.join(directory[sens][self.target_var], "**", "*"), recursive=True)
                            if hem.lower() in os.path.basename(file_path.lower()) ]
            self.file_list[sens] = file_list

    def get_file_dates(self):
        for sens in self.sensor:
            config = self.config[sens][self.target_var]
            date_str = config['date_str']
            date_pt = config['date_pt']

            dates = [
                datetime.datetime.strptime(re.search(r'\d' + date_str, file).group(), date_pt)
                for file in self.file_list[sens]
                ]
        #test if it requires to be unique
            self.file_dates[sens] = dates

    def get_target_files(self, t0, t1):
        for sens in self.sensor:
            dates = self.file_dates[sens]
            file_list = self.file_list[sens]
            self.target_files[sens] = [file for date, file in zip(dates, file_list) if t0 <= date < t1]

    def get_product(self, sensor):
        for sens in sensor:
            key = (sens, self.target_var)
            product_temp = self.function_map[key](sens)
            product_temp[sens] = 1
            #product_temp['sensor'] = sens
            self.product_dict[sens] = product_temp

        product = pd.concat(
            [gdf.assign(sensor=k) 
             for k, gdf in self.product_dict.items() if gdf is not None], 
             ignore_index=True)
        for sens in sensor:
            product[sens] = product[sens].fillna(0)
        self.product = product
        

        
