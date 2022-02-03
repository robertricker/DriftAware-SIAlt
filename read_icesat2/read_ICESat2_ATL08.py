#!/usr/bin/env python
u"""
read_ICESat2_ATL08.py (10/2021)

OPTIONS:
    ATTRIBUTES: read HDF5 attributes for groups and variables
    SIGNAL_PHOTONS: read ATL06 residual_histogram variables

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    h5py: Python interface for Hierarchal Data Format 5 (HDF5)
        https://www.h5py.org/
"""
from __future__ import print_function

import os
import io
import re
import h5py
import logging
import numpy as np

#-- PURPOSE: read ICESat-2 ATL08 HDF5 data files
def read_HDF5_ATL08(FILENAME, ATTRIBUTES=False, SIGNAL_PHOTONS=False,**kwargs):
    """
    Reads ICESat-2 ATL08 data files

    Arguments
    ---------
    FILENAME: full path to ATL06 file

    Keyword arguments
    -----------------
    ATTRIBUTES: read HDF5 attributes for groups and variables
    signal_photons: read ATL08 residual_histogram variables

    Returns
    -------
    IS2_atl08_mds: dictionary with ATL08 variables
    IS2_atl08_attrs: dictionary with ATL08 attributes
    IS2_atl08_beams: list with valid ICESat-2 beams within ATL08 file
    """
    #-- Open the HDF5 file for reading
    if isinstance(FILENAME, io.IOBase):
        fileID = h5py.File(FILENAME, 'r')
    else:
        fileID = h5py.File(os.path.expanduser(FILENAME), 'r')

    #-- Output HDF5 file information
    logging.info(fileID.filename)
    logging.info(list(fileID.keys()))

    #-- allocate python dictionaries for ICESat-2 ATL06 variables and attributes
    IS2_atl08_mds = {}
    IS2_atl08_attrs = {}

    #-- read each input beam within the file
    IS2_atl08_beams = []
    for gtx in [k for k in fileID.keys() if bool(re.match(r'gt\d[lr]',k))]:
        #-- check if subsetted beam contains land ice data
        try:
            fileID[gtx]['land_segments']['segment_id_beg']
        except KeyError:
            pass
        else:
            IS2_atl08_beams.append(gtx)

    #-- read each input beam within the file
    for gtx in IS2_atl08_beams:
        IS2_atl08_mds[gtx] = {}
        IS2_atl08_mds[gtx]['land_segments'] = {}
        IS2_atl08_mds[gtx]['land_segments']['canopy'] = {}
        IS2_atl08_mds[gtx]['land_segments']['terrain'] = {}
        #-- get each HDF5 variable
        #-- ICESat-2 land_segments Group
        for key,val in fileID[gtx]['land_segments'].items():
            if isinstance(val, h5py.Dataset):
                IS2_atl08_mds[gtx]['land_segments'][key] = val[:]
            elif isinstance(val, h5py.Group):
                for k,v in val.items():
                    IS2_atl08_mds[gtx]['land_segments'][key][k] = v[:]

        #-- ICESat-2 signal_photons Group
        if SIGNAL_PHOTONS:
            IS2_atl08_mds[gtx]['signal_photons'] = {}
            for key,val in fileID[gtx]['signal_photons'].items():
                IS2_atl08_mds[gtx]['signal_photons'][key] = val[:]

        #-- Getting attributes of included variables
        if ATTRIBUTES:
            #-- Getting attributes of ICESat-2 ATL08 beam variables
            IS2_atl08_attrs[gtx] = {}
            IS2_atl08_attrs[gtx]['land_segments'] = {}
            IS2_atl08_attrs[gtx]['land_segments']['canopy'] = {}
            IS2_atl08_attrs[gtx]['land_segments']['terrain'] = {}
            #-- Global Group Attributes for ATL08 beam
            for att_name,att_val in fileID[gtx].attrs.items():
                IS2_atl08_attrs[gtx][att_name] = att_val
            for key,val in fileID[gtx]['land_segments'].items():
                IS2_atl08_attrs[gtx]['land_segments'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl08_attrs[gtx]['land_segments'][key][att_name] = att_val
                if isinstance(val, h5py.Group):
                    for k,v in val.items():
                        IS2_atl08_attrs[gtx]['land_segments'][key][k] = {}
                        for att_name,att_val in v.attrs.items():
                            IS2_atl08_attrs[gtx]['land_segments'][key][k][att_name] = att_val
        #-- Getting attributes of signal_photons variables
        if ATTRIBUTES and SIGNAL_PHOTONS:
            #-- ICESat-2 signal_photons Group
            IS2_atl08_attrs[gtx]['signal_photons'] = {}
            for key,val in fileID[gtx]['signal_photons'].items():
                IS2_atl08_attrs[gtx]['signal_photons'][key] = {}
                for att_name,att_val in val.attrs.items():
                    IS2_atl08_attrs[gtx]['signal_photons'][key][att_name] = att_val
       
    #-- ICESat-2 orbit_info Group
    IS2_atl08_mds['orbit_info'] = {}
    for key,val in fileID['orbit_info'].items():
        IS2_atl08_mds['orbit_info'][key] = val[:]
    #-- ICESat-2 quality_assessment Group
    IS2_atl08_mds['quality_assessment'] = {}
    for key,val in fileID['quality_assessment'].items():
        if isinstance(val, h5py.Dataset):
            IS2_atl08_mds['quality_assessment'][key] = val[:]
        elif isinstance(val, h5py.Group):
            IS2_atl08_mds['quality_assessment'][key] = {}
            for k,v in val.items():
                IS2_atl08_mds['quality_assessment'][key][k] = v[:]

    #-- number of GPS seconds between the GPS epoch (1980-01-06T00:00:00Z UTC)
    #-- and ATLAS Standard Data Product (SDP) epoch (2018-01-01:T00:00:00Z UTC)
    #-- Add this value to delta time parameters to compute full gps_seconds
    #-- could alternatively use the Julian day of the ATLAS SDP epoch: 2458119.5
    #-- and add leap seconds since 2018-01-01:T00:00:00Z UTC (ATLAS SDP epoch)
    IS2_atl08_mds['ancillary_data'] = {}
    IS2_atl08_attrs['ancillary_data'] = {}
    for key in ['atlas_sdp_gps_epoch']:
        #-- get each HDF5 variable
        IS2_atl08_mds['ancillary_data'][key] = fileID['ancillary_data'][key][:]
        #-- Getting attributes of group and included variables
        if ATTRIBUTES:
            #-- Variable Attributes
            IS2_atl08_attrs['ancillary_data'][key] = {}
            for att_name,att_val in fileID['ancillary_data'][key].attrs.items():
                IS2_atl08_attrs['ancillary_data'][key][att_name] = att_val

    #-- land ancillary information (first photon bias and statistics)
    cal1,cal2 = ('ancillary_data','land')
    IS2_atl08_mds[cal1][cal2] = {}
    IS2_atl08_attrs[cal1][cal2] = {}
    for key,val in fileID[cal1][cal2].items():
        #-- get each HDF5 variable
        IS2_atl08_mds[cal1][cal2][key] = val[:]
        #-- Getting attributes of group and included variables
        if ATTRIBUTES:
            #-- Variable Attributes
            IS2_atl08_attrs[cal1][cal2][key] = {}
            for att_name,att_val in val.attrs.items():
                IS2_atl08_attrs[cal1][cal2][key][att_name] = att_val

    #-- get each global attribute and the attributes for orbit and quality
    if ATTRIBUTES:
        #-- ICESat-2 HDF5 global attributes
        for att_name,att_val in fileID.attrs.items():
            IS2_atl08_attrs[att_name] = att_name
        #-- ICESat-2 orbit_info Group
        IS2_atl08_attrs['orbit_info'] = {}
        for key,val in fileID['orbit_info'].items():
            IS2_atl08_attrs['orbit_info'][key] = {}
            for att_name,att_val in val.attrs.items():
                IS2_atl08_attrs['orbit_info'][key][att_name]= att_val
        #-- ICESat-2 quality_assessment Group
        IS2_atl08_attrs['quality_assessment'] = {}
        for key,val in fileID['quality_assessment'].items():
            IS2_atl08_attrs['quality_assessment'][key] = {}
            for att_name,att_val in val.attrs.items():
                IS2_atl08_attrs['quality_assessment'][key][att_name]= att_val
            if isinstance(val, h5py.Group):
                for k,v in val.items():
                    IS2_atl08_attrs['quality_assessment'][key][k] = {}
                    for att_name,att_val in v.attrs.items():
                        IS2_atl08_attrs['quality_assessment'][key][k][att_name]= att_val

    #-- Closing the HDF5 file
    fileID.close()
    #-- Return the datasets and variables
    return (IS2_atl08_mds,IS2_atl08_attrs,IS2_atl08_beams)

#-- PURPOSE: find valid beam groups within ICESat-2 ATL06 HDF5 data files
def find_HDF5_ATL08_beams(FILENAME, **kwargs):
    """
    Find valid beam groups within ICESat-2 ATL08 (Land Along-Track
    Height Product) data files

    Arguments
    ---------
    FILENAME: full path to ATL06 file

    Returns
    -------
    IS2_atl08_beams: list with valid ICESat-2 beams within ATL08 file
    """
    #-- Open the HDF5 file for reading
    if isinstance(FILENAME, io.IOBase):
        fileID = h5py.File(FILENAME, 'r')
    else:
        fileID = h5py.File(os.path.expanduser(FILENAME), 'r')
    #-- output list of beams
    IS2_atl08_beams = []
    #-- read each input beam within the file
    for gtx in [k for k in fileID.keys() if bool(re.match(r'gt\d[lr]',k))]:
        #-- check if subsetted beam contains land ice data
        try:
            fileID[gtx]['land_segments']['segment_id_beg']
        except KeyError:
            pass
        else:
            IS2_atl08_beams.append(gtx)
    #-- Closing the HDF5 file
    fileID.close()
    #-- return the list of beams
    return IS2_atl08_beams

#-- PURPOSE: read ICESat-2 ATL06 HDF5 data files for beam variables
def read_HDF5_ATL08_beam(FILENAME, gtx, ATTRIBUTES=False, **kwargs):
    """
    Reads ICESat-2 ATL08 (Land Ice Along-Track Height Product) data files
    for a specific beam

    Arguments
    ---------
    FILENAME: full path to ATL08 file
    gtx: beam name based on ground track and position
        gt1l
        gt1r
        gt2l
        gt2r
        gt3l
        gt3r

    Keyword arguments
    -----------------
    ATTRIBUTES: read HDF5 attributes for groups and variables
    HISTOGRAM: read ATL08 residual_histogram variables
    QUALITY: read ATL08 segment_quality variables

    Returns
    -------
    IS2_atl08_mds: dictionary with ATL08 variables
    IS2_atl08_attrs: dictionary with ATL08 attributes
    """
    #-- Open the HDF5 file for reading
    if isinstance(FILENAME, io.IOBase):
        fileID = h5py.File(FILENAME, 'r')
    else:
        fileID = h5py.File(os.path.expanduser(FILENAME), 'r')

    #-- Output HDF5 file information
    logging.info(fileID.filename)
    logging.info(list(fileID.keys()))

    #-- allocate python dictionaries for ICESat-2 ATL06 variables and attributes
    IS2_atl08_mds = {}
    IS2_atl08_attrs = {}

    #-- read input beam within the file
    IS2_atl08_mds[gtx] = {}
    IS2_atl08_mds[gtx]['land_segments'] = {}
    IS2_atl08_mds[gtx]['land_segments']['canopy'] = {}
    IS2_atl08_mds[gtx]['land_segments']['terrain'] = {}
    #-- get each HDF5 variable
    #-- ICESat-2 land_ice_segments Group
    for key,val in fileID[gtx]['land'].items():
        if isinstance(val, h5py.Dataset):
            IS2_atl08_mds[gtx]['land'][key] = val[:]
        elif isinstance(val, h5py.Group):
            for k,v in val.items():
                IS2_atl08_mds[gtx]['land'][key][k] = v[:]

    #-- Getting attributes of included variables
    if ATTRIBUTES:
        #-- Getting attributes of ICESat-2 ATL06 beam variables
        IS2_atl08_attrs[gtx] = {}
        IS2_atl08_attrs[gtx]['land_ice_segments'] = {}
        IS2_atl08_attrs[gtx]['land_ice_segments']['canopy'] = {}
        IS2_atl08_attrs[gtx]['land_ice_segments']['terrain'] = {}
        #-- Global Group Attributes for ATL06 beam
        for att_name,att_val in fileID[gtx].attrs.items():
            IS2_atl08_attrs[gtx][att_name] = att_val
        for key,val in fileID[gtx]['land_ice_segments'].items():
            IS2_atl08_attrs[gtx]['land_ice_segments'][key] = {}
            for att_name,att_val in val.attrs.items():
                IS2_atl08_attrs[gtx]['land_ice_segments'][key][att_name] = att_val
            if isinstance(val, h5py.Group):
                for k,v in val.items():
                    IS2_atl08_attrs[gtx]['land_ice_segments'][key][k] = {}
                    for att_name,att_val in v.attrs.items():
                        IS2_atl08_attrs[gtx]['land_ice_segments'][key][k][att_name] = att_val

    #-- Closing the HDF5 file
    fileID.close()
    #-- Return the datasets and variables
    return (IS2_atl08_mds,IS2_atl08_attrs)