# Extract_AGDC_for_study_sites_raijin.py

'''
This code extracts all available measurement data for each of a list of case study sites, then saves the time averaged output to a netcdf file for later use. This code was designed 
to minimise the amount of time required to perform analyses with datacube data over known areas. By creating the netcdf files of each measurement, these can be called in by another 
program and quickly analysed to look for patterns.

Note that the netcdf files should eventually be deleted once the further analyses have been conducted to avoid duplication of data, and to save memory on /g/data.

Created by Claire Krause January 2017 Datacube version 1.1.13

Dependancies in this code:
- csv file with the lat/lon coordinates of the case study bounding box/es

Accompanying code
- Extract_AGDC_for_study_sites.ipynb - steps through this code with explanations of how this code is compiled. See the acompanying code for more detailed explanations and
examples
- Run_AGDC_extraction - PBS submission code to generate single CPU jobs for each study site
'''

# Import the libraries we need in the code and tell matplotlib to display the plots here
import fiona
import shapely.geometry
import rasterio
import rasterio.features
import datacube
datacube.set_options(reproject_threads=1)
import numpy as np
from datacube.storage import masking
import matplotlib.pyplot as plt
import xarray as xr
import scipy.stats
import pandas
import os 
import sys
from affine import Affine

# Set up some functions to use later in the code
def warp_geometry(geom, src_crs, dst_crs):
    """
    warp geometry from src_crs to dst_crs
    """
    return shapely.geometry.shape(rasterio.warp.transform_geom(src_crs, dst_crs, shapely.geometry.mapping(geom)))

def geometry_mask(geom, geobox, all_touched=False, invert=False):
    """
    rasterize geometry into a binary mask where pixels that overlap geometry are False
    """
    return rasterio.features.geometry_mask([geom],
                                           out_shape=geobox.shape,
                                           transform=geobox.affine,
                                           all_touched=all_touched,
                                           invert=invert)

def pq_fuser(dest, src):
    valid_bit = 8
    valid_val = (1 << valid_bit)

    no_data_dest_mask = ~(dest & valid_val).astype(bool)
    np.copyto(dest, src, where=no_data_dest_mask)

    both_data_mask = (valid_val & dest & src).astype(bool)
    np.copyto(dest, src & dest, where=both_data_mask)

def return_good_pixels(nbar, pq):
    """
    This function uses pixel quality information to mask out and remove pixel quality artifacts from extracted data.
    """
    mask_components = {'cloud_acca':'no_cloud',
    'cloud_shadow_acca' :'no_cloud_shadow',
    'cloud_shadow_fmask' : 'no_cloud_shadow',
    'cloud_fmask' :'no_cloud',
    'blue_saturated' : False,
    'green_saturated' : False,
    'red_saturated' : False,
    'nir_saturated' : False,
    'swir1_saturated' : False,
    'swir2_saturated' : False,
    'contiguous':True}
    pqmask = masking.make_mask(pq.pixelquality,  **mask_components)
    return nbar.where(pqmask)

##############################################################################################

names = pandas.read_csv('/g/data/p25/cek156/case_study_sites_small.csv', delimiter = ',')

print(sys.argv)
num = int(sys.argv[1])
Studysite = names.ix[num]
print('Working on ' + Studysite.Name)

# Pull in some data from the datacube to apply our mask to
dc = datacube.Datacube(app='poly-drill-recipe')

#Define wavelengths/bands of interest, remove this kwarg to retrieve all bands
bands_of_interest = ['blue',
                     'green',
                     'red', 
                     'nir',
                     'swir1', 
                     'swir2',
                     ]

#Define sensors of interest
sensors = ['ls8', 'ls7', 'ls5'] 

OUTPUT_dir = '/g/data/p25/cek156/' + Studysite.Name

# Set up AGDC extraction query
query = {'lat': (names.maxlat[num], names.minlat[num]), 
         'lon': (names.minlon[num], names.maxlon[num]),
         'resolution': (-250, 250)}

# Loop through the bands of interest
for band in enumerate(bands_of_interest):
    print('Working on ' + band[1])

    OUTPUT_file = '/g/data/p25/cek156/' + Studysite.Name + '/' + Studysite.Name + '_' + band[1] + '_time_mean.nc'

    # Check if this file already exists, and if so, skip this iteration
    file_check = file_check1 = os.path.isfile(OUTPUT_file)
    if file_check == True:
        print(band[1] + ' file aready created. Moving onto the next one...')
    elif file_check == False:
        # Grab data from all three sensors (ls5, ls7 and ls8). We will loop through the three sensors, then calculate an average.
        for idx, sensor in enumerate(sensors):
            #Retrieve the data and pixel quality (PQ) data for sensor n
            sens = 1
            nbar = dc.load(product = sensor +'_nbar_albers', group_by='solar_day', measurements = [band[1]],  **query)
            if nbar :    
                pq = dc.load(product = sensor +'_pq_albers', group_by='solar_day', fuse_func=pq_fuser, **query)
                crs = nbar.crs.wkt
                affine = nbar.affine
                geobox = nbar.geobox
                # Filter the data to remove bad pixels
                nbar = return_good_pixels(nbar, pq)
                print('Finished grabbing data for ' + sensor)
                if sens == 1:
                    allsens = nbar
                    sens = sens + 1
                elif sens == 2:
                    allsens = xr.concat([allsens, nbar], dim = 'new_dim')
                    sens = sens + 1
                else:
                    nbar = xr.concat([nbar], dim = 'new_dim')
                    allsens = xr.concat([allsens, nbar], dim = 'new_dim')
                    sens = sens + 1                 
        if sens >= 3:
            datamean = allsens.mean(dim = 'new_dim') 
            datamean = datamean.mean(dim = 'time')
        else:
            datamean = allsens.mean(dim = 'time')

        ## Save the time-mean, all sensor mean data into a netcdf file for later use.
        # Check whether the Study site directory has been created, and if not, create it.
        dir_check = os.path.isdir(OUTPUT_dir)
        if dir_check == False:
            os.makedirs(OUTPUT_dir)

        # Save the outputs so that we don't need to keep running this script
        datamean.attrs['affine'] = affine
        datamean.attrs['crs'] = crs
        #ds.attrs['geobox'] = geobox
        datamean.to_netcdf(path = OUTPUT_file, mode = 'w')
        print('Saved ' + band[1] + ' averages for ' + Studysite.Name)
print('Finished with ' + Studysite.Name)
