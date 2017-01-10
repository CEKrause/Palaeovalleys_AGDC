# Palaeovalley_NDVI_linear_regression.py

'''
This code extracts Landsat red and NIR data and calculates NDVI for a specified case study region. 
This code handles large data areas by splitting up the queries to the AGDC into 1 month blocks, then 
pulling them all together to create monthly mean NDVI. 

The change in NDVI between two months (e.g. January - August) is calculated be performing a linear 
regression and recording the slope of the relationship. This indicates whether vegetation is "green-ing" 
through time. This can provide an indication of a groundwater supply. 

The data array and geotiff are output to /g/data for later use. 

Created by Claire Krause December 2016 Datacube version 2
Based on regression code from Neil Symington

**Code dependencies**
- csv file containing the bounding boxes for the case study site/s
- access to /g/data for writing out interim code
'''

import datacube
datacube.set_options(reproject_threads=1)
import xarray as xr
from datacube.storage import masking
from datacube.storage.masking import mask_to_dict
from matplotlib import pyplot as plt
import matplotlib.dates
import json
import pandas as pd
from IPython.display import display
import ipywidgets as widgets
import fiona
import shapely
import shapely.geometry
from shapely.geometry import shape
import rasterio
import pandas
from scipy import stats
import os
import numpy as np
import sys
from affine import Affine

dc = datacube.Datacube(app='dc-example')

# Set up the functions for the code
def geom_query(geom, geom_crs='EPSG:28352'):
    """
    Create datacube query snippet for geometry
    """
    return {
        'x': (geom.bounds[0], geom.bounds[2]),
        'y': (geom.bounds[1], geom.bounds[3]),
        'crs': geom_crs
    }

def warp_geometry(geom, crs_crs, dst_crs):
    """
    warp geometry from crs_crs to dst_crs
    """
    return shapely.geometry.shape(rasterio.warp.transform_geom(crs_crs, dst_crs, shapely.geometry.mapping(geom)))

def transect(data, geom, resolution, method='nearest', tolerance=None):
    """
    
    """
    dist = [i for i in range(0, int(geom.length), resolution)]
    points = zip(*[geom.interpolate(d).coords[0] for d in dist])
    indexers = {
        data.crs.dimensions[0]: list(points[1]),
        data.crs.dimensions[1]: list(points[0])        
        }
    return data.sel_points(xr.DataArray(dist, name='distance', dims=['distance']),
                           method=method,
                           tolerance=tolerance,
                           **indexers)

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

def iterate_end_date(start_date):
    """ 
    Using the starting date, this function returns an end date, calculated as two months from the starting date
    """
    # Split the string again so we can create the end date
    Year, Month, Day = start_date.split('-')
    # Now create the string for the end date
    Monthnum = int(Month)
    Monthnum = Monthnum + 2
    # Check whether we've gone into a new year
    if Monthnum == 13:
        Monthnum = 1
        Yearnum = int(Year)
        Yearnum = Yearnum + 1
        Year = str(Yearnum)
    if Monthnum == 14:
        Monthnum = 2
        Yearnum = int(Year)
        Yearnum = Yearnum + 1
        Year = str(Yearnum)
    # Change it back to a string
    Month = str(Monthnum)
    # And zero pad it so that there are two numbers
    Month = Month.zfill(2)
    # Now construct the end date
    end_date = Year + '-' + Month + '-' + Day
    return end_date, Year

def add_ds_info(new_ds, old_ds):
    """ 
    This function explicitly adds the coordinate reference and affine info to a new dataset, based on the info from the
    original dataset.
    """
    new_ds.attrs['affine'] = old_ds.affine
    new_ds.attrs['crs'] = old_ds.crs

def linear_regression_grid(input_array, mask_no_trend = True):
    '''
    This function applies a linear regression to a grid over a set time interval by looping through lat and lon 
    and calculating the linear regression through time for each pixel.
    '''

    ylen = len(input_array.y)
    xlen = len(input_array.x)
    from itertools import product
    coordinates = product(range(ylen), range(xlen))

    slopes = np.zeros((ylen, xlen))
    p_values = np.zeros((ylen, xlen))
    print('Slope shape is ', slopes.shape)

    for y, x in coordinates:
        val = input_array.isel(x = x, y = y)
        val[val<0] = np.nan

        # Check that we have at least three values to perform our linear regression on
        if np.count_nonzero(~np.isnan(val)) > 3:
            slopes[y, x], intercept, r_sq, p_values[y, x], std_err = stats.linregress(val.month,val)
        else:
            slopes[y, x] = np.nan
            intercept = np.nan
            r_sq = np.nan
            p_values[y, x] = np.nan

    #Get coordinates from the original xarray
    lat  = input_array.coords['y']
    long = input_array.coords['x']
    #Mask out values with insignificant trends (ie. p-value > 0.05) if user wants
    if mask_no_trend == True:
        slopes[p_values>0.05]=np.nan        
    # Write arrays into a x-array
    slope_xr = xr.DataArray(slopes, coords = [lat, long], dims = ['y', 'x'])
    p_val_xr = xr.DataArray(p_values, coords = [lat, long], dims = ['y', 'x']) 
    return slope_xr, p_val_xr

def month_cut(data, month_1, month_2):
    sliced_xr = list(data.groupby('month'))[monthDict[month_1]:monthDict[month_2] + 1]
    #Concatenate all the arrays into one xarray
    split_xr = sliced_xr[0][1]
    for i in range(int(monthDict[month_2]) - int(monthDict[month_1])):
        split_xr = xr.concat([split_xr, sliced_xr[i+1][1]], dim = 'month')      
    return split_xr

def write_geotiff(filename, dataset, time_index=None, profile_override=None):
    """
    Write an xarray dataset to a geotiff
    :attr bands: ordered list of dataset names
    :attr time_index: time index to write to file
    :attr dataset: xarray dataset containing multiple bands to write to file
    :attr profile_override: option dict, overrides rasterio file creation options.
    """
    profile_override = profile_override or {}
    dtypes = {val.dtype for val in dataset.data_vars.values()}
    assert len(dtypes) == 1  # Check for multiple dtypes
    profile = DEFAULT_PROFILE.copy()
    profile.update({
        'width': dataset.dims['x'],
        'height': dataset.dims['y'],
        'transform': Affine(*slope.affine[:6]),
        #'crs': dataset.crs.crs_str,
        'crs': dataset.crs,
        'count': len(dataset.data_vars),
        'dtype': str(dtypes.pop())
    })
    profile.update(profile_override)
    with rasterio.open(filename, 'w', **profile) as dest:
        for bandnum, data in enumerate(dataset.data_vars.values(), start=1):
            #dest.write(data.isel(time=time_index).data, bandnum)
            dest.write(data, bandnum)
            print ('Done')


### Begin the analysis ######################################################################

# Set up the case study bounding box (to make the file smaller and avoid memory errors)
names = pandas.read_csv('/g/data/p25/cek156/case_study_sites_small.csv', delimiter = ',')

# The looping of sites is done in the bash code, so we can just call on the passed
# argument for our site number

print(sys.argv)
num = int(sys.argv[1])
Studysite = names.ix[num]
print('Working on ' + Studysite.Name)

########################################################################################
########### Set up your inputs and data query ##########################################
########################################################################################

concat_output_name = '/g/data/p25/cek156/NDVI/' + Studysite.Name + '/NDVI_monthly_concat.nc'

#Define time interval and months range for linear regression
month_1 = 'July'
#f you want to only look at one month then simply make month_1 equal to month_2
month_2 = 'December'

slope_output_name = '/g/data/p25/cek156/NDVI/' + Studysite.Name + '/' + Studysite.Name + '_NDVI_slope_JulDec.nc'
pval_output_name = '/g/data/p25/cek156/NDVI/' + Studysite.Name + '/' + Studysite.Name + '_NDVI_pVal_JulDec.nc'
geotiff_output_name = '/g/data/p25/cek156/NDVI/' + Studysite.Name + '/' + Studysite.Name + '_NDVI_drying_trend_JulDec.tiff'

##############################################################################################
### You shouldn't need to change anything below here #########################################
##############################################################################################

print("Let's open the concat file")

file_check = os.path.isfile(concat_output_name)
if file_check == True:

    ndvi_monthly = xr.open_dataset(concat_output_name)
    affine = ndvi_monthly.attrs['affine']
    crs = ndvi_monthly.attrs['crs']
    ndvi_monthly = ndvi_monthly.ndvi

    print('Now to the rest of the code...')

    #Define a dictionary for months
    monthDict = {'January':0, 'February':1, 'March':2, 'April':3, 'May':4, 'June':5, 'July':6, 'August':7, 'September':8,
               'October':9, 'November':10, 'December':11}

    # We need to check for slope and pval files
    file_check1 = os.path.isfile(slope_output_name)
    file_check2 = os.path.isfile(pval_output_name)

    if file_check1 & file_check2 == True:
        print ('Slope and pval files for Jul - Dec already created')
    else:
        #Split pull out all of the months of interest using the function defined above
        split_data = month_cut(ndvi_monthly, month_1, month_2)
        #Now perform the linear regression
        print('Calculating slope and p values')
        slope_xr, p_val_xr = linear_regression_grid(split_data, mask_no_trend = True)

        # Save the outputs so that we don't need to keep running this script
        ds1 = p_val_xr.to_dataset(name='p_val')
        ds1.attrs['affine'] = affine
        ds1.attrs['crs'] = crs
        output_filename = pval_output_name
        ds1.to_netcdf(path = output_filename, mode = 'w')
        print('wrote %s' % output_filename)

        ds2 = slope_xr.to_dataset(name='slope')
        ds2.attrs['affine'] = affine
        ds2.attrs['crs'] = crs
        output_filename = slope_output_name
        ds2.to_netcdf(path = output_filename, mode = 'w')
        print('wrote %s' % output_filename)

    # Check for geotiff file
    tiff_check = os.path.isfile(geotiff_output_name)
    if tiff_check == True:
        print('Geotiff for Jul - Dec already there')
    else:
        slope = xr.open_dataset(slope_output_name)
        #Write the files out into a tif file for viewing in GIS
        #You may need to adjust the default_profile values so that blockysize is not larger than the xr spatial dimensions
        outfile = geotiff_output_name

        #Function below is from https://github.com/data-cube/agdc-v2/blob/develop/datacube/helpers.py
        DEFAULT_PROFILE = {
           #'blockxsize': 128,
           #'blockysize': 128,
           'compress': 'lzw',
           'driver': 'GTiff',
           'interleave': 'band',
           'nodata': 0.0,
           'photometric': 'RGBA',
           'tiled': True}

        write_geotiff(outfile, slope)
    print('finished with ' + Studysite.Name)

else:
    print('No concat file yet...')
