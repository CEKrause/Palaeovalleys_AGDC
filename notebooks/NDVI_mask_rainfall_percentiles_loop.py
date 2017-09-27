# NDVI_mask_rainfall_percentiles_loop.py
'''
This code uses a CSV input of case study site bounding boxes and known palaeovalley locations from an arcGIS polygon layer to determine if there is a 
statistically significant difference between the data within and outside of the valley boundaries. The polygons are used as a mask to separate the
data into two datasets for each bounding box, and their statistical difference is determined by a t-tesk and KS test. 

Code by Claire Krause, December 2016. Datacube version 1.1.

Code Dependencies
- csv of bounding boxes for study locations
- polygon layer of known palaeovalley sites
'''

# Import the libraries we need in the code and tell matplotlib to display the plots here
import fiona
import shapely.geometry
import rasterio
import rasterio.features
import datacube
import numpy as np
from datacube.storage import masking
import matplotlib.pyplot as plt
import xarray as xr
import scipy.stats
import pandas 
import csv
import geopandas as gp

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
    
def return_good_pixels(data, sensor_pq, start_date, end_date):
    """
    This function uses pixel quality information to mask out and remove pixel quality artifacts from extracted data.
    """
    #Define which pixel quality artefacts you want removed from the datacube results
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
    #grab the projection info before masking/sorting
    crs = data.crs
    crswkt = data.crs.wkt
    affine = data.affine
    #Apply the PQ masks to the NDVI
    cloud_free = masking.make_mask(sensor_pq, **mask_components)
    good_data = cloud_free.pixelquality.loc[start_date:end_date]
    quality_data = data.where(good_data)
    return quality_data

def season_end_date(start_date):
    """ 
    Using the starting date, this function returns an end date, calculated as three months from the starting date
    """
    # Split the string again so we can create the end date
    Year, Month, Day = start_date.split('-')
    # Now create the string for the end date - if it's December, make it 3, else just add 3 months
    Monthnum = int(Month)
    if Monthnum == 12:
        Monthnum = 3
        # And increment the year
        Year = int(Year)
        Year = Year + 1
        Year = str(Year)
    else:
        Monthnum = Monthnum + 3
    # Change it back to a string
    Month = str(Monthnum)
    # And zero pad it so that there are two numbers
    Month = Month.zfill(2)
    # Now construct the end date
    end_date = Year + '-' + Month + '-' + Day
    return end_date

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

def find_bomnum_index(num):
    '''
    Because we have cut up the bounding boxes, there is no longer a relationship between the location of a site in 
    our rainfall data csv and bounding box csv. We will set up an index so that it's still correct
    '''
    if 0 <= num <=3:
        bomnum = 0
        print('Using data from' + wetdry_times.name[bomnum])
    elif 4 <= num <=5:
        bomnum = 1
        print('Using data from' + wetdry_times.name[bomnum])
    elif 6 <= num <=23:
        bomnum = 2
        print('Using data from' + wetdry_times.name[bomnum])
    elif 24 <= num <=27:
        bomnum = 3
        print('Using data from' + wetdry_times.name[bomnum])
    elif 28 <= num <=39:
        bomnum = 4
        print('Using data from' + wetdry_times.name[bomnum])
    elif num ==40:
        bomnum = 5
        print('Using data from' + wetdry_times.name[bomnum])
    elif 41 <= num <=42:
        bomnum = 6
        print('Using data from' + wetdry_times.name[bomnum])
    elif num ==43:
        bomnum = 7
        print('Using data from' + wetdry_times.name[bomnum])
    elif 44 <= num <=45:
        bomnum = 8
    return bomnum

def write_to_csv(times, OUTPUT_path, row):
    if times == 1 and Studysite.Name == 'Blackwood2A':
        with open(OUTPUT_path,'w') as csvFile:
            writer = csv.writer(csvFile)
            header = ['name', 'start_date', 'ttest', 'KS_test']
            writer.writerow(header)
            writer.writerow(row)
    else:
        with open(OUTPUT_path,'a') as csvFile:
           writer = csv.writer(csvFile)
           writer.writerow(row)

#########################################################################################
######################### Set up the code to run here ###################################
#########################################################################################

# set up the paths to the input files
Case_study_file = '/g/data/p25/cek156/case_study_sites_small.csv'
rainfall_percentiles = '/g/data/p25/cek156/BOM_site_rainfall.csv'
polygon_path = '/g/data/p25/cek156/Palaeovalleys_2012.shp'

print('input paths set')

# Set up the paths to the output files
OUTPUT_wet = '/g/data/p25/cek156/NDVI/NDVI_wet.csv'
OUTPUT_dry = '/g/data/p25/cek156/NDVI/NDVI_dry.csv'
print('output paths set')

# Read in the input files
names = pandas.read_csv(Case_study_file, delimiter = ',')
wetdry_times = pandas.read_csv(rainfall_percentiles, delimiter = ',')
shp = gp.GeoDataFrame.from_file(polygon_path)
print('files read in')

############################################################################################
############## Work on the wet years first #################################################
############################################################################################

# Work out how many sites we want to analyse, and set up a list of numbers we can loop through
x = len(names.index)
iterable = list(range(0,x-6)) 
# NB -6 because we don't have pv polygons for Daintree, Laura or Blackwood1, and we don't want to run the testing site

for num in iterable:
    Studysite = names.ix[num]
    print ('Working on ' + Studysite.Name)
    # Because we have cut up the bounding boxes, there is no longer a relationship between the location of a site in 
    # our rainfall data csv and bounding box csv. We will set up an index so that it's still correct
    bomnum = find_bomnum_index(num)
    print('Using data from' + wetdry_times.name[bomnum])
    # Now wet years...
    wet_date = wetdry_times.wet[bomnum]
    wet_date = wet_date.split("'")
    xx = len(wet_date)-1
    wet_times = list(range(1,xx,2))
    for times in wet_times:
        # Create a bounding box from the locations specified above
        box = shapely.geometry.box(names.minlon[num], names.minlat[num], names.maxlon[num], names.maxlat[num], ccw = True)
        # Only get the polygons that intersect the bounding box (i.e. remove all the irrelevant ones)
        filtered = shp.where(shp.intersects(box)).dropna()
        # Combine all of the relevant polygons into a single polygon
        shp_union = shapely.ops.unary_union(filtered.geometry)

        # Set up the start and end times
        start_date = wet_date[times]
        end_date = season_end_date(start_date)

        # Set up the query for the datacube extraction
        #Define wavelengths/bands of interest, remove this kwarg to retrieve all bands
        bands_of_interest = [#'blue',
                         #'green',
                         'red', 
                         'nir',
                         #'swir1', 
                         #'swir2',
                             ]
        #Define sensors of interest
        sensor = ['ls8', 'ls7', 'ls5'] 
        # And pull together our bounding box and start and end times 
        query = {'time': (start_date, end_date),
                 'lat': (names.maxlat[num], names.minlat[num]), 
                 'lon': (names.minlon[num], names.maxlon[num]) }

        # Grab NDVI data from the datacube and filter it for pixel quality
        dc = datacube.Datacube(app='poly-drill-recipe')
        sensor_all = len(sensor)
        sens = 1
        data_check = False
        for sensor_iterable in range (0,sensor_all):
        #Retrieve the data and pixel quality (PQ) data for sensor n
             nbar = dc.load(product = sensor[sensor_iterable] +'_nbar_albers', group_by='solar_day', measurements = bands_of_interest,  **query)
             if nbar :    
                 data_check = True
                 pq = dc.load(product = sensor[sensor_iterable] +'_pq_albers', group_by='solar_day', fuse_func=pq_fuser, **query)
                 crs = nbar.crs.wkt
                 affine = nbar.affine
                 geobox = nbar.geobox
                 # Filter the data to remove bad pixels
                 nbar = return_good_pixels(nbar, pq)
                 nbar['ndvi'] = (nbar['nir'] - nbar['red']) / (nbar['nir'] + nbar['red'])
                 # Calculate the mean for each sensor, then later, the mean for all sensors.
                 nbar = nbar.mean(dim = 'time')
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
                 print('sens = ', sens)
                 print(start_date)
                 if sens >= 3:
                     allsens = allsens.mean(dim = 'new_dim') 
        if data_check == True:
            # Create the mask based on our shapefile
            mask = geometry_mask(warp_geometry(shp_union, shp.crs, crs), geobox, invert=True)
            # Get data only where the mask is 'true'
            data_masked = allsens.where(mask)
            # Get data only where the mask is 'false'
            data_maskedF = allsens.where(~ mask)

            # Create a new numpy array with just the ndvi values
            data_masked2 = np.array(data_masked.ndvi)
            data_maskedF2 = np.array(data_maskedF.ndvi)
            # Grab only values that aren't nan
            data_masked_nonan = data_masked2[~np.isnan(data_masked2)]
            data_maskedF_nonan = data_maskedF2[~np.isnan(data_maskedF2)]
            masked_both = [data_masked_nonan,data_maskedF_nonan]

            # Test with an unpaired t-test. Equal var is false because our populations are different sizes
            # Our null hypothesis that 2 independent samples are drawn from the same continuous distribution
            stats_ttest, ttestpval = scipy.stats.ttest_ind(data_masked_nonan,data_maskedF_nonan, equal_var = 'False')

            # Test with a Kolmogorov-Smirnov test 
            # Our null hypothesis that 2 independent samples are drawn from the same continuous distribution
            stats_KS, KSpval = scipy.stats.ks_2samp(data_masked_nonan,data_maskedF_nonan)

            # Pull together everything we want in our csv file
            row = [Studysite.Name, start_date, stats_ttest, stats_KS]
            # Write our stats to a csv file so we can compare them later
            # If this is the first site, make a new file, otherwise, append the existing file
            print('writing to csv')
            write_to_csv(times, OUTPUT_wet, row)
        # Or if there is no data...
        else:
            print('writing no data to csv')
            row = [Studysite.Name, start_date, 'nan', 'nan']
            write_to_csv(times, OUTPUT_wet, row)
    nbar = None
    pq = None
    data_masked_nonan = None
    data_maskedF_nonan = None
    allsens = None

############################################################################################
############## Work on the dry years #######################################################
############################################################################################

# Work out how many sites we want to analyse, and set up a list of numbers we can loop through
x = len(names.index)
iterable = list(range(0,x-6)) 
# NB -6 because we don't have pv polygons for Daintree, Laura or Blackwood1, and we don't want to run the testing site

for num in iterable:
    Studysite = names.ix[num]
    print ('Working on ' + Studysite.Name)
    # Because we have cut up the bounding boxes, there is no longer a relationship between the location of a site in 
    # our rainfall data csv and bounding box csv. We will set up an index so that it's still correct
    bomnum = find_bomnum_index(num)
    print('Using data from' + wetdry_times.name[bomnum])
    # Now dry years...
    dry_date = wetdry_times.dry[bomnum]
    dry_date = dry_date.split("'")
    xx = len(dry_date)-1
    dry_times = list(range(1,xx,2))
    for times in dry_times:
        # Create a bounding box from the locations specified above
        box = shapely.geometry.box(names.minlon[num], names.minlat[num], names.maxlon[num], names.maxlat[num], ccw = True)
        # Only get the polygons that intersect the bounding box (i.e. remove all the irrelevant ones)
        filtered = shp.where(shp.intersects(box)).dropna()
        # Combine all of the relevant polygons into a single polygon
        shp_union = shapely.ops.unary_union(filtered.geometry)

        # Set up the start and end times
        start_date = dry_date[times]
        end_date = season_end_date(start_date)

        # Set up the query for the datacube extraction
        #Define wavelengths/bands of interest, remove this kwarg to retrieve all bands
        bands_of_interest = [#'blue',
                         #'green',
                         'red', 
                         'nir',
                         #'swir1', 
                         #'swir2',
                             ]
        #Define sensors of interest
        sensor = ['ls8', 'ls7', 'ls5'] 
        # And pull together our bounding box and start and end times 
        query = {'time': (start_date, end_date),
                 'lat': (names.maxlat[num], names.minlat[num]), 
                 'lon': (names.minlon[num], names.maxlon[num]) }

        # Grab NDVI data from the datacube and filter it for pixel quality
        dc = datacube.Datacube(app='poly-drill-recipe')
        sensor_all = len(sensor)
        sens = 1
        data_check = False
        for sensor_iterable in range (0,sensor_all):
        #Retrieve the data and pixel quality (PQ) data for sensor n
             nbar = dc.load(product = sensor[sensor_iterable] +'_nbar_albers', group_by='solar_day', measurements = bands_of_interest,  **query)
             if nbar :    
                 data_check = True
                 pq = dc.load(product = sensor[sensor_iterable] +'_pq_albers', group_by='solar_day', fuse_func=pq_fuser, **query)
                 crs = nbar.crs.wkt
                 affine = nbar.affine
                 geobox = nbar.geobox
                 # Filter the data to remove bad pixels
                 nbar = return_good_pixels(nbar, pq)
                 nbar['ndvi'] = (nbar['nir'] - nbar['red']) / (nbar['nir'] + nbar['red'])
                 # Calculate the mean for each sensor, then later, the mean for all sensors.
                 nbar = nbar.mean(dim = 'time')
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
                 print('sens = ', sens)
                 print(start_date)
                 if sens >= 3:
                     allsens = allsens.mean(dim = 'new_dim') 
        if data_check == True:
            # Create the mask based on our shapefile
            mask = geometry_mask(warp_geometry(shp_union, shp.crs, crs), geobox, invert=True)
            # Get data only where the mask is 'true'
            data_masked = allsens.where(mask)
            # Get data only where the mask is 'false'
            data_maskedF = allsens.where(~ mask)

            # Create a new numpy array with just the ndvi values
            data_masked2 = np.array(data_masked.ndvi)
            data_maskedF2 = np.array(data_maskedF.ndvi)
            # Grab only values that aren't nan
            data_masked_nonan = data_masked2[~np.isnan(data_masked2)]
            data_maskedF_nonan = data_maskedF2[~np.isnan(data_maskedF2)]
            masked_both = [data_masked_nonan,data_maskedF_nonan]

            # Test with an unpaired t-test. Equal var is false because our populations are different sizes
            # Our null hypothesis that 2 independent samples are drawn from the same continuous distribution
            stats_ttest, ttestpval = scipy.stats.ttest_ind(data_masked_nonan,data_maskedF_nonan, equal_var = 'False')

            # Test with a Kolmogorov-Smirnov test 
            # Our null hypothesis that 2 independent samples are drawn from the same continuous distribution
            stats_KS, KSpval = scipy.stats.ks_2samp(data_masked_nonan,data_maskedF_nonan)

            # Pull together everything we want in our csv file
            row = [Studysite.Name, start_date, stats_ttest, stats_KS]
            # Write our stats to a csv file so we can compare them later
            # If this is the first site, make a new file, otherwise, append the existing file
            print('writing to csv')
            write_to_csv(times, OUTPUT_dry, row)
        # Or if there is no data...
        else:
            print('writing no data to csv')
            row = [Studysite.Name, start_date, 'nan', 'nan']
            write_to_csv(times, OUTPUT_dry, row)
    nbar = None
    pq = None
    data_masked_nonan = None
    data_maskedF_nonan = None
    allsens = None
