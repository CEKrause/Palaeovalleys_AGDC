# Directory_file_check.py

''' This code checks for files in each file folders. It reads in the case
study location names from a csv, and uses them to search through the prescribed folder
structure. Ensure that the csv file read into the regression analysis is the same as that
used here. 

'''

import pandas
import os

names = pandas.read_csv('/g/data/p25/cek156/case_study_sites_small.csv', delimiter = ',')

x = len(names)
for num in range(0,x-6):
    Studysite = names.ix[num]

    catname = '/g/data/p25/cek156/NDVI/' + Studysite.Name + '/NDVI_monthly_concat.nc'
    slopename = '/g/data/p25/cek156/NDVI/' + Studysite.Name + '/NDVI_slope.nc'
    pname = '/g/data/p25/cek156/NDVI/' + Studysite.Name + '/NDVI_pVal.nc'

    # Check for the monthly concat file
    if os.path.isfile(catname):
        print(Studysite.Name + ' has a concat file')    
    else:
        print(Studysite.Name + ' DOES NOT HAVE a concat file') 
    if os.path.isfile(slopename):
        print(Studysite.Name + ' has a slope file')
    if os.path.isfile(pname):
        print(Studysite.Name + ' has a p value file')
