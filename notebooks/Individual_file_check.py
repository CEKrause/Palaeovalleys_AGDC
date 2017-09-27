# Individual_file_check.py

''' This code checks for files in the individual files folders. It reads in the case
study location names from a csv, and uses them to search through the prescribed folder
structure. Ensure that the csv file read into the regression analysis is the same as that
used here. 

If you do not want to delete the individual files, but just find them, comment out the last
four lines of this code.
'''

import pandas
import os

names = pandas.read_csv('/g/data/p25/cek156/case_study_sites_small.csv', delimiter = ',')

x = len(names)
for num in range(0,x-6):
    Studysite = names.ix[num]
    print('Working on ' + Studysite.Name)

    pathname = '/g/data/p25/cek156/NDVI/' + Studysite.Name + '/individual'

    # Check for any individual files from previous runs
    files = os.listdir(pathname)
    if files == []:
        print(Studysite.Name + ' file clean')
    else:
        #print(Studysite.Name + ' still has individual files')
        print('Cleaning up old individual files for ' + Studysite.Name)
        os.chdir(pathname)
        os.system('rm *')
        print(Studysite.Name + 'file clean')
  
