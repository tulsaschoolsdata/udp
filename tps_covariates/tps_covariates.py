"""
File: tps_covariates.py
Author: zachandfox
Date Created: 2019-10-31

Description: Pull together student data of interest, aggregated at the tract level:
    - Student count in each tract for each year

"""

from __tpsdata__ import project_directory, student_geos_path
import numpy as np
import pandas as pd

###### Initialize #######
start_year = 2010
end_year = 2020

###### Load Raw Data ######
student_geos_raw = {year: pd.read_csv(student_geos_path+'/student_geos_'+str(year)+'.csv') for year in range(start_year, end_year+1)}
#student_geos_raw = pd.read_csv(student_geos+'/student_geos_2010.csv')
#students_blocks_raw = pd.read_csv(project_directory+'/input/students_blocks.csv')

###### Clean Raw Data ######
student_tracts_list = {}
for year in range(start_year, end_year+1):
    ### Student Geographies ###
    temp = student_geos_raw[year].copy()
    # Create a 'tract' column #
    temp['tract'] = temp['COUNTYFP10']*10000 + temp['TRACTCE10']
    # Keep columns of interest #
    temp = temp[['studentid', 'tract', 'year']]
    # Save to list #
    student_tracts_list[year] = temp.copy()

# Combine to single dataframe #
student_tracts = pd.concat(student_tracts_list)

####### Analyze ########
### Student counts by tract, year ###
student_counts = student_tracts.groupby(['year', 'tract'])['studentid'].agg(['count']).reset_index(level=['year', 'tract'])
# Reshape wide #
student_counts_wide = student_counts.pivot(index='tract', columns='year', values='count').fillna(0).astype('int64')

###### Export ########
student_counts_wide.to_csv(project_directory+'output/tps_student_counts_by_tract/tps_student_counts_by_tract.csv')
